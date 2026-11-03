"""Orchestrate capture, online vision, utterance STT and bounded LLM workers.

The main thread owns visual inference and the optional desktop window. Capture,
segmentation, STT and language inference each have one producer/consumer thread.
All persisted events are compact; raw camera frames and PCM stay in memory only.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from copy import deepcopy
from dataclasses import asdict
from math import ceil
from pathlib import Path
from typing import Any, Callable

from care_hero.realtime.audio import SpeechSegmenter
from care_hero.realtime.capture import CapturedFrame
from care_hero.realtime.channels import BoundedChannel
from care_hero.realtime.events import EventJournal
from care_hero.realtime.contracts import AudioSource, FrameAnalyzer, FrameSource, TurnTranscriber
from care_hero.realtime.language import StructuredLanguageModel
from care_hero.realtime.metrics import OnlineMetrics
from care_hero.realtime.settings import RealtimeSettings
from care_hero.realtime.vision import draw_preview
from care_hero.scoring import ScenarioRules


class RealtimeSession:
    def __init__(
        self, settings: RealtimeSettings, rules: ScenarioRules, *,
        source: FrameSource | None, analyzer: FrameAnalyzer, output: Path,
        microphone: AudioSource | None = None,
        transcriber_factory: Callable[[], TurnTranscriber] | None = None,
        language_model: StructuredLanguageModel | None = None,
        preview: bool = False, synthetic: bool = False,
    ) -> None:
        self.settings, self.rules = settings, rules
        self.source, self.analyzer = source, analyzer
        self.microphone, self.transcriber_factory = microphone, transcriber_factory
        self.language_model, self.preview, self.synthetic = language_model, preview, synthetic
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if any((self.output / name).exists() for name in ("events.jsonl", "windows.jsonl", "session.json")):
            raise FileExistsError("Choose a new output directory for each session.")
        self.events = EventJournal(self.output / "events.jsonl")
        self.video = BoundedChannel(settings.video_queue_size, replace_oldest=True)
        self.audio = BoundedChannel(settings.audio_queue_size, replace_oldest=True)
        self.speech = BoundedChannel(settings.utterance_queue_size, replace_oldest=False)
        self.language = BoundedChannel(settings.llm_queue_size, replace_oldest=False)
        self.stop = threading.Event()
        self.capture_done, self.audio_done = threading.Event(), threading.Event()
        self.segment_done, self.stt_done, self.language_done = (threading.Event() for _ in range(3))
        self.frozen = threading.Event()
        self._lock = threading.Lock()
        self._latest: dict[str, Any] = {}
        history_seconds = settings.window_seconds + settings.audio_queue_size * settings.audio.block_seconds + 2
        self._snapshots: deque[dict] = deque(maxlen=ceil(history_seconds / settings.snapshot_interval_seconds) + 2)
        self._turns: list[dict] = []
        self._threads: list[threading.Thread] = []
        self._capture_error = False
        self._epoch = 0.0
        self._captured = 0
        self._audio_active = False
        self.metrics = OnlineMetrics(settings, rules)

    def _spawn(self, target, name: str) -> None:
        thread = threading.Thread(target=target, name=f"care-hero-{name}", daemon=True)
        self._threads.append(thread)
        thread.start()

    def _capture(self) -> None:
        index = 0
        try:
            while not self.stop.is_set() and time.monotonic() - self._epoch < self.settings.duration_seconds:
                tick = time.monotonic()
                ok, image = self.source.read()
                if not ok:
                    self._capture_error = True
                    self.events.emit("capture", "error", code="capture_read_failed")
                    break
                self.video.put(CapturedFrame(index, time.monotonic() - self._epoch, image))
                index += 1
                self._captured = index
                # Also pace file replay and synthetic sources; avoid a hot read loop.
                self.stop.wait(max(0.0, 1 / 30 - (time.monotonic() - tick)))
        except Exception as exc:
            self._capture_error = True
            self.events.emit("capture", "error", code=type(exc).__name__)
        finally:
            self.capture_done.set()
            self.events.emit("capture", "error" if self._capture_error else "done", processed=index, dropped=self.video.dropped)

    def _submit_utterance(self, utterance) -> None:
        with self._lock:
            # Audio endpointing may itself lag behind camera analysis. Select by
            # media time, not by when this Python worker happened to run.
            context = next((deepcopy(item) for item in reversed(self._snapshots)
                            if item["window_end_seconds"] is not None
                            and item["window_end_seconds"] <= utterance.end_seconds), {})
        accepted = self.speech.put((utterance, context))
        self.events.emit("segmentation", "completed" if accepted else "dropped",
                         utterance_id=utterance.identifier, start_seconds=utterance.start_seconds,
                         end_seconds=utterance.end_seconds, reason=utterance.reason,
                         queued=self.speech.size, dropped=self.speech.dropped)

    def _segment_audio(self) -> None:
        segmenter = SpeechSegmenter(self.settings.audio)
        processed = 0
        try:
            while not self.frozen.is_set() and (not self.audio_done.is_set() or self.audio.size):
                try:
                    block = self.audio.get()
                except queue.Empty:
                    continue
                previous = segmenter.discontinuities
                for utterance in segmenter.push(block):
                    self._submit_utterance(utterance)
                if segmenter.discontinuities > previous:
                    self.events.emit("audio", "discontinuity", resets=segmenter.discontinuities, dropped_blocks=self.audio.dropped)
                processed += 1
            tail = segmenter.flush()
            if tail is not None:
                self._submit_utterance(tail)
        except Exception as exc:
            self.events.emit("segmentation", "error", code=type(exc).__name__)
        finally:
            self.segment_done.set()
            self.events.emit("segmentation", "done", processed_blocks=processed, discontinuities=segmenter.discontinuities)

    def _transcribe(self, backend) -> None:
        count = 0
        try:
            while not self.frozen.is_set() and (not self.segment_done.is_set() or self.speech.size):
                try:
                    utterance, context = self.speech.get()
                except queue.Empty:
                    continue
                started = time.monotonic()
                self.events.emit("stt", "running", utterance_id=utterance.identifier, queued=self.speech.size)
                record = {
                    "utterance_id": utterance.identifier,
                    "start_seconds": utterance.start_seconds,
                    "end_seconds": utterance.end_seconds,
                    "vision_context": context,
                    "vision_context_age_seconds": (
                        max(0.0, utterance.end_seconds - context["window_end_seconds"])
                        if context.get("window_end_seconds") is not None else None
                    ),
                    "llm_status": "skipped" if self.language_model is None else "pending",
                }
                try:
                    transcript = backend.transcribe(utterance)
                    record["transcription"] = transcript.to_dict()
                    record["stt_status"] = "success"
                    if self.language_model is not None and transcript.full_text.strip():
                        if not self.language.put((record, transcript.full_text)):
                            record["llm_status"] = "dropped"
                            self.events.emit("llm_turn", "dropped", utterance_id=utterance.identifier, dropped=self.language.dropped)
                    elif self.language_model is not None:
                        record["llm_status"] = "no_speech"
                except Exception as exc:
                    record.update(stt_status="error", error_code=type(exc).__name__, llm_status="skipped")
                record["stt_latency_ms"] = round((time.monotonic() - started) * 1000, 2)
                with self._lock:
                    if not self.frozen.is_set():
                        self._turns.append(record)
                count += 1
                self.events.emit("stt", record["stt_status"], utterance_id=utterance.identifier,
                                 processed=count, latency_ms=record["stt_latency_ms"],
                                 output_age_seconds=round(time.monotonic() - self._epoch - utterance.end_seconds, 3))
        finally:
            self.stt_done.set()
            self.events.emit("stt", "done", processed=count)

    def _interpret(self) -> None:
        history: list[dict] = []
        count = 0
        try:
            while not self.frozen.is_set() and (not self.stt_done.is_set() or self.language.size):
                try:
                    record, transcript = self.language.get()
                except queue.Empty:
                    continue
                started = time.monotonic()
                self.events.emit("llm_turn", "running", utterance_id=record["utterance_id"], queued=self.language.size)
                try:
                    parsed = self.language_model.interpret_turn(transcript, record["vision_context"], history)
                    update = {"interpretation": parsed.model_dump(), "llm_status": "success"}
                    history.append(parsed.model_dump())
                    history = history[-max(1, self.settings.llm.history_turns):]
                except Exception as exc:
                    update = {"llm_status": "error", "llm_error_code": type(exc).__name__}
                update["llm_latency_ms"] = round((time.monotonic() - started) * 1000, 2)
                with self._lock:
                    if not self.frozen.is_set():
                        record.update(update)
                count += 1
                self.events.emit("llm_turn", update["llm_status"], processed=count,
                                 utterance_id=record["utterance_id"], latency_ms=update["llm_latency_ms"])
        finally:
            self.language_done.set()

    def run(self) -> dict:
        failure = None
        snapshot_count = 0
        stopped_by_user = False
        capture_elapsed = 0.0
        self.events.emit("session", "initializing", synthetic=self.synthetic)
        try:
            with self.analyzer:
                self.events.emit("vision", "ready")
                backend = None
                if self.microphone is not None:
                    try:
                        self.events.emit("stt", "initializing")
                        backend = self.transcriber_factory()
                        self.events.emit("stt", "ready")
                    except Exception as exc:
                        self.events.emit("audio", "error", code=type(exc).__name__)
                        failure = "audio_initialization_failed"
                self._epoch = time.monotonic()
                if self.microphone is not None and backend is not None:
                    try:
                        self.microphone.start(self.audio, self._epoch)
                        self._audio_active = True
                        self._spawn(self._segment_audio, "segment")
                        self._spawn(lambda: self._transcribe(backend), "stt")
                        if self.language_model is not None:
                            self._spawn(self._interpret, "language")
                        self.events.emit("audio", "running", sample_rate=self.settings.audio.sample_rate)
                    except Exception as exc:
                        failure = "microphone_start_failed"
                        self.events.emit("audio", "error", code=type(exc).__name__)
                if not self._audio_active:
                    for flag in (self.audio_done, self.segment_done, self.stt_done, self.language_done):
                        flag.set()
                    self.events.emit("stt", "skipped" if self.microphone is None else "error")
                    if self.source is None:
                        raise RuntimeError("no_active_input")
                if self.language_model is None:
                    self.language_done.set()
                    self.events.emit("llm_turn", "skipped")
                if self.source is not None:
                    self.source.open()
                    self.events.emit("capture", "running")
                    self._spawn(self._capture, "capture")
                else:
                    self.capture_done.set()
                    self.events.emit("capture", "skipped")

                next_analysis = 0.0
                next_snapshot = 0.0
                with (self.output / "windows.jsonl").open("x", encoding="utf-8") as windows:
                    while not self.stop.is_set():
                        elapsed = time.monotonic() - self._epoch
                        if elapsed >= self.settings.duration_seconds:
                            break
                        if self.source is not None and self.capture_done.is_set() and not self.video.size:
                            break
                        if self.source is None:
                            self.stop.wait(0.02)
                        elif elapsed >= next_analysis:
                            try:
                                packet = self.video.get()
                            except queue.Empty:
                                continue
                            before = time.monotonic()
                            frame, image, timings = self.analyzer.analyze(packet)
                            height, width = image.shape[:2]
                            self.metrics.update(frame, width, height)
                            self.events.emit("vision", "running", processed=self.metrics.frame_count,
                                             pose_detected=frame.detected,
                                             face_status=frame.face_mesh.status,
                                             face_landmark_count=len(frame.face_mesh.landmarks),
                                             queue_age_ms=round((before - self._epoch - packet.timestamp_seconds) * 1000, 2),
                                             latency_ms=round((time.monotonic() - before) * 1000, 2),
                                             **{key: round(value, 2) for key, value in timings.items()})
                            next_analysis = max(elapsed + 1 / self.settings.analysis_hz, time.monotonic() - self._epoch)
                            if self.preview:
                                import cv2
                                label = f"{elapsed:.1f}/{self.settings.duration_seconds:.0f}s | analyzed={self.metrics.frame_count} dropped={self.video.dropped} | Q: stop"
                                cv2.imshow("CARE HERO real-time", draw_preview(image, frame, label))
                                if cv2.waitKey(1) & 0xFF == ord("q"):
                                    stopped_by_user = True
                                    break
                        else:
                            self.stop.wait(min(0.01, next_analysis - elapsed))
                        if elapsed >= next_snapshot:
                            snapshot = self.metrics.snapshot()
                            with self._lock:
                                self._latest = snapshot
                                self._snapshots.append(snapshot)
                            windows.write(json.dumps(snapshot, ensure_ascii=False, allow_nan=False) + "\n")
                            windows.flush()
                            snapshot_count += 1
                            self.events.emit("rules", "running", processed=snapshot_count,
                                             window_end_seconds=snapshot["window_end_seconds"],
                                             output_status=snapshot["rule_based"]["overall_status"])
                            self.events.emit("session", "running", elapsed_seconds=round(elapsed, 2),
                                             session_fraction=min(1.0, elapsed / self.settings.duration_seconds),
                                             analyzed=self.metrics.frame_count, video_dropped=self.video.dropped,
                                             audio_dropped=self.audio.dropped)
                            states = self.events.snapshot()
                            print(f"\r{elapsed:6.1f}s | frames={self.metrics.frame_count:4d} drop={self.video.dropped:4d} | "
                                  f"STT={states.get('stt', {}).get('status', 'waiting')} "
                                  f"LLM={states.get('llm_turn', {}).get('status', 'waiting')}", end="", flush=True)
                            next_snapshot = elapsed + self.settings.snapshot_interval_seconds
        except KeyboardInterrupt:
            stopped_by_user = True
        except Exception as exc:
            failure = type(exc).__name__
            self.events.emit("session", "error", code=failure)
        finally:
            if self._epoch:
                capture_elapsed = min(self.settings.duration_seconds, max(0.0, time.monotonic() - self._epoch))
            self.stop.set()
            if self.microphone is not None:
                try:
                    self.microphone.close()
                except Exception:
                    self.events.emit("audio", "error", code="microphone_close_failed")
            self.audio_done.set()
            if self.source is not None:
                try:
                    self.source.close()
                except Exception:
                    failure = failure or "camera_close_failed"
                    self.events.emit("capture", "error", code="camera_close_failed")
            if self.preview:
                import cv2
                cv2.destroyAllWindows()
            deadline = time.monotonic() + self.settings.drain_timeout_seconds
            self.events.emit("session", "draining")
            for thread in self._threads:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
            unfinished = [thread.name for thread in self._threads if thread.is_alive()]
            with self._lock:
                self.frozen.set()
                turns = deepcopy(self._turns)

        self.events.emit("vision", "done", processed=self.metrics.frame_count)
        self.events.emit("rules", "done", processed=snapshot_count)
        if self._audio_active:
            self.events.emit("audio", "done", dropped_blocks=self.audio.dropped)
        rules = self.metrics.final_report()
        final_feedback, final_status = None, "skipped"
        if self.language_model is not None and not unfinished:
            self.events.emit("llm_final", "running")
            try:
                recent = [turn["interpretation"] for turn in turns if turn.get("llm_status") == "success"]
                final_feedback = self.language_model.final_feedback(rules, recent).model_dump()
                final_status = "success"
            except Exception as exc:
                final_status = "error"
                self.events.emit("llm_final", "error", code=type(exc).__name__)
            self.events.emit("llm_final", final_status)
        elif unfinished:
            final_status = "not_run_incomplete_workers"

        self.events.emit("export", "running")
        result = {
            "schema_version": "2.0", "mode": "synthetic_demo" if self.synthetic else "live",
            "status": "partial" if failure or unfinished or self._capture_error or (self.source is not None and not self.metrics.frame_count) else "stopped" if stopped_by_user else "completed",
            "error_code": failure,
            "capture_duration_seconds": round(capture_elapsed, 4),
            "settings": self.settings.model_dump(),
            "scenario": asdict(self.rules),
            "summary": self.metrics.summary(),
            "rule_based": rules.to_dict(),
            "turns": sorted(turns, key=lambda item: item["utterance_id"]),
            "llm_status": final_status, "llm_feedback": final_feedback,
            "queues": {
                "video": {"dropped": self.video.dropped, "pending": self.video.size},
                "audio": {"dropped": self.audio.dropped, "pending": self.audio.size},
                "stt": {"dropped": self.speech.dropped, "pending": self.speech.size},
                "llm": {"dropped": self.language.dropped, "pending": self.language.size},
            },
            "unfinished_workers": unfinished,
        }
        if self.language_model is not None:
            result["llm_manifest"] = {"model": self.language_model.model, "prompts_sha256": self.language_model.prompts.manifest()}
        try:
            (self.output / "report.md").write_text(_markdown_report(result), encoding="utf-8")
            result["process_status"] = self.events.snapshot()
            _write_json(self.output / "session.json", result)
            self.events.emit("export", "completed")
            self.events.emit("session", result["status"])
            result["process_status"] = self.events.snapshot()
            _write_json(self.output / "session.json", result)
        except Exception as exc:
            self.events.emit("export", "error", code=type(exc).__name__)
            raise
        finally:
            self.events.close()
        print()
        return result


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _markdown_report(result: dict) -> str:
    lines = ["# CARE:HERO real-time session", "", f"- Mode: `{result['mode']}`", f"- Status: `{result['status']}`", "",
             "## Posture measurements", "", "| Metric | Value | Unit | Status | Provisional score |", "|---|---:|---|---|---:|"]
    for metric in result["rule_based"]["metrics"]:
        lines.append(f"| {metric['display_name']} | {metric['raw_value']} | {metric['unit']} | {metric['status']} | {metric['score']} |")
    lines += ["", f"Provisional total: {result['rule_based']['total_score']}", "", "## Completed utterances", ""]
    for turn in result["turns"]:
        text = turn.get("transcription", {}).get("full_text", "[STT unavailable]")
        # Quote trainee text to keep it distinct from generated report instructions.
        lines += [f"### Turn {turn['utterance_id']} ({turn['start_seconds']:.2f}–{turn['end_seconds']:.2f}s)", "",
                  *[f"> {part}" for part in text.splitlines()], "", f"LLM status: `{turn['llm_status']}`", ""]
        if turn.get("interpretation"):
            lines += [turn["interpretation"]["observation"], ""]
    if result["llm_feedback"]:
        feedback = result["llm_feedback"]
        lines += ["## Final feedback", "", feedback["overall_summary"], ""]
        for item in feedback["strengths"] + feedback["improvements"]:
            lines += [f"- {item['title']}: {item['message']} / {item['action']}"]
        lines += ["", "## Practice plan", ""]
        lines += [f"{step['order']}. {step['instruction']} — {step['success_check']}" for step in feedback["practice_plan"]]
    lines += ["", "## Measurement limitations", ""]
    lines += [f"- {message}" for message in result["rule_based"]["caveats"]]
    return "\n".join(lines) + "\n"
