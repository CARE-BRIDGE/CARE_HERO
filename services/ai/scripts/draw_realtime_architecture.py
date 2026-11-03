#!/usr/bin/env python3
"""Render a vector-first research architecture figure; no image generation service."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(17.6, 8.8))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.975, bottom=0.025)
    ax.set_xlim(0, 17.2)
    ax.set_ylim(0, 8.5)
    ax.axis("off")
    ink, blue, green = "#25303C", "#315C7D", "#39645D"

    def group(x, y, w, h, title):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.08",
                                   fc="#FAFBFC", ec="#AAB3BB", lw=0.9, linestyle=(0, (4, 3)), zorder=0))
        ax.text(x + 0.15, y + h - 0.28, title, ha="left", va="center", weight="bold", color=ink, fontsize=11)

    def box(x, y, w, h, title, detail="", color=blue):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.05",
                                   fc="white", ec=color, lw=1.15, zorder=2))
        ax.text(x + w / 2, y + h * (0.69 if detail else 0.5), title, ha="center", va="center",
                color=ink, fontsize=10, weight="bold", zorder=3)
        if detail:
            ax.text(x + w / 2, y + h * 0.29, detail, ha="center", va="center", color="#485561", fontsize=8.3, zorder=3)

    def arrow(start, end, *, label="", color=blue, bend=0, dashed=False):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, lw=1.15,
                                    color=color, linestyle="--" if dashed else "-",
                                    connectionstyle=f"arc3,rad={bend}", zorder=1))
        if label:
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.12, label,
                    ha="center", va="bottom", color=color, fontsize=7.7,
                    bbox={"facecolor": "#FAFBFC", "edgecolor": "none", "pad": 1}, zorder=4)

    ax.text(0.15, 8.22, "CARE:HERO: Observable Real-Time Multimodal Analysis", fontsize=17, weight="bold", color=ink)
    ax.text(0.15, 7.88, "Timestamped acquisition, bounded asynchronous workers, and evidence-grounded language feedback", color="#52606C", fontsize=10.3)
    group(0.2, 2.05, 3.0, 5.25, "(a) Acquisition")
    group(3.65, 2.05, 7.7, 5.25, "(b) Online processing")
    group(11.8, 2.05, 5.05, 5.25, "(c) Evidence and feedback")

    box(0.45, 5.5, 2.5, 1.05, "Webcam", "OpenCV / capture thread")
    box(0.45, 3.3, 2.5, 1.05, "Microphone", "16 kHz mono PCM / callback", green)
    box(3.95, 5.5, 1.55, 1.05, "Frame queue", "K = 2\nreplace oldest")
    box(5.95, 5.5, 2.4, 1.05, "Pose + FaceMesh", "33 / up to 478 landmarks\nvision.py / main thread")
    box(8.8, 5.5, 2.25, 1.05, "Windowed rules", "5 s window / 1 s updates\nmetrics.py + scoring.py")
    arrow((2.95, 6.02), (3.95, 6.02))
    arrow((5.5, 6.02), (5.95, 6.02))
    arrow((8.35, 6.02), (8.8, 6.02))

    box(3.95, 3.3, 1.55, 1.05, "PCM queue", "K = 64\ngap detection", green)
    box(5.95, 3.3, 2.4, 1.05, "Utterance endpointing", "RMS + silence / 15 s cap\naudio.py / segment thread", green)
    box(8.8, 3.3, 2.25, 1.05, "faster-whisper", "bounded speech queue K = 4\nSTT worker / final segments", green)
    arrow((2.95, 3.82), (3.95, 3.82), color=green)
    arrow((5.5, 3.82), (5.95, 3.82), color=green)
    arrow((8.35, 3.82), (8.8, 3.82), color=green)

    box(12.1, 5.5, 4.45, 1.05, "Time-stamped visual evidence", "window bounds / raw values / status / provisional scores")
    box(12.1, 3.3, 4.45, 1.05, "Structured language worker (optional)", "transcript + endpoint context / LLM queue K = 4\nprompt v1.0 / Pydantic + exact-quote validation", green)
    box(12.1, 2.25, 4.45, 0.65, "End-of-session report", "local rules + turn records + final feedback")
    arrow((11.05, 6.02), (12.1, 6.02))
    arrow((11.05, 3.82), (12.1, 3.82), color=green)
    arrow((14.33, 5.5), (14.33, 4.35), label="compact evidence")
    arrow((14.33, 3.3), (14.33, 2.9), color=green)
    # Local rule evidence also reaches the report when language inference is disabled.
    ax.plot([16.55, 16.73, 16.73], [6.02, 6.02, 2.58], color=blue, lw=1.0)
    arrow((16.73, 2.58), (16.55, 2.58))

    group(0.2, 0.48, 16.65, 1.02, "(d) Control and observability")
    ax.text(0.55, 0.78, "Shared monotonic clock    |    Event journal: stage, state, counts, latency, queue drops    |    Timed capture + bounded drain",
            fontsize=10, color=ink, ha="left", va="center")
    arrow((1.7, 2.05), (1.7, 1.5), color="#7C8690", dashed=True)
    arrow((7.5, 2.05), (7.5, 1.5), color="#7C8690", dashed=True)
    arrow((14.3, 2.05), (14.3, 1.5), color="#7C8690", dashed=True)
    ax.text(0.3, 0.08, "Figure 1. Solid arrows: data flow. Dashed boundaries: execution groups. Language inference consumes completed utterances; no token-level ASR or TTS is assumed.",
            fontsize=8.5, color="#52606C")
    output = ROOT / "docs/figures"
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        fig.savefig(output / f"realtime_architecture.{extension}", dpi=240, facecolor="white")
    plt.close(fig)
    print(f"Architecture figure exported to {output}")


if __name__ == "__main__":
    main()
