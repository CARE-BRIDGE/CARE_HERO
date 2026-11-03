"""외부 ML 프레임워크 없이 학습 원리를 볼 수 있는 작은 로지스틱 회귀 모델.

이 모델은 Pose나 FaceMesh 자체를 학습하지 않는다. MediaPipe 검출 결과에서 계산한
수치 특징을 입력받아 `needs_improvement` 라벨을 학습한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


FEATURE_NAMES = (
    "pose_detection_rate",
    "face_detection_rate",
    "front_facing_ratio",
    "mean_abs_torso_tilt_deg",
    "torso_tilt_std_deg",
    "upper_body_ratio",
)

# 상반신 촬영에서는 몸통 기울기 값이 계산되지 않을 수 있다. 결측 여부 자체도 모델에
# 알려 주어, 없는 값을 억지로 0점으로 취급하지 않도록 한다.
MISSING_INDICATOR_FEATURES = (
    "mean_abs_torso_tilt_deg",
    "torso_tilt_std_deg",
)


@dataclass
class BeginnerBehaviorModel:
    """특징 표준화 정보와 로지스틱 회귀 weight를 함께 보관한다."""

    imputation_values: np.ndarray | None = None
    standard_means: np.ndarray | None = None
    standard_stds: np.ndarray | None = None
    weights: np.ndarray | None = None
    bias: float = 0.0

    @property
    def expanded_feature_names(self) -> tuple[str, ...]:
        missing_names = tuple(f"{name}_missing" for name in MISSING_INDICATOR_FEATURES)
        return FEATURE_NAMES + missing_names

    def fit(
        self,
        rows: Iterable[Mapping[str, float | None]],
        labels: Iterable[int],
        *,
        learning_rate: float = 0.08,
        epochs: int = 3000,
        l2_strength: float = 0.01,
    ) -> "BeginnerBehaviorModel":
        """경사하강법으로 이진 분류 weight를 학습한다."""

        raw_matrix = self._rows_to_raw_matrix(rows)
        y = np.asarray(list(labels), dtype=np.float64)
        if raw_matrix.shape[0] != y.shape[0] or y.size == 0:
            raise ValueError("특징 행 수와 라벨 수가 같고 1개 이상이어야 합니다.")
        if not np.all(np.isin(y, [0.0, 1.0])):
            raise ValueError("needs_improvement 라벨은 0 또는 1이어야 합니다.")

        numeric = raw_matrix[:, : len(FEATURE_NAMES)]
        self.imputation_values = _column_nan_means(numeric)
        expanded = self._impute_and_expand(raw_matrix)
        self.standard_means = expanded.mean(axis=0)
        self.standard_stds = expanded.std(axis=0)
        self.standard_stds[self.standard_stds < 1e-8] = 1.0
        x = (expanded - self.standard_means) / self.standard_stds

        self.weights = np.zeros(x.shape[1], dtype=np.float64)
        self.bias = 0.0
        sample_count = x.shape[0]
        for _ in range(epochs):
            probability = _sigmoid(x @ self.weights + self.bias)
            error = probability - y
            weight_gradient = (x.T @ error) / sample_count + l2_strength * self.weights
            bias_gradient = float(error.mean())
            self.weights -= learning_rate * weight_gradient
            self.bias -= learning_rate * bias_gradient
        return self

    def predict_probability(self, row: Mapping[str, float | None]) -> float:
        """한 행의 '개선 필요 가능성'을 0~1 사이 값으로 반환한다."""

        self._check_fitted()
        raw = self._rows_to_raw_matrix([row])
        expanded = self._impute_and_expand(raw)
        standardized = (expanded - self.standard_means) / self.standard_stds
        value = _sigmoid(standardized @ self.weights + self.bias)[0]
        return float(value)

    def predict(self, row: Mapping[str, float | None], threshold: float = 0.5) -> int:
        return int(self.predict_probability(row) >= threshold)

    def save(self, path: str | Path) -> Path:
        """사람이 열어볼 수 있는 JSON 형식으로 weight를 저장한다."""

        self._check_fitted()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": "numpy_logistic_regression",
            "target": "needs_improvement",
            "feature_names": list(FEATURE_NAMES),
            "expanded_feature_names": list(self.expanded_feature_names),
            "imputation_values": self.imputation_values.tolist(),
            "standard_means": self.standard_means.tolist(),
            "standard_stds": self.standard_stds.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "warning": "교육용 합성 데이터로 생성된 예제 weight이며 공식 점수나 진단에 사용할 수 없음",
        }
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> "BeginnerBehaviorModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if tuple(payload["feature_names"]) != FEATURE_NAMES:
            raise ValueError("weight 파일의 특징 순서가 현재 코드와 다릅니다.")
        return cls(
            imputation_values=np.asarray(payload["imputation_values"], dtype=np.float64),
            standard_means=np.asarray(payload["standard_means"], dtype=np.float64),
            standard_stds=np.asarray(payload["standard_stds"], dtype=np.float64),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            bias=float(payload["bias"]),
        )

    def _rows_to_raw_matrix(
        self, rows: Iterable[Mapping[str, float | None]]
    ) -> np.ndarray:
        values: list[list[float]] = []
        for row in rows:
            values.append(
                [
                    np.nan if row.get(name) is None else float(row[name])
                    for name in FEATURE_NAMES
                ]
            )
        if not values:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
        return np.asarray(values, dtype=np.float64)

    def _impute_and_expand(self, raw_matrix: np.ndarray) -> np.ndarray:
        if self.imputation_values is None:
            raise RuntimeError("결측값 대체 기준이 없습니다.")
        numeric = raw_matrix[:, : len(FEATURE_NAMES)].copy()
        missing_columns = []
        for feature_name in MISSING_INDICATOR_FEATURES:
            column_index = FEATURE_NAMES.index(feature_name)
            missing_columns.append(np.isnan(numeric[:, column_index]).astype(np.float64))
        missing_mask = np.isnan(numeric)
        numeric[missing_mask] = np.take(self.imputation_values, np.where(missing_mask)[1])
        return np.column_stack([numeric, *missing_columns])

    def _check_fitted(self) -> None:
        required = (
            self.imputation_values,
            self.standard_means,
            self.standard_stds,
            self.weights,
        )
        if any(value is None for value in required):
            raise RuntimeError("먼저 모델을 학습하거나 weight 파일을 불러와야 합니다.")


def _column_nan_means(matrix: np.ndarray) -> np.ndarray:
    means = np.nanmean(matrix, axis=0)
    if np.isnan(means).any():
        raise ValueError("모든 행에서 비어 있는 특징이 있습니다.")
    return means


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))

