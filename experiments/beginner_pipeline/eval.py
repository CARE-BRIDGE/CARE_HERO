"""학습에 쓰지 않은 CSV로 분류 모델을 평가한다."""

from __future__ import annotations

import argparse
from pathlib import Path

from model import BeginnerBehaviorModel
from train import read_labeled_csv


ROOT = Path(__file__).parent


def calculate_metrics(labels: list[int], predictions: list[int]) -> dict[str, float | int]:
    true_positive = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    true_negative = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    false_positive = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    false_negative = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    total = len(labels)
    accuracy = (true_positive + true_negative) / total if total else 0.0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CARE:HERO 예제 모델 평가")
    parser.add_argument(
        "--data",
        default=ROOT / "data" / "demo_eval.csv",
        help="평가 CSV 경로",
    )
    parser.add_argument(
        "--weights",
        default=ROOT / "weights" / "behavior_model.json",
        help="평가할 JSON weight 경로",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    rows, labels = read_labeled_csv(args.data)
    model = BeginnerBehaviorModel.load(args.weights)
    probabilities = [model.predict_probability(row) for row in rows]
    predictions = [int(value >= args.threshold) for value in probabilities]
    metrics = calculate_metrics(labels, predictions)

    print(f"평가 샘플 수: {len(labels)}")
    print(f"Accuracy : {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall   : {metrics['recall']:.3f}")
    print(f"F1       : {metrics['f1']:.3f}")
    print(
        "Confusion matrix: "
        f"TP={metrics['true_positive']}, TN={metrics['true_negative']}, "
        f"FP={metrics['false_positive']}, FN={metrics['false_negative']}"
    )
    print("주의: 합성 예제 데이터 평가이며 실제 성능 근거로 사용할 수 없습니다.")


if __name__ == "__main__":
    main()

