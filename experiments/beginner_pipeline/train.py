"""합성 예제 CSV로 초보자용 행동 분류 모델을 학습한다."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from model import FEATURE_NAMES, BeginnerBehaviorModel


ROOT = Path(__file__).parent


def read_labeled_csv(path: str | Path) -> tuple[list[dict[str, float | None]], list[int]]:
    rows: list[dict[str, float | None]] = []
    labels: list[int] = []
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = set(FEATURE_NAMES) | {"needs_improvement"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV에는 다음 열이 필요합니다: {sorted(required)}")
        for csv_row in reader:
            rows.append(
                {
                    name: None if csv_row[name].strip() == "" else float(csv_row[name])
                    for name in FEATURE_NAMES
                }
            )
            labels.append(int(csv_row["needs_improvement"]))
    return rows, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="CARE:HERO 예제 모델 학습")
    parser.add_argument(
        "--data",
        default=ROOT / "data" / "demo_train.csv",
        help="학습 CSV 경로",
    )
    parser.add_argument(
        "--output",
        default=ROOT / "weights" / "behavior_model.json",
        help="저장할 JSON weight 경로",
    )
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    args = parser.parse_args()

    feature_rows, labels = read_labeled_csv(args.data)
    model = BeginnerBehaviorModel().fit(
        feature_rows,
        labels,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    output_path = model.save(args.output)
    predictions = [model.predict(row) for row in feature_rows]
    correct = sum(prediction == label for prediction, label in zip(predictions, labels))
    print(f"학습 샘플 수: {len(labels)}")
    print(f"학습 데이터 정확도(참고용): {correct / len(labels):.3f}")
    print(f"weight 저장: {output_path.resolve()}")
    print("주의: 포함된 데이터는 실행 확인용 합성 예제이며 공식 교육 기준이 아닙니다.")


if __name__ == "__main__":
    main()

