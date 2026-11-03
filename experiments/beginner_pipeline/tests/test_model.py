from __future__ import annotations

from model import BeginnerBehaviorModel


GOOD = {
    "pose_detection_rate": 0.96,
    "face_detection_rate": 0.95,
    "front_facing_ratio": 0.92,
    "mean_abs_torso_tilt_deg": 3.0,
    "torso_tilt_std_deg": 1.5,
    "upper_body_ratio": 0.1,
}

IMPROVE = {
    "pose_detection_rate": 0.55,
    "face_detection_rate": 0.60,
    "front_facing_ratio": 0.40,
    "mean_abs_torso_tilt_deg": 15.0,
    "torso_tilt_std_deg": 8.0,
    "upper_body_ratio": 0.2,
}


def test_model_learns_separable_examples() -> None:
    model = BeginnerBehaviorModel().fit(
        [GOOD, GOOD, IMPROVE, IMPROVE],
        [0, 0, 1, 1],
        epochs=1200,
    )
    assert model.predict_probability(GOOD) < 0.5
    assert model.predict_probability(IMPROVE) > 0.5


def test_model_save_and_load_round_trip(tmp_path) -> None:
    model = BeginnerBehaviorModel().fit(
        [GOOD, GOOD, IMPROVE, IMPROVE],
        [0, 0, 1, 1],
        epochs=1200,
    )
    path = model.save(tmp_path / "weights.json")
    restored = BeginnerBehaviorModel.load(path)
    assert abs(
        restored.predict_probability(GOOD) - model.predict_probability(GOOD)
    ) < 1e-10


def test_upper_body_missing_values_are_supported() -> None:
    upper_good = {**GOOD, "mean_abs_torso_tilt_deg": None, "torso_tilt_std_deg": None, "upper_body_ratio": 0.95}
    upper_improve = {**IMPROVE, "mean_abs_torso_tilt_deg": None, "torso_tilt_std_deg": None, "upper_body_ratio": 0.90}
    model = BeginnerBehaviorModel().fit(
        [GOOD, IMPROVE, upper_good, upper_improve],
        [0, 1, 0, 1],
        epochs=1200,
    )
    assert 0.0 <= model.predict_probability(upper_good) <= 1.0

