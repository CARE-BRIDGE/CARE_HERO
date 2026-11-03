from __future__ import annotations

from pathlib import Path

import pytest

from care_hero.config import load_config
from care_hero.errors import ConfigurationError


ROOT = Path(__file__).resolve().parents[1]


def test_face_mesh_is_enabled_for_one_trainee() -> None:
    config = load_config(ROOT / "configs" / "base.yaml")
    assert config.face_mesh.enabled
    assert config.face_mesh.max_num_faces == 1


def test_multiple_faces_are_rejected_explicitly(tmp_path) -> None:
    source = (ROOT / "configs" / "base.yaml").read_text(encoding="utf-8")
    path = tmp_path / "base.yaml"
    path.write_text(
        source.replace("max_num_faces: 1", "max_num_faces: 2"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="얼굴 한 명"):
        load_config(path)
