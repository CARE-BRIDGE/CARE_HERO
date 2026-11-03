from pathlib import Path

import pytest

from care_hero.config import load_config


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[1]
    return load_config(root / "configs" / "base.yaml")

