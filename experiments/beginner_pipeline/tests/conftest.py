from __future__ import annotations

import sys
from pathlib import Path


# beginner_pipeline을 설치하지 않아도 테스트에서 바로 import할 수 있게 한다.
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

