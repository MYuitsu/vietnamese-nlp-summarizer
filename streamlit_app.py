"""Cloud-friendly Streamlit entrypoint."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

runpy.run_path(str(SRC_DIR / "nlp_summary/ui/app.py"), run_name="__main__")
