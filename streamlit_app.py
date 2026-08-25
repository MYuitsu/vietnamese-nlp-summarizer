"""Cloud-friendly Streamlit entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

APP_PATH = SRC_DIR / "nlp_summary/ui/app.py"

# ``runpy.run_path`` temporarily mutates ``sys.modules`` and ``sys.argv``.
# That can race with Streamlit's threaded script runner on Python 3.14 and
# surface intermittent KeyError exceptions during package imports. Compile the
# trusted application script into an isolated namespace instead.
app_globals = {
    "__name__": "__main__",
    "__file__": str(APP_PATH),
    "__package__": None,
}
exec(compile(APP_PATH.read_bytes(), str(APP_PATH), "exec"), app_globals)
