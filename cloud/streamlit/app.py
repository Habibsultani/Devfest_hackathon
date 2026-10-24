"""Community Cloud entrypoint for the existing integrated application."""

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
runpy.run_path(str(PROJECT_ROOT / "app.py"), run_name="__main__")
