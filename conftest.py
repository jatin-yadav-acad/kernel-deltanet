import sys
from pathlib import Path

# Disable bytecode (.pyc) compilation to keep repository clean
sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure kernel_deltanet namespace is seamlessly accessible
