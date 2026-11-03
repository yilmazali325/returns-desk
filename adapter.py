#!/usr/bin/env python3
"""Contract entry point: python adapter.py --input cases.json --output results.json [options]

Standard library only at import time, so the literal grader command works with any python3 from
any directory. If the project's dependencies are not importable here, it re-executes itself with
<repo>/.venv/bin/python (.venv/Scripts/python.exe on Windows; same arguments, same working directory, so
relative --input/--output paths still resolve). The real work lives in desk/adapter_main.py.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(ROOT, ".venv")
VENV_PYTHON = (os.path.join(VENV_DIR, "Scripts", "python.exe") if os.name == "nt"
               else os.path.join(VENV_DIR, "bin", "python"))
REEXEC_MARKER = "RETURNS_DESK_ADAPTER_REEXEC"
SETUP_HINT = "Set up the environment first: cd {root} && uv sync --frozen (see scripts/setup.sh)"


def dependencies_available() -> bool:
    try:
        import httpx  # noqa: F401
        import psutil  # noqa: F401
    except ImportError:
        return False
    return True


def running_in_venv() -> bool:
    return os.path.realpath(sys.prefix) == os.path.realpath(VENV_DIR)


def main() -> int:
    if not dependencies_available():
        if os.path.exists(VENV_PYTHON) and not running_in_venv() and not os.environ.get(REEXEC_MARKER):
            os.environ[REEXEC_MARKER] = "1"  # never loop if the venv itself is broken
            if os.name == "nt":  # Windows has no exec: run the venv's python and pass its exit code on
                import subprocess

                return subprocess.call([VENV_PYTHON, os.path.abspath(__file__), *sys.argv[1:]])
            os.execv(VENV_PYTHON, [VENV_PYTHON, os.path.abspath(__file__), *sys.argv[1:]])
        sys.stderr.write("adapter: project dependencies are missing. " + SETUP_HINT.format(root=ROOT) + "\n")
        return 2
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from desk.adapter_main import main as run_adapter

    return run_adapter(sys.argv[1:])


if __name__ == "__main__":  # pragma: no cover - run as a real subprocess in tests/unit/test_adapter.py
    sys.exit(main())
