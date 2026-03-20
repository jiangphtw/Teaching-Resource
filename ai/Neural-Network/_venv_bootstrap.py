from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


_ENV_FLAG = "_NN_AUTO_VENV_REEXEC"


def rerun_with_nearest_venv() -> None:
    if os.environ.get(_ENV_FLAG) == "1":
        return

    script_path = Path(sys.argv[0]).resolve()
    current_python = Path(sys.executable).resolve()

    search_roots = [script_path.parent, *script_path.parents]
    for root in search_roots:
        venv_python = root / ".venv" / "Scripts" / "python.exe"
        if not venv_python.exists():
            continue

        venv_python = venv_python.resolve()
        if current_python == venv_python:
            return

        env = os.environ.copy()
        env[_ENV_FLAG] = "1"
        print(f"Switching to project venv: {venv_python}", flush=True)
        completed = subprocess.run(
            [str(venv_python), str(script_path), *sys.argv[1:]],
            env=env,
        )
        raise SystemExit(completed.returncode)
