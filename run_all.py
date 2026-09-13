#!/usr/bin/env python3

# Run the two scripts described in the Supplementary Code section.

import subprocess
import sys
from pathlib import Path


folder = Path(__file__).resolve().parent
scripts = [
    folder / "verify_hj7.py",
    folder / "verify_from_q.py",
]

for script in scripts:
    print()
    print(script.name, flush=True)
    print(flush=True)
    result = subprocess.run([sys.executable, str(script)], cwd=folder)
    if result.returncode != 0:
        sys.exit(result.returncode)

sys.exit(0)
