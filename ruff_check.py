import os
import subprocess
import sys

self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
subprocess.check_call([sys.executable, "-m", "ruff", "check", "--fix", "."], cwd=self_folder)
