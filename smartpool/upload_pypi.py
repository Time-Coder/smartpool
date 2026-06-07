import os
import subprocess
import sys

self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
subprocess.call([sys.executable, "-m", "twine", "upload", f"{self_folder}/dist/pysmartpool-*.tar.gz", f"{self_folder}/dist/pysmartpool-*.whl", "--verbose"])
