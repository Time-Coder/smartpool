import subprocess
import sys
import os


self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
subprocess.call([sys.executable, "-m", "twine", "upload", f"{self_folder}/dist/smartprocesspool_examples-*.tar.gz", f"{self_folder}/dist/smartprocesspool_examples-*.whl", "--verbose"])