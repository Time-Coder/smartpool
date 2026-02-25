import subprocess
import sys
import os


self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
subprocess.check_call([sys.executable, "-m", "build", self_folder])