import subprocess
import sys
import os


self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")

subprocess.check_call([sys.executable, "-m", "smartpool_examples.count_prime"], cwd=self_folder)
