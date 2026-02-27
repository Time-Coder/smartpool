import subprocess
import sys
import os
import multiprocessing as mp

self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")

subprocess.check_call([sys.executable, "-m", "smartpool_examples.cross_validation", "--pool", "ray", "--max_workers", str(min(25, mp.cpu_count()))], cwd=self_folder)
