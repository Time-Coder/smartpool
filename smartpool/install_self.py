import os
import subprocess
import sys

self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", f"{self_folder}", "--upgrade", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])
