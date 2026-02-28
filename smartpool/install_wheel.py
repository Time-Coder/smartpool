import subprocess
import sys
import os


self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")

dist_dir = self_folder + "/dist"
files = os.listdir(dist_dir)
newest_file = ""
newest_mtime = 0
for file in files:
    if file.endswith(".whl") and os.path.getmtime(dist_dir + "/" + file) > newest_mtime:
        newest_file = file

subprocess.check_call([sys.executable, "-m", "pip", "uninstall", f"{dist_dir}/{newest_file}", "-y"])
subprocess.check_call([sys.executable, "-m", "pip", "install", f"{dist_dir}/{newest_file}", "--upgrade"])