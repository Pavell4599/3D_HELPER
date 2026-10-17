import os
import subprocess

SLICER = r"C:\Program Files\Bambu Studio\bambu-studio.exe"
MODEL_PATH = os.path.abspath(r"Backend\static\settings\TEST.3mf")

subprocess.Popen([SLICER, MODEL_PATH])
