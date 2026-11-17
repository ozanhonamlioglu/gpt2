import subprocess
import sys

# Each stage runs in its own process so pretraining's dropout=0.0 doesn't leak into fine-tuning
subprocess.run([sys.executable, "pretraining.py"], check=True)
subprocess.run([sys.executable, "supertuning.py"], check=True)
