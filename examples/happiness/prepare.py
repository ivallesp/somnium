import subprocess
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = "mathurinache/world-happiness-report"
ZIP_FILE = os.path.join(DATA_DIR, "world-happiness-report.zip")
CSV_FILE = os.path.join(DATA_DIR, "2022.csv")

if not os.path.exists(CSV_FILE):
    subprocess.run(["uvx", "kaggle", "datasets", "download", "-d", DATASET, "-p", DATA_DIR], check=True)
    subprocess.run(["unzip", ZIP_FILE, "-d", DATA_DIR], check=True)
