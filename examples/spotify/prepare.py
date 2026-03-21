import subprocess
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = "maharshipandya/-spotify-tracks-dataset"
ZIP_FILE = os.path.join(DATA_DIR, "-spotify-tracks-dataset.zip")
CSV_FILE = os.path.join(DATA_DIR, "dataset.csv")

if not os.path.exists(CSV_FILE):
    subprocess.run(["uvx", "kaggle", "datasets", "download", "-d", DATASET, "-p", DATA_DIR], check=True)
    subprocess.run(["unzip", ZIP_FILE, "-d", DATA_DIR], check=True)
