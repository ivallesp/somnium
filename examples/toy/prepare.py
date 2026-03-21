import subprocess
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = "giovamata/airlinedelaycauses"
ZIP_FILE = os.path.join(DATA_DIR, "airlinedelaycauses.zip")
CSV_FILE = os.path.join(DATA_DIR, "DelayedFlights.csv")

if not os.path.exists(CSV_FILE):
    subprocess.run(["uvx", "kaggle", "datasets", "download", "-d", DATASET, "-p", DATA_DIR], check=True)
    subprocess.run(["unzip", ZIP_FILE, "-d", DATA_DIR], check=True)
