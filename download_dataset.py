from dotenv import load_dotenv
from roboflow import download_dataset

# Load ROBOFLOW_API_KEY from .env file
load_dotenv()

# Download the dataset
dataset = download_dataset("https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-2/13", "coco")

