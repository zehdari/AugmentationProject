from pathlib import Path

# To unmodified dataset
BASE_DATASET = Path(r"Yolo dataset here")

# The unmodified data.yaml
SOURCE_DATA_YAML = Path(r"path to data.yaml")

# YOLO project directory
YOLO_PROJECT = Path(r"your run dir here") 


# Master metrics CSV (will append resutls)
METRICS_CSV = YOLO_PROJECT / "aug_results.csv"

# Which augmentations to sweep, defined in make_aug_dataset.py
AUGS = ["mirror","hsv", "gamma", "clahe", "gauss_noise", "motion_blur_small", "affine_small", "rotate", "zoom", "crop", "brightness", "contrast", "sharpness", "blur", "dropout"]

# Probabilities to sweep (1 <= p <= 0)
PS = [1.0, 0.5, 0.25]

# Image size used in some augmentations (to match yolo11n input)
IMGSZ = 640

RUN_CONTROL_FIRST = True

# Cleanup (so you don't run out of disk space)
DELETE_AUG_DATASET_AFTER_RUN = True
DELETE_AUG_YAML_AFTER_RUN = True

### Fixed training parameters
MODEL_WEIGHTS = "yolo11n.pt"
EPOCHS = 30
BATCH = 32
IMGSZ = 640
WORKERS = 8
AUGMENT = False