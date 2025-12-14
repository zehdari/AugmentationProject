import subprocess
from pathlib import Path
import csv
import sys
PY = sys.executable
### CONFIG



# Where this runner lives (and where the scripts live)
HERE = Path(__file__).resolve().parent
AUG_SCRIPT = HERE / "make_aug_dataset.py"
TRAIN_SCRIPT = HERE / "train_yolo.py"

# Dataset roots
BASE_DATASET = Path(r"/users/PAS2119/darklord/CVfinalproject/AugmentationProject/coco_minitrain_25k")
DATASET_PARENT = Path(r"/users/PAS2119/darklord/CVfinalproject/AugmentationProject/modifieddataset")

# YOLO project directory
YOLO_PROJECT = Path(r"/users/PAS2119/darklord/CVfinalproject/AugmentationProject/runs")  # relative is fine

# Master metrics CSV (will append resutls)
METRICS_CSV = HERE / "aug_results.csv"

AUGS = ["rotate", "zoom", "crop", "brightness", "contrast", "sharpness", "blur", "dropout"]
PS = [1.0, 0.5, 0.25]
IMGSZ = 640

### Helpers

def run_cmd(cmd_list):
    print("\n>>>", " ".join(str(x) for x in cmd_list))
    subprocess.run(cmd_list, check=True)


def csv_last_row(path: Path) -> dict:
    """
    Read a CSV and return the last row as dict.
    (works for Ultralytics results.csv)
    """
    if not path.exists():
        raise FileNotFoundError(f"results.csv not found: {path}")

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        last = None
        for row in reader:
            last = row
    if last is None:
        raise RuntimeError(f"results.csv exists but had no rows: {path}")
    return last

def append_metrics(row: dict):
    write_header = not METRICS_CSV.exists()
    with METRICS_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def main():
    # sanity checks
    if not AUG_SCRIPT.exists():
        raise FileNotFoundError(f"Could not find {AUG_SCRIPT} (expected next to this runner)")
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Could not find {TRAIN_SCRIPT} (expected next to this runner)")

    for p in PS:
        p_tag = str(p).replace(".", "")

        for aug in AUGS:
            dataset_name = f"Dataset_{aug}_p{p_tag}"
            output_dataset = DATASET_PARENT / dataset_name
            data_yaml = output_dataset.parent / f"data_{output_dataset.name}.yaml"

            run_name = f"coco_{aug}_p{p_tag}"
            run_dir = YOLO_PROJECT / run_name
            results_csv = run_dir / "results.csv"

            # 1) Augment
            run_cmd([
                PY, str(AUG_SCRIPT),
                "--input_root", str(BASE_DATASET),
                "--output_root", str(output_dataset),
                "--aug", aug,
                "--p", str(p),
                "--imgsz", str(IMGSZ),
            ])

            # 2) Train
            run_cmd([
                PY, str(TRAIN_SCRIPT),
                "--data", str(data_yaml),
                "--project", str(YOLO_PROJECT),
                "--name", run_name,
            ])

            # 3) Collect metrics (last row of results.csv) and append
            print("Collecting metrics from:", results_csv)
            final = csv_last_row(results_csv)

            record = {
                "aug": aug,
                "p": p,
                "dataset": dataset_name,
                "project": str(YOLO_PROJECT),
                "run_name": run_name,
                **final,
            }
            append_metrics(record)
            print(f"Appended metrics to {METRICS_CSV}")

    print("\nALL EXPERIMENTS COMPLETE")

if __name__ == "__main__":
    main()
