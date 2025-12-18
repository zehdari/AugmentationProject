import subprocess
from pathlib import Path
import csv
import sys
import shutil
from configs import *

# Executable for running the commands
PY = sys.executable

### CONFIG

# Where this runner lives (and where the scripts live)
HERE = Path(__file__).resolve().parent
AUG_SCRIPT = HERE / "make_aug_dataset.py"
TRAIN_SCRIPT = HERE / "train_yolo.py"

DATASET_PARENT = BASE_DATASET.parent

### Helpers

def run_cmd(cmd: str):
    print("\n>>>", cmd)
    subprocess.run(cmd, check=True)
    
# Read a CSV and return the last row as dict (works for Ultralytics results.csv)
def csv_last_row(path: Path) -> dict:
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

    # Optional initial control run
    if RUN_CONTROL_FIRST:
        print("RUNNING: CONTROL (base dataset, no augmentation)")

        dataset_name = "Dataset_control"
        data_yaml = "data.yaml"

        run_name = "kitti_control"
        run_dir = Path(YOLO_PROJECT) / run_name
        results_csv = run_dir / "results.csv"

        # Train in yolo env
        run_cmd([
            PY, str(TRAIN_SCRIPT),
            "--data", str(data_yaml),
            "--project", str(YOLO_PROJECT),
            "--name", run_name,
        ])

        # Collect metrics (last row (epoch) of results.csv) and append
        print("Collecting metrics from:", results_csv)
        final = csv_last_row(results_csv)

        record = {
            "aug": "control",
            "p": 0.0,
            "dataset": dataset_name,
            "project": str(YOLO_PROJECT),
            "run_name": run_name,
            **final,
        }
        append_metrics(record)
        print(f"Appended metrics to {METRICS_CSV}")

    for p in PS:
        p_tag = str(p).replace(".", "")

        for aug in AUGS:
            print(f"RUNNING: aug={aug}, p={p}")

            dataset_name = f"Dataset_{aug}_p{p_tag}"
            output_dataset = DATASET_PARENT / dataset_name
            data_yaml = output_dataset.parent / f"data_{output_dataset.name}.yaml"

            run_name = f"kitti_{aug}_p{p}"
            run_dir = Path(YOLO_PROJECT) / run_name
            results_csv = run_dir / "results.csv"

            # If augmented dataset exists, skip augmentation only
            # Otherwise augment
            if output_dataset.exists():
                print(f"Skipping augmentation (dataset exists): {output_dataset}")
            else:
                run_cmd([
                    PY, str(AUG_SCRIPT),
                    "--input_root", str(BASE_DATASET),
                    "--output_root", str(output_dataset),
                    "--aug", aug,
                    "--p", str(p),
                    "--imgsz", str(IMGSZ),
                ])

            # Train
            run_cmd([
                PY, str(TRAIN_SCRIPT),
                "--data", str(data_yaml),
                "--project", str(YOLO_PROJECT),
                "--name", run_name,
            ])

            # Collect metrics (last row of results.csv) and append
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

            # Optional cleanup
            if DELETE_AUG_DATASET_AFTER_RUN:
                shutil.rmtree(output_dataset)
                print(f"Deleted augmented dataset folder: {output_dataset}")

            if DELETE_AUG_YAML_AFTER_RUN and data_yaml.exists():
                data_yaml.unlink()
                print(f"Deleted augmented yaml: {data_yaml}")

    print("\nALL EXPERIMENTS COMPLETE")

if __name__ == "__main__":
    main()