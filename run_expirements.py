import subprocess
from pathlib import Path
import csv

### CONFIG

AUG_ENV = "augment"
TRAIN_ENV = "yolov8_segmentation"

# Where this runner lives (and where the scripts live)
HERE = Path(__file__).resolve().parent
AUG_SCRIPT = HERE / "make_aug_dataset.py"
TRAIN_SCRIPT = HERE / "train_yolo.py"

# Dataset roots
BASE_DATASET = Path(r"S:\AugProject\kitti_rf_detr\train\rfdetr_dataset\Dataset")
DATASET_PARENT = Path(r"S:\AugProject\kitti_rf_detr\train\rfdetr_dataset")

# YOLO project directory
YOLO_PROJECT = Path(r"runs\kitti")  # relative is fine

# Master metrics CSV (will append resutls)
METRICS_CSV = HERE / "aug_results.csv"

AUGS = ["mirror","hsv", "gamma", "clahe", "gauss_noise", "motion_blur_small", "affine_small", "rotate", "zoom", "crop", "brightness", "contrast", "sharpness", "blur", "dropout"]
#AUGS = ["hsv", "gamma", "clahe", "gauss_noise", "motion_blur_small", "affine_small"]
PS = [1.0, 0.5, 0.25]
IMGSZ = 640

RUN_CONTROL_FIRST = False
DELETE_AUG_DATASET_AFTER_RUN = False
DELETE_AUG_YAML_AFTER_RUN = False

### Helpers

def run_cmd(cmd: str):
    print("\n>>>", cmd)
    subprocess.run(cmd, shell=True, check=True)

def conda_cmd(env: str, inner_cmd: str) -> str:
    """
    Run inner_cmd inside a conda env, in ONE cmd.exe session.
    /d lets cmd run even if AutoRun is configured.
    """
    return f'cmd.exe /d /c "conda deactivate && conda activate {env} && {inner_cmd}"'

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

    if RUN_CONTROL_FIRST:
        print("\n" + "=" * 80)
        print("RUNNING: CONTROL (base dataset, no augmentation)")
        print("=" * 80)

        dataset_name = "Dataset_control"
        data_yaml = DATASET_PARENT / "data.yaml"

        run_name = "kitti_control"
        run_dir = Path(YOLO_PROJECT) / run_name
        results_csv = run_dir / "results.csv"

        # 2) Train in yolo env
        train_cmd = (
            f'python "{TRAIN_SCRIPT}" '
            f'--data "{data_yaml}" '
            f'--project "{YOLO_PROJECT}" '
            f'--name "{run_name}"'
        )
        run_cmd(conda_cmd(TRAIN_ENV, train_cmd))

        # 3) Collect metrics (last row of results.csv) and append
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
        p_tag = str(p).replace(".", "")  # 1.0->10, 0.5->05, 0.25->025

        for aug in AUGS:
            print("\n" + "=" * 80)
            print(f"RUNNING: aug={aug}, p={p}")
            print("=" * 80)

            dataset_name = f"Dataset_{aug}_p{p_tag}"
            output_dataset = DATASET_PARENT / dataset_name
            data_yaml = output_dataset.parent / f"data_{output_dataset.name}.yaml"

            run_name = f"kitti_{aug}_p{p}"
            run_dir = Path(YOLO_PROJECT) / run_name
            results_csv = run_dir / "results.csv"

            # If run exists skip all
            if run_dir.exists():
                print(f"Skipping run entirely (run exists): {run_dir}")
                continue

            # If augmented dataset exists, skip augmentation only
            if output_dataset.exists():
                print(f"Skipping augmentation (dataset exists): {output_dataset}")
            else:
                aug_cmd = (
                    f'python "{AUG_SCRIPT}" '
                    f'--input_root "{BASE_DATASET}" '
                    f'--output_root "{output_dataset}" '
                    f'--aug {aug} '
                    f'--p {p} '
                    f'--imgsz {IMGSZ}'
                )
                run_cmd(conda_cmd(AUG_ENV, aug_cmd))

            # Train
            train_cmd = (
                f'python "{TRAIN_SCRIPT}" '
                f'--data "{data_yaml}" '
                f'--project "{YOLO_PROJECT}" '
                f'--name "{run_name}"'
            )
            run_cmd(conda_cmd(TRAIN_ENV, train_cmd))

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

            if DELETE_AUG_DATASET_AFTER_RUN:
                try:
                    import shutil
                    shutil.rmtree(output_dataset)
                    print(f"Deleted augmented dataset folder: {output_dataset}")
                except Exception as e:
                    print(f"Warning: failed to delete dataset folder {output_dataset}: {e}")

            if DELETE_AUG_YAML_AFTER_RUN:
                try:
                    if data_yaml.exists():
                        data_yaml.unlink()
                        print(f"Deleted augmented yaml: {data_yaml}")
                except Exception as e:
                    print(f"Warning: failed to delete yaml {data_yaml}: {e}")

    print("\nALL EXPERIMENTS COMPLETE")

if __name__ == "__main__":
    main()