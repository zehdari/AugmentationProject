from ultralytics import YOLO
import multiprocessing as mp
import argparse
from configs import *


def main():
    parser = argparse.ArgumentParser(description="YOLO training entrypoint")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to data.yaml for this run",
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Ultralytics project directory (e.g. runs/kitti)",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Run name (subdirectory under project)",
    )

    args = parser.parse_args()

    # Initialize model
    model = YOLO(MODEL_WEIGHTS)

    # Train
    model.train(
        data=args.data,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        project=args.project,
        name=args.name,
        augment=AUGMENT,
        workers=WORKERS,
    )

if __name__ == "__main__":
    mp.freeze_support()  # Important if you want to run on a windows machine, otherwise shouldn't matter
    main()
