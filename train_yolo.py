from ultralytics import YOLO
import multiprocessing as mp

def main():
    # Initialize model
    model = YOLO('yolo11n.pt')

    # Basic training configuration
    model.train(
        data=r'S:\AugProject\kitti_rf_detr\train\rfdetr_dataset\data.yaml',      # dataset config file
        epochs=30,                    # number of epochs
        batch=-1,                      # batch size (-1 for auto)
        imgsz=640,                     # image size
        project='runs/robot_training', # main project directory
        name='detection_model',         # subfolder name for this run
        augment=False,
        workers=8
    )

if __name__ == "__main__":
    mp.freeze_support()  # helps on Windows (fine for other os)
    main()
