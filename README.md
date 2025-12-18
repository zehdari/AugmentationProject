# AugmentationProject

## Setup

Tested using python 3.13.9

### Using penv

```bash
pyenv install 3.13.9
pyenv virtualenv 3.13.9 venv
pyenv activate venv
pip install albumentationsx ultralytics pillow opencv-python
```

### Experiments runs
The runs including all models trained for the project can be downloaded as a zip file [here]()
### Dataset

This was tested using the [KITTI 2d left train](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) and [coco-minitrain](https://github.com/giddyyupp/coco-minitrain) datasets
 
The yolo ready datasets can be downloaded from google drive below:

To download from google drive, you can use the gdown package
```bash
pip install gdown
```

- The presplit (80/20 train/val) kitti dataset converted to yolo format zip can be downloaded using:

```bash
gdown https://drive.google.com/uc?id=11m1htBVId8jpAd4E16GaqzXKcPrAFjTw
```

- The coco-minitrain dataset can be downloaded using:
```bash
gdown https://drive.google.com/uc?id=11m1htBVId8jpAd4E16GaqzXKcPrAFjTw
```

**The kitti dataset was converted to yolo format using the provided conversion scripts in `convert/`.**

## Configure the sweep

The augmentation sweep can be configured in `run_experiments.py`. You can choose the probabilites of application, and which augmentations to perform, as well as run a control and handle cleanup.

The fixed training parameters for the yolo model can be configured in `train_yolo.py`.

The fixed augmentation strengths can be modified in `make_aug_dataset.py`. Additional augmentations can also be added here.

## Running the sweep

Once the environment and datasets are setup, the `run_experiments.py` script can be run, and will perform the sweep.
The pipeline will augment the train data, train a model, and repeat for each possible augmentation/probability.
The results will be collected in the output csv.
