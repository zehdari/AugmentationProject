# AugmentationProject

## Experiments runs / Results
The runs including all models trained for the project can be found [here](https://drive.google.com/drive/folders/1BxBJGI2VWoK_FDQPvEgaUGvLi2vEwqQk)
The generated csvs of the runs (last epoch of each run), and detailed results can be found in the project report [pdf](results/Report.pdf) in `results/`.

**If you want to run the sweep yourself follow the instructions below:**

## Setup

### Using venv

```bash
python3 -m venv aug_venv
source ./aug_venv/bin/activate
pip install albumentationsx ultralytics pillow opencv-python pyyaml
```

### Dataset

This was tested using the [KITTI 2d left train](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) and [coco-minitrain](https://github.com/giddyyupp/coco-minitrain) datasets. *The kitti dataset was converted to yolo format using the conversion scripts in `convert/` (not needed if using the yolo ready datasets)*

The yolo ready datasets can be downloaded from my google drive below.
To download from google drive, you can use the gdown package
```bash
pip install gdown
```

**The presplit (80/20 train/val) kitti dataset (~5GB):**
```bash
gdown https://drive.google.com/uc?id=11m1htBVId8jpAd4E16GaqzXKcPrAFjTw
```

**The coco-minitrain dataset (~10GB):**
```bash
gdown https://drive.google.com/uc?id=1DAkqZ8PPPWjD-aOIP63T_dg5p44Ec3G6
```

## Sweep

### Configure the sweep

Provide the paths to the dataset, data.yaml, and where you want the runs to be output in `config.py`
Other configurations can be done for the sweep and the model.

### Running the sweep

Once the environment and datasets are setup and in configs.py, the script can be run to will perform the sweep:

```bash
python run_experiments.py
```

The pipeline will augment the train data, train a model, and repeat for each possible augmentation/probability, evaluating against val.
The results will be collected in the output csv.

## Augmentations
Augmentations can be added or modified in `make_aug_dataset.py` in `build_transforms()`. These are from the [Albumentations](https://albumentations.ai/) library.