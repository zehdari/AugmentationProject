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

### Dataset

This was tested using the [KITTI 2d left train](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) and [coco-minitrain](https://github.com/giddyyupp/coco-minitrain) datasets

The kitti dataset can be converted to yolo format using the provided conversion scripts in `convert/`.
Coco-minitrain can be converted to json coco format using the script from the coco-minitrain repo, and converted to yolo with the script in `convert/`.

## Configure the sweep

The augmentation sweep can be configured in `run_experiments.py`. You can choose the probabilites of application, and which augmentations to perform, as well as run a control and handle cleanup.

The fixed training parameters for the yolo model can be configured in `train_yolo.py`.

The fixed augmentation strengths can be modified in `make_aug_dataset.py`. Additional augmentations can also be added here.

## Running the sweep

Once the environment and datasets are setup, the `run_experiments.py` script can be run, and will perform the sweep.
The pipeline will augment the train data, train a model, and repeat for each possible augmentation/probability.
The results will be collected in the output csv.
