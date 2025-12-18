# AugmentationProject

## Setup

### Using venv

```bash
python3 -m venv aug_venv
source ./aug_venv/bin/activate
pip install albumentationsx ultralytics pillow opencv-python pyyaml
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

- The coco-minitrain dataset can be downloaded using: [here](https://huggingface.co/datasets/bryanbocao/coco_minitrain/resolve/main/coco_minitrain_25k.zip)

```bash
gdown https://drive.google.com/uc?id=11m1htBVId8jpAd4E16GaqzXKcPrAFjTw
```

**The kitti dataset was converted to yolo format using the provided conversion scripts in `convert/`.**

## Configure the sweep

Provide the paths to the dataset, data.yaml, and where you want the runs to be output in `config.py`
Other configurations can be done for the sweep and the model here.

## Running the sweep

Once the environment and datasets are setup, the `run_experiments.py` script can be run, and will perform the sweep.
The pipeline will augment the train data, train a model, and repeat for each possible augmentation/probability.
The results will be collected in the output csv.
