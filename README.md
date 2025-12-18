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

## Running the scripts

### Test inference

```bash
python inference_test.py
```

### Test augmentations

```bash
python augmentation_test.py
```

### Test Training

#### Download a test training dataset

To download a sample dataset for training we can use roboflow. Create a roboflow account and go to [roboflow settings](https://www.google.com/url?q=https%3A%2F%2Fapp.roboflow.com%2Fsettings%2Fapi)

Create a file called `.env` and inside of it put `ROBOFLOW_API_KEY=your_api_key_here`

With the API key set up you can now run:

```bash
python download_dataset.py
```

#### Run the training

Now that the dataset is downloaded, we can start training with

```bash
python train_test.py
```
