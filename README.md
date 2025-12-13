# Introduction and current progress
- [ ] Train one epoch then do inference on test.
- [ ] apply rotation using Albumentations
- [ ] 





# AugmentationProject OSC

## Setup

The set up is based on creating a Python Built-in Virtual Environment in OSC.

### Using venv

```bash
python3 -m venv aug_venv
source ./aug_venv/bin/activate
pip install albumentationsx rfdetr
```

### Apply for GPU
```bash
srun \
  --account=PAS2119 \
  --time=00:30:00 \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=8 \
  --gpus-per-node=1 \
  --pty /bin/bash
```


### Run training
```bash
torchrun --nproc_per_node=1 traincoco.py
```

## Augmentation
### rf-detr auto-apply augmentation
RF-DETR automatically applies a random crop and a random horizontal flip.

In addition, model randomly resize the image during training, allowing user to run with confidence at different resolutions at inference. People often refer to this as a multi scale augmentation.
Link to the details: [Forum](https://discuss.roboflow.com/t/rf-detr-augmentations/10996) and [code](https://github.com/roboflow/rf-detr/blob/24ce179cb5d71d9049724c9f3bc25b506d9f42a4/rfdetr/datasets/coco.py#L160)

