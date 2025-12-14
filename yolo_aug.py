import cv2
import albumentations as A
from pathlib import Path

INPUT_ROOT = Path(r"/users/PAS2119/darklord/CVfinalproject/AugmentationProject/coco_minitrain_25k")
OUTPUT_ROOT = Path(r"/users/PAS2119/darklord/CVfinalproject/AugmentationProject/Dataset_brightness__p1")

SPLIT = "train"
COPIES_PER_IMAGE = 1 # No more for now

# for min_visibility=0.3
MIN_VISIBILITY = 0.3

IMG_IN = INPUT_ROOT / "images" / SPLIT
LBL_IN = INPUT_ROOT / "labels" / SPLIT
IMG_OUT = OUTPUT_ROOT / "images" / SPLIT
LBL_OUT = OUTPUT_ROOT / "labels" / SPLIT
IMG_OUT.mkdir(parents=True, exist_ok=True)
LBL_OUT.mkdir(parents=True, exist_ok=True)

### ROTATE
# transform = A.Compose(
#     [A.Rotate(limit=30, p=1.0)],
#     bbox_params=A.BboxParams(
#         format="albumentations",          # <-- we pass x_min,y_min,x_max,y_max (normalized)
#         label_fields=["class_labels"],
#         min_visibility=MIN_VISIBILITY,
#     ),
# )

### SCALE
# transform = A.Compose(
#     [
#         A.Affine(
#             scale=(0.8, 1.2),          # <1 = zoom out, >1 = zoom in
#             translate_percent=0.0,
#             rotate=0,
#             shear=0,
#             border_mode=cv2.BORDER_CONSTANT,
#             p=1.0,
#         ),
#     ],
#     bbox_params=A.BboxParams(
#         format="albumentations",
#         label_fields=["class_labels"],
#         min_visibility=MIN_VISIBILITY,
#     ),
# )

# ### CROP
# transform = A.Compose(
#     [
#         A.RandomSizedBBoxSafeCrop(
#             height=640,
#             width=640,
#             erosion_rate=0.05,  # allow slight trimming
#             p=1.0,
#         ),
#     ],
#     bbox_params=A.BboxParams(
#         format="albumentations",
#         label_fields=["class_labels"],
#         min_visibility=MIN_VISIBILITY,
#     ),
# )

### BRIGHTNESS/CONTRAST
transform = A.Compose(
    [
        A.RandomBrightnessContrast(
            brightness_limit=0.2,   # ±20% brightness
            contrast_limit=0.2,     # ±20% contrast
            p=1.0,
        ),
    ],
    bbox_params=A.BboxParams(
        format="albumentations",
        label_fields=["class_labels"],
    ),
)

### SHARPNESS/BLUR
transform = A.Compose(
    [
        A.OneOf(
            [
                A.MotionBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ],
            p=0.5,
        ),
        A.Sharpen(
            alpha=(0.1, 0.3),   # strength of sharpening
            lightness=(0.7, 1.3),
            p=0.5,
        ),
    ],
    bbox_params=A.BboxParams(
        format="albumentations",
        label_fields=["class_labels"],
    ),
)


def yolo_to_xyxy(box):
    # box: [xc, yc, w, h] normalized
    xc, yc, w, h = box
    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0
    return [x1, y1, x2, y2]

def xyxy_to_yolo(box):
    # box: [x1, y1, x2, y2] normalized
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    xc = x1 + w / 2.0
    yc = y1 + h / 2.0
    return [xc, yc, w, h]

def clamp01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

def sanitize_xyxy(box, eps=1e-9):
    """
    Clamp coords into [0,1] and ensure x1<x2, y1<y2.
    Returns None if invalid after clamp.
    """
    x1, y1, x2, y2 = box
    x1 = clamp01(x1)
    y1 = clamp01(y1)
    x2 = clamp01(x2)
    y2 = clamp01(y2)

    # fix inverted
    if x2 <= x1 + eps or y2 <= y1 + eps:
        return None
    return [x1, y1, x2, y2]

def load_yolo_labels(txt_path):
    boxes_xyxy = []
    labels = []
    if not txt_path.exists():
        return boxes_xyxy, labels

    for line in txt_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:])

        xyxy = yolo_to_xyxy([xc, yc, w, h])
        xyxy = sanitize_xyxy(xyxy)
        if xyxy is None:
            continue

        boxes_xyxy.append(xyxy)
        labels.append(cls)

    return boxes_xyxy, labels

def save_yolo_labels(txt_path, boxes_xyxy, labels):
    lines = []
    for cls, xyxy in zip(labels, boxes_xyxy):
        yolo = xyxy_to_yolo(xyxy)

        # final clamp just in case
        xc, yc, w, h = [clamp01(v) for v in yolo]
        # skip degenerate
        if w <= 0 or h <= 0:
            continue

        lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

def main():
    images = list(IMG_IN.glob("*.*"))
    print(f"Found {len(images)} images to augment")

    for img_path in images:
        label_path = LBL_IN / f"{img_path.stem}.txt"

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[skip] Could not read {img_path}")
            continue

        bboxes_xyxy, labels = load_yolo_labels(label_path)
        if len(bboxes_xyxy) == 0:
            continue

        for i in range(COPIES_PER_IMAGE):
            augmented = transform(image=image, bboxes=bboxes_xyxy, class_labels=labels)
            aug_img = augmented["image"]
            aug_boxes = augmented["bboxes"]
            aug_labels = augmented["class_labels"]

            if len(aug_boxes) == 0:
                continue

            out_img_name = f"{img_path.stem}_rot{i}.jpg"
            out_lbl_name = f"{img_path.stem}_rot{i}.txt"

            cv2.imwrite(str(IMG_OUT / out_img_name), aug_img)
            save_yolo_labels(LBL_OUT / out_lbl_name, aug_boxes, aug_labels)

    print("Augmentation complete")

if __name__ == "__main__":
    main()
