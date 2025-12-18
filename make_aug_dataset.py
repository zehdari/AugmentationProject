import cv2
import albumentations as A
import yaml
from pathlib import Path
import argparse
import json
from configs import *

'''
Augments a yolo dataset for the provided augmentation, only modifying train. 
Creats a new data.yaml for training, using the unmodified val.
'''

# To add augmentations, and an option to build transform and the arg choice 

SPLIT_DEFAULT = "train"
COPIES_PER_IMAGE_DEFAULT = 1
MIN_VISIBILITY_DEFAULT = 0.3
IMGSZ_DEFAULT = 640

def build_transform(aug: str, p: float, imgsz: int, min_visibility: float):
    bbox_params_geo = A.BboxParams(
        format="albumentations",
        label_fields=["class_labels"],
        min_visibility=min_visibility,
    )
    bbox_params_photo = A.BboxParams(
        format="albumentations",
        label_fields=["class_labels"],
    )

    if aug == "hsv":
        return A.Compose(
            [
                A.HueSaturationValue(
                    hue_shift_limit=5,
                    sat_shift_limit=15,
                    val_shift_limit=10,
                    p=p,
                )
            ],
            bbox_params=bbox_params_photo,
        )

    if aug == "gamma":
        return A.Compose(
            [
                A.RandomGamma(
                    gamma_limit=(80, 120),
                    p=p,
                )
            ],
            bbox_params=bbox_params_photo,
        )

    if aug == "clahe":
        return A.Compose(
            [
                A.CLAHE(
                    clip_limit=(1.0, 3.0),
                    tile_grid_size=(8, 8),
                    p=p,
                )
            ],
            bbox_params=bbox_params_photo,
        )

    if aug == "gauss_noise":
        return A.Compose(
            [
                A.GaussNoise(
                    var_limit=(5.0, 25.0),
                    mean=0,
                    p=p,
                )
            ],
            bbox_params=bbox_params_photo,
        )

    if aug == "motion_blur_small":
        return A.Compose(
            [
                A.MotionBlur(
                    blur_limit=3,
                    p=p,
                )
            ],
            bbox_params=bbox_params_photo,
        )

    if aug == "affine_small":
        return A.Compose(
            [
                A.Affine(
                    scale=(0.98, 1.02),
                    translate_percent=(-0.02, 0.02),
                    rotate=(-2, 2),
                    shear=(-2, 2),
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=p,
                )
            ],
            bbox_params=bbox_params_geo,
        )

    if aug == "mirror":
        return A.Compose(
            [A.HorizontalFlip(p=p)],
            bbox_params=bbox_params_geo,
        )

    if aug == "rotate":
        return A.Compose(
            [A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT_101, p=p)],
            bbox_params=bbox_params_geo,
        )

    if aug == "zoom":
        return A.Compose(
            [
                A.Affine(
                    scale=(0.8, 1.2),
                    translate_percent=0.0,
                    rotate=0,
                    shear=0,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=p,
                )
            ],
            bbox_params=bbox_params_geo,
        )

    if aug == "crop":
        return A.Compose(
            [
                A.RandomSizedBBoxSafeCrop(
                    height=imgsz,
                    width=imgsz,
                    erosion_rate=0.05,
                    p=p,
                )
            ],
            bbox_params=bbox_params_geo,
        )

    if aug == "brightness":
        return A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.0,
                    p=p,
                )
            ],
            bbox_params=bbox_params_photo,
        )

    if aug == "contrast":
        return A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.0,
                    contrast_limit=0.2,
                    p=p,
                )
            ],
            bbox_params=bbox_params_photo,
        )

    if aug == "sharpness":
        return A.Compose(
            [A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3), p=p)],
            bbox_params=bbox_params_photo,
        )

    if aug == "blur":
        return A.Compose(
            [
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=7, p=1.0),
                        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                        A.MedianBlur(blur_limit=5, p=1.0),
                    ],
                    p=p,
                )
            ],
            bbox_params=bbox_params_photo,
        )

    if aug == "dropout":
        return A.Compose(
            [
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(0.2 * imgsz),
                    max_width=int(0.2 * imgsz),
                    min_holes=1,
                    min_height=int(0.05 * imgsz),
                    min_width=int(0.05 * imgsz),
                    fill_value=0,
                    p=p,
                )
            ],
            bbox_params=bbox_params_photo,
        )

    raise ValueError("Unknown aug")

# Writes a yaml for the augmented dataset with the modified train and original val dir
def write_modified_data_yaml(output_root: Path):
    if not SOURCE_DATA_YAML.exists():
        raise FileNotFoundError(f"Source data.yaml not found: {SOURCE_DATA_YAML}")

    cfg = yaml.safe_load(SOURCE_DATA_YAML.read_text(encoding="utf-8")) or {}

    # Make train come from the augmented dataset
    cfg["path"] = str(output_root.resolve())
    cfg["train"] = "images/train"

    # Keep val coming from the original dataset
    cfg["val"] = str((BASE_DATASET / "images" / "val").resolve())

    out_yaml = output_root.parent / f"data_{output_root.name}.yaml"
    out_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out_yaml

### Helpers

def yolo_to_xyxy(box):
    xc, yc, w, h = box
    return [xc - w / 2.0, yc - h / 2.0, xc + w / 2.0, yc + h / 2.0]

def xyxy_to_yolo(box):
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return [x1 + w / 2.0, y1 + h / 2.0, w, h]

def clamp01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

def sanitize_xyxy(box, eps=1e-9):
    x1, y1, x2, y2 = box
    x1 = clamp01(x1)
    y1 = clamp01(y1)
    x2 = clamp01(x2)
    y2 = clamp01(y2)
    if x2 <= x1 + eps or y2 <= y1 + eps:
        return None
    return [x1, y1, x2, y2]

def load_yolo_labels(txt_path: Path):
    boxes_xyxy, labels = [], []
    if not txt_path.exists():
        return boxes_xyxy, labels
    for line in txt_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:])
        xyxy = sanitize_xyxy(yolo_to_xyxy([xc, yc, w, h]))
        if xyxy is None:
            continue
        boxes_xyxy.append(xyxy)
        labels.append(cls)
    return boxes_xyxy, labels

def save_yolo_labels(txt_path: Path, boxes_xyxy, labels):
    lines = []
    for cls, xyxy in zip(labels, boxes_xyxy):
        xc, yc, w, h = [clamp01(v) for v in xyxy_to_yolo(xyxy)]
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser("Generate an augmented TRAIN set (val stays original via data.yaml)")
    ap.add_argument("--input_root", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument(
        "--aug",
        required=True,
        choices=[
            "combo_cs",
            "mirror",
            "rotate",
            "zoom",
            "crop",
            "brightness",
            "contrast",
            "sharpness",
            "blur",
            "dropout",
            "hsv",
            "gamma",
            "clahe",
            "gauss_noise",
            "motion_blur_small",
            "affine_small",
        ],
    )
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--split", default="train")  # keep train only
    ap.add_argument("--copies", type=int, default=COPIES_PER_IMAGE_DEFAULT)
    ap.add_argument("--imgsz", type=int, default=IMGSZ_DEFAULT)
    ap.add_argument("--min_visibility", type=float, default=MIN_VISIBILITY_DEFAULT)
    ap.add_argument("--ext", default=".jpg")
    args = ap.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    img_in = input_root / "images" / args.split
    lbl_in = input_root / "labels" / args.split
    img_out = output_root / "images" / args.split
    lbl_out = output_root / "labels" / args.split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    out_yaml = write_modified_data_yaml(output_root)
    print(f"Wrote data.yaml -> {out_yaml}")
    print(f"    train -> {output_root.name}/images/train")
    print(f"    val   -> Dataset/images/val (original)")

    transform = build_transform(args.aug, args.p, args.imgsz, args.min_visibility)

    images = list(img_in.glob("*.*"))
    print(f"[{args.aug} p={args.p}] Found {len(images)} train images")

    written = 0
    skipped_no_labels = 0
    skipped_empty_after = 0

    for img_path in images:
        label_path = lbl_in / f"{img_path.stem}.txt"
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        bboxes_xyxy, labels = load_yolo_labels(label_path)
        if not bboxes_xyxy:
            skipped_no_labels += 1
            continue

        for i in range(args.copies):
            augmented = transform(image=image, bboxes=bboxes_xyxy, class_labels=labels)
            aug_img = augmented["image"]
            aug_boxes = augmented.get("bboxes", [])
            aug_labels = augmented.get("class_labels", [])

            if len(aug_boxes) == 0:
                skipped_empty_after += 1
                continue

            tag = f"{args.aug}_p{str(args.p).replace('.','')}"
            out_img_name = f"{img_path.stem}_{tag}_{i}{args.ext}"
            out_lbl_name = f"{img_path.stem}_{tag}_{i}.txt"

            cv2.imwrite(str(img_out / out_img_name), aug_img)
            save_yolo_labels(lbl_out / out_lbl_name, aug_boxes, aug_labels)
            written += 1

    manifest = {
        "aug": args.aug,
        "p": args.p,
        "split": args.split,
        "copies": args.copies,
        "imgsz": args.imgsz,
        "min_visibility": args.min_visibility,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "output_data_yaml": str(out_yaml),
        "written_samples": written,
        "skipped_no_labels": skipped_no_labels,
        "skipped_empty_after_aug": skipped_empty_after,
        "val_source": "Dataset/images/val (original, not copied)",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done. Wrote {written} samples to {output_root}")
    print(f"skipped_no_labels={skipped_no_labels}, skipped_empty_after_aug={skipped_empty_after}")

if __name__ == "__main__":
    main()
