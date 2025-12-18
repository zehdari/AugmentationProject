import json
import random
import shutil
from pathlib import Path
from PIL import Image

# Paths
KITTI_ROOT = Path(r"S:\AugProject\kitti_rf_detr\train")
IMG_DIR = KITTI_ROOT / "image_2"
LBL_DIR = KITTI_ROOT / "label_2"
IMAGESETS_DIR = KITTI_ROOT / "ImageSets"

# Output dataset root
RFDETR_ROOT = KITTI_ROOT / "rfdetr_dataset"

# Split config (80/10/10)
SPLIT_SEED = 42
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1  # remaining 0.1 goes to test

# Image move strategy
MOVE_MODE = "copy"

# Classes
KEEP_CLASSES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]
SKIP_CLASSES = {"DontCare"}

CATEGORIES = [{"id": i + 1, "name": name, "supercategory": "kitti"} for i, name in enumerate(KEEP_CLASSES)]
CAT_ID = {c["name"]: c["id"] for c in CATEGORIES}


def get_all_image_stems():
    stems = [p.stem for p in IMG_DIR.glob("*.png")]
    stems += [p.stem for p in IMG_DIR.glob("*.jpg")]
    return sorted(set(stems))


def ensure_random_split_files():
    """
    Create deterministic 80/10/10 train/valid/test splits once and persist them
    under ImageSets/ as train.txt, valid.txt, test.txt.
    """
    IMAGESETS_DIR.mkdir(parents=True, exist_ok=True)

    train_file = IMAGESETS_DIR / "train.txt"
    valid_file = IMAGESETS_DIR / "valid.txt"
    test_file = IMAGESETS_DIR / "test.txt"

    # If they already exist, keep using them forever (no reshuffle)
    if train_file.exists() and valid_file.exists() and test_file.exists():
        return

    ids = get_all_image_stems()
    if not ids:
        raise RuntimeError(f"No images found in {IMG_DIR}")

    rng = random.Random(SPLIT_SEED)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(TRAIN_RATIO * n)
    n_valid = int(VALID_RATIO * n)

    train_ids = ids[:n_train]
    valid_ids = ids[n_train:n_train + n_valid]
    test_ids = ids[n_train + n_valid:]

    # Guard against weird rounding issues on very small datasets
    if len(train_ids) == 0 or len(valid_ids) == 0 or len(test_ids) == 0:
        raise RuntimeError(
            f"Split produced empty set(s): train={len(train_ids)}, valid={len(valid_ids)}, test={len(test_ids)} "
            f"(n={n}). Adjust ratios."
        )

    train_file.write_text("\n".join(train_ids) + "\n")
    valid_file.write_text("\n".join(valid_ids) + "\n")
    test_file.write_text("\n".join(test_ids) + "\n")

    print(f"[ok] created split files in {IMAGESETS_DIR}")
    print(f"     train: {len(train_ids)}")
    print(f"     valid: {len(valid_ids)}")
    print(f"     test:  {len(test_ids)}")
    print(f"     seed={SPLIT_SEED}, ratios={TRAIN_RATIO}/{VALID_RATIO}/{1.0-TRAIN_RATIO-VALID_RATIO:.1f}")


def load_split_ids(split_name: str):
    ensure_random_split_files()
    split_file = IMAGESETS_DIR / f"{split_name}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")
    return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]


def kitti_line_to_bbox(line: str):
    parts = line.strip().split()
    if len(parts) < 8:
        return None
    cls = parts[0]
    xmin, ymin, xmax, ymax = map(float, parts[4:8])
    return cls, xmin, ymin, xmax, ymax


def find_image_path(stem: str):
    png = IMG_DIR / f"{stem}.png"
    if png.exists():
        return png
    jpg = IMG_DIR / f"{stem}.jpg"
    if jpg.exists():
        return jpg
    return None


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if MOVE_MODE == "copy":
        shutil.copy2(src, dst)
    elif MOVE_MODE == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError("MOVE_MODE must be 'copy' or 'symlink'")


def build_coco_for_split(split_name: str, out_dir: Path):
    """
    Writes out_dir/_annotations.coco.json and places images in out_dir/.
    COCO 'file_name' is just the basename (since json lives in same folder).
    """
    image_ids = load_split_ids(split_name)

    coco = {
        "info": {"description": f"KITTI -> COCO ({split_name})"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": CATEGORIES,
    }

    ann_id = 1
    img_id = 1

    out_dir.mkdir(parents=True, exist_ok=True)

    for stem in image_ids:
        src_img = find_image_path(stem)
        if not src_img:
            print(f"[warn] missing image for {stem}, skipping")
            continue

        # Put image into out_dir
        dst_img = out_dir / src_img.name
        link_or_copy(src_img, dst_img)

        # Read size from source image
        with Image.open(src_img) as im:
            width, height = im.size

        coco["images"].append({
            "id": img_id,
            "file_name": dst_img.name,  # basename only
            "width": width,
            "height": height,
        })

        lbl_path = LBL_DIR / f"{stem}.txt"
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parsed = kitti_line_to_bbox(line)
                if not parsed:
                    continue
                cls, xmin, ymin, xmax, ymax = parsed

                if cls in SKIP_CLASSES or cls not in CAT_ID:
                    continue

                # clip bbox
                x1 = max(0.0, min(xmin, width - 1))
                y1 = max(0.0, min(ymin, height - 1))
                x2 = max(0.0, min(xmax, width - 1))
                y2 = max(0.0, min(ymax, height - 1))

                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= 1 or h <= 1:
                    continue

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": CAT_ID[cls],
                    "bbox": [x1, y1, w, h],
                    "area": float(w * h),
                    "iscrowd": 0,
                    "segmentation": [],
                })
                ann_id += 1

        img_id += 1

    ann_path = out_dir / "_annotations.coco.json"
    ann_path.write_text(json.dumps(coco, indent=2))
    print(f"[ok] wrote {ann_path} ({len(coco['images'])} images, {len(coco['annotations'])} anns)")


if __name__ == "__main__":
    train_dir = RFDETR_ROOT / "train"
    valid_dir = RFDETR_ROOT / "valid"
    test_dir = RFDETR_ROOT / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    build_coco_for_split("train", train_dir)
    build_coco_for_split("valid", valid_dir)
    build_coco_for_split("test", test_dir)

    print(f"[done] RF-DETR dataset at: {RFDETR_ROOT}")
    print(f"       move mode: {MOVE_MODE}")

#AUGS = ["hsv", "gamma", "clahe", "gauss_noise", "motion_blur_small", "affine_small","mirror", "rotate", "zoom", "crop", "brightness", "contrast", "sharpness", "blur", "dropout"]
AUGS = ["hsv", "gamma", "clahe", "gauss_noise", "motion_blur_small", "affine_small"]