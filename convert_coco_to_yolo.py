import json
import shutil
from pathlib import Path

# Input dataset root that contains split folders: train/, valid/, test/
DATASET_ROOT = Path(r"S:\AugProject\kitti_rf_detr\train\rfdetr_dataset")

# Output root
OUT_ROOT = DATASET_ROOT

SPLITS = ["train", "valid", "test"]
SPLIT_NAME_MAP = {"valid": "val"}   # output naming

WRITE_EMPTY_LABELS = True
SKIP_MISSING_IMAGES = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def coco_to_yolo_box(bbox, img_w, img_h):
    """COCO bbox [x,y,w,h] -> YOLO [xc,yc,w,h] normalized."""
    x, y, w, h = bbox
    xc = x + w / 2.0
    yc = y + h / 2.0
    return (xc / img_w, yc / img_h, w / img_w, h / img_h)

def resolve_image_path(split_dir: Path, file_name: str) -> Path | None:
    p = Path(file_name)

    cand = split_dir / p
    if cand.exists():
        return cand

    for alt in [
        split_dir / "images" / p.name,
        split_dir / p.name,
    ]:
        if alt.exists():
            return alt

    matches = list(split_dir.rglob(p.name))
    if matches:
        return matches[0]

    return None

def load_categories_for_consistent_mapping():
    """
    Build one consistent (COCO cat_id -> YOLO index) mapping (prefer train split).
    Returns:
      cat_id_to_yolo: dict
      cats_sorted: list of categories sorted by COCO id
    """
    preferred = DATASET_ROOT / "train" / "_annotations.coco.json"
    ann = preferred if preferred.exists() else None

    if ann is None:
        for s in SPLITS:
            cand = DATASET_ROOT / s / "_annotations.coco.json"
            if cand.exists():
                ann = cand
                break
    if ann is None:
        raise FileNotFoundError("Could not find any _annotations.coco.json to read categories from.")

    coco = json.loads(ann.read_text(encoding="utf-8"))
    cats = coco.get("categories", [])
    cats_sorted = sorted(cats, key=lambda c: c["id"])
    cat_id_to_yolo = {c["id"]: i for i, c in enumerate(cats_sorted)}
    return cat_id_to_yolo, cats_sorted

def write_data_yaml(out_root: Path, cats_sorted: list, have_test: bool):
    """
    Writes data.yaml like:
      train: Dataset/images/train
      val: Dataset/images/val
      test: Dataset/images/test
      nc: N
      names: [...]
    """
    names = [c.get("name", f"class{i}") for i, c in enumerate(cats_sorted)]
    nc = len(names)

    lines = [
        "# data.yaml",
        "train: Dataset/images/train",
        "val: Dataset/images/val",
    ]
    if have_test:
        lines.append("test: Dataset/images/test  # Optional test path")
    lines += [
        "",
        f"nc: {nc}  # number of classes",
        "names: [" + ", ".join([f"'{n}'" for n in names]) + "]",
        "",
    ]
    (out_root / "data.yaml").write_text("\n".join(lines), encoding="utf-8")

def main():
    dataset_out = OUT_ROOT / "Dataset"
    (dataset_out / "images").mkdir(parents=True, exist_ok=True)
    (dataset_out / "labels").mkdir(parents=True, exist_ok=True)

    cat_id_to_yolo, cats_sorted = load_categories_for_consistent_mapping()

    # Track whether we actually processed test
    processed_splits_outnames = set()

    for split in SPLITS:
        split_dir = DATASET_ROOT / split
        ann_path = split_dir / "_annotations.coco.json"
        if not ann_path.exists():
            print(f"[skip] Missing COCO json: {ann_path}")
            continue

        out_split = SPLIT_NAME_MAP.get(split, split)
        processed_splits_outnames.add(out_split)

        out_images_dir = dataset_out / "images" / out_split
        out_labels_dir = dataset_out / "labels" / out_split
        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_labels_dir.mkdir(parents=True, exist_ok=True)

        coco = json.loads(ann_path.read_text(encoding="utf-8"))

        images = {img["id"]: img for img in coco.get("images", [])}

        ann_by_image = {}
        for ann in coco.get("annotations", []):
            if ann.get("iscrowd", 0) == 1:
                continue
            img_id = ann["image_id"]
            ann_by_image.setdefault(img_id, []).append(ann)

        label_written = 0
        img_copied = 0

        for img_id, img in images.items():
            img_w, img_h = img["width"], img["height"]
            file_name = img["file_name"]

            src_img = resolve_image_path(split_dir, file_name)
            dst_img = out_images_dir / Path(file_name).name

            stem = dst_img.stem
            out_txt = out_labels_dir / f"{stem}.txt"

            anns = ann_by_image.get(img_id, [])
            lines = []

            for ann in anns:
                cat_id = ann["category_id"]
                if cat_id not in cat_id_to_yolo:
                    continue
                cls = cat_id_to_yolo[cat_id]

                bbox = ann["bbox"]
                _, _, w, h = bbox
                if w <= 0 or h <= 0:
                    continue

                xc, yc, wn, hn = coco_to_yolo_box(bbox, img_w, img_h)

                xc = min(max(xc, 0.0), 1.0)
                yc = min(max(yc, 0.0), 1.0)
                wn = min(max(wn, 0.0), 1.0)
                hn = min(max(hn, 0.0), 1.0)

                lines.append(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

            if lines or WRITE_EMPTY_LABELS:
                out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                label_written += 1

            if src_img and src_img.exists():
                if src_img.suffix.lower() in IMAGE_EXTS or src_img.suffix:
                    shutil.copy2(src_img, dst_img)
                    img_copied += 1
            else:
                msg = f"[warn] Image not found for COCO file_name='{file_name}' in split '{split}'"
                if SKIP_MISSING_IMAGES:
                    print(msg)
                else:
                    raise FileNotFoundError(msg)

        print(f"[ok] {split} -> {out_split}")
        print(f"     copied images: {img_copied} -> {out_images_dir}")
        print(f"     wrote labels : {label_written} -> {out_labels_dir}")

    have_test = "test" in processed_splits_outnames
    write_data_yaml(OUT_ROOT, cats_sorted, have_test=have_test)

    print(f"\n[ok] wrote data.yaml -> {OUT_ROOT / 'data.yaml'}")
    print("classes (YOLO idx -> name):")
    for i, c in enumerate(cats_sorted):
        print(f"  {i}: {c.get('name')}")

if __name__ == "__main__":
    main()
