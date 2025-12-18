import cv2
import albumentations as A
import numpy as np
from pathlib import Path
import math
import random

s="6812"
# =========================
# CHANGE THESE TWO STRINGS
# =========================
INPUT_IMAGE = r"S:\AugProject\kitti_rf_detr\train\rfdetr_dataset\Dataset\images\val\00" + s + ".png"
OUT_DIR     = r"C:\Users\Chef\Downloads"

# If your label file isn't exactly image_stem + ".txt", set it explicitly:
# INPUT_LABEL = r"C:\path\to\image.txt"
INPUT_LABEL = r"S:\AugProject\kitti_rf_detr\train\rfdetr_dataset\Dataset\labels\val\00" + s + ".txt"  # auto: same folder as image, same stem, .txt

# =========================
# Hardcoded settings
# =========================
SEED = 42
MIN_VISIBILITY = 0.30
TILE_PAD = 8

# choose endpoint only
STRENGTH_END = "high"  # or "low"

AUG_ORDER = [
    "original",
    "mirror",
    "rotate",
    "zoom",
    "crop",
    "affine",
    "brightness",
    "contrast",
    "sharpness",
    "blur",
    "dropout",
    "hsv",
    "gamma",
    "clahe",
    "gauss_noise",
    "motion_blur"
]

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def yolo_to_xyxy(box):
    xc, yc, w, h = box
    return [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]

def xyxy_to_yolo(box):
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return [x1 + w / 2, y1 + h / 2, w, h]

def clamp01(v): return max(0.0, min(1.0, v))

def sanitize_xyxy(box, eps=1e-9):
    x1, y1, x2, y2 = map(clamp01, box)
    if x2 <= x1 + eps or y2 <= y1 + eps:
        return None
    return [x1, y1, x2, y2]

def load_yolo_labels(txt):
    boxes, labels = [], []
    if not txt.exists():
        return boxes, labels
    for line in txt.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:])
        xyxy = sanitize_xyxy(yolo_to_xyxy([xc, yc, w, h]))
        if xyxy:
            boxes.append(xyxy)
            labels.append(cls)
    return boxes, labels

def save_yolo_labels(path, boxes, labels):
    lines = []
    for c, b in zip(labels, boxes):
        xc, yc, w, h = map(clamp01, xyxy_to_yolo(b))
        if w > 0 and h > 0:
            lines.append(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

def bbox_geo():
    return A.BboxParams(
        format="albumentations",
        label_fields=["class_labels"],
        min_visibility=MIN_VISIBILITY,
    )

def bbox_photo():
    return A.BboxParams(
        format="albumentations",
        label_fields=["class_labels"],
    )

def end(lo, hi):  # pick endpoint only
    return hi if STRENGTH_END == "high" else lo

def build_transform(aug, h, w):
    # Endpoint-only transforms. No probabilities.

    if aug == "hsv":
        hs = end(-5, 5)
        ss = end(-15, 15)
        vs = end(-10, 10)
        return A.Compose(
            [A.HueSaturationValue(
                hue_shift_limit=(hs, hs),
                sat_shift_limit=(ss, ss),
                val_shift_limit=(vs, vs),
                p=1.0
            )],
            bbox_params=bbox_photo()
        )

    if aug == "gamma":
        g = end(80, 120)
        return A.Compose([A.RandomGamma(gamma_limit=(g, g), p=1.0)], bbox_params=bbox_photo())

    if aug == "clahe":
        c = end(1.0, 3.0)
        return A.Compose([A.CLAHE(clip_limit=(c, c), tile_grid_size=(8, 8), p=1.0)], bbox_params=bbox_photo())

    if aug == "gauss_noise":
        # Albumentations v2 uses std_range in [0..1] (fraction of max intensity).
        # Endpoints only:
        #   low  -> subtle noise
        #   high -> strong noise
        sr = end(0.02, 0.10)
        return A.Compose([A.GaussNoise(std_range=(sr, sr), mean_range=(0.0, 0.0), per_channel=True, p=1.0)],
                         bbox_params=bbox_photo())

    if aug == "motion_blur":
        # Endpoint only: kernel size
        k = 7 if STRENGTH_END == "high" else 3
        return A.Compose([A.MotionBlur(blur_limit=(k, k), p=1.0)], bbox_params=bbox_photo())

    if aug == "affine":
        sc = end(0.98, 1.02)
        tp = end(-0.02, 0.02)
        r  = end(-2, 2)
        sh = end(-2, 2)
        return A.Compose(
            [A.Affine(
                scale=(sc, sc),
                translate_percent=(tp, tp),
                rotate=(r, r),
                shear=(sh, sh),
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0
            )],
            bbox_params=bbox_geo()
        )

    if aug == "mirror":
        return A.Compose([A.HorizontalFlip(p=1.0)], bbox_params=bbox_geo())

    if aug == "rotate":
        r = end(-30, 30)
        return A.Compose([A.Rotate(limit=(r, r), border_mode=cv2.BORDER_REFLECT_101, p=1.0)], bbox_params=bbox_geo())

    if aug == "zoom":
        sc = end(0.8, 1.2)
        return A.Compose(
            [A.Affine(scale=(sc, sc), translate_percent=0.0, rotate=0, shear=0,
                      border_mode=cv2.BORDER_REFLECT_101, p=1.0)],
            bbox_params=bbox_geo()
        )

    if aug == "crop":
        # crop to original size (no resizing)
        return A.Compose([A.RandomSizedBBoxSafeCrop(height=h, width=w, erosion_rate=0.05, p=1.0)], bbox_params=bbox_geo())

    if aug == "brightness":
        b = end(-0.2, 0.2)
        return A.Compose([A.RandomBrightnessContrast(brightness_limit=(b, b), contrast_limit=(0.0, 0.0), p=1.0)],
                         bbox_params=bbox_photo())

    if aug == "contrast":
        c = end(-0.2, 0.2)
        return A.Compose([A.RandomBrightnessContrast(brightness_limit=(0.0, 0.0), contrast_limit=(c, c), p=1.0)],
                         bbox_params=bbox_photo())

    if aug == "sharpness":
        a = end(0.1, 0.3)
        l = end(0.7, 1.3)
        return A.Compose([A.Sharpen(alpha=(a, a), lightness=(l, l), p=1.0)], bbox_params=bbox_photo())

    if aug == "blur":
        k = 7 if STRENGTH_END == "high" else 3
        return A.Compose([A.GaussianBlur(blur_limit=(k, k), p=1.0)], bbox_params=bbox_photo())

    if aug == "dropout":
        s = min(h, w)
        max_frac = 0.20 if STRENGTH_END == "high" else 0.10
        min_frac = 0.05 if STRENGTH_END == "high" else 0.03
        return A.Compose([A.CoarseDropout(
            max_holes=8,
            max_height=int(max_frac * s),
            max_width=int(max_frac * s),
            min_height=int(min_frac * s),
            min_width=int(min_frac * s),
            fill_value=0,
            p=1.0
        )], bbox_params=bbox_photo())

    raise ValueError(aug)

def draw_label_top_centered(img, text):
    out = img.copy()
    H, W = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.35
    thickness = 3
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    bar_h = th + 26
    cv2.rectangle(out, (0, 0), (W, bar_h), (0, 0, 0), -1)
    cv2.putText(out, text, ((W - tw) // 2, th + 16), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out

def _bgr_random(rng):
    # avoid super-dark colors
    return (rng.randint(40, 255), rng.randint(40, 255), rng.randint(40, 255))

def draw_boxes(img, boxes, labels):
    h, w = img.shape[:2]
    out = img.copy()

    # Encounter-order palette (BGR for OpenCV)
    palette = [
        (180, 105, 255),  # pink
        (0, 0, 255),      # red
        (255, 0, 0),      # blue
    ]

    rng = random.Random(SEED)  # deterministic "random" colors
    class_to_color = {}
    seen_order = []

    for c in labels:
        if c not in class_to_color:
            if len(seen_order) < len(palette):
                class_to_color[c] = palette[len(seen_order)]
            else:
                class_to_color[c] = _bgr_random(rng)
            seen_order.append(c)

    for c, (x1, y1, x2, y2) in zip(labels, boxes):
        p1 = (int(x1 * w), int(y1 * h))
        p2 = (int(x2 * w), int(y2 * h))
        cv2.rectangle(out, p1, p2, class_to_color[c], 2)

    return out


def main():
    seed_everything(SEED)

    img_p = Path(INPUT_IMAGE)
    lbl_p = img_p.with_suffix(".txt") if INPUT_LABEL is None else Path(INPUT_LABEL)
    out_d = Path(OUT_DIR)
    out_d.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(img_p))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_p}")

    boxes, labels = load_yolo_labels(lbl_p)
    if not boxes:
        print(f"WARNING: No labels found in {lbl_p} (will still save images).")

    h, w = img.shape[:2]

    # store vis images for tiling
    vis_list = []

    orig_vis = draw_label_top_centered(draw_boxes(img, boxes, labels), "original")
    cv2.imwrite(str(out_d / f"{img_p.stem}_original_vis.jpg"), orig_vis)
    vis_list.append(orig_vis)

    for aug in AUG_ORDER:
        if aug == "original":
            continue
        t = build_transform(aug, h, w)
        r = t(image=img, bboxes=boxes, class_labels=labels)

        ai = r["image"]
        ab = list(r.get("bboxes", []))
        al = list(r.get("class_labels", []))

        cv2.imwrite(str(out_d / f"{img_p.stem}_{aug}.jpg"), ai)
        save_yolo_labels(out_d / f"{img_p.stem}_{aug}.txt", ab, al)

        vis = draw_label_top_centered(draw_boxes(ai, ab, al), aug)
        cv2.imwrite(str(out_d / f"{img_p.stem}_{aug}_vis.jpg"), vis)
        vis_list.append(vis)

    # tile (no resizing; cells are max dims)
    cols = int(math.ceil(math.sqrt(len(vis_list))))
    cell_h = max(v.shape[0] for v in vis_list)
    cell_w = max(v.shape[1] for v in vis_list)

    rows = math.ceil(len(vis_list) / cols)
    canvas_h = rows * cell_h + (rows + 1) * TILE_PAD
    canvas_w = cols * cell_w + (cols + 1) * TILE_PAD
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for i, v in enumerate(vis_list):
        r = i // cols
        c = i % cols
        y0 = TILE_PAD + r * (cell_h + TILE_PAD)
        x0 = TILE_PAD + c * (cell_w + TILE_PAD)
        canvas[y0:y0 + v.shape[0], x0:x0 + v.shape[1]] = v

    tiled_path = out_d / f"{img_p.stem}_TILED_{STRENGTH_END}.jpg"
    cv2.imwrite(str(tiled_path), canvas)

    print("Done.")
    print("Tiled:", tiled_path)

if __name__ == "__main__":
    main()
