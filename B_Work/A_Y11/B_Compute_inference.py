import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import re
from sklearn.metrics import jaccard_score

FOLD_BASE        = os.path.abspath(os.path.join(".", "\A_Datasets", "A_Train_Yolo", "fold_data"))
BASE_DIR         = os.path.abspath(os.path.join(".", "B_Work", "A_Y11"))

DIMENSIONS       = ["1_runs_nano","2_runs_small","3_runs_medium","4_runs_large","5_runs_xlarge"]
MODEL_TRAINS     = ["null","train1","train2","train3","train4","train5"]

IMG_SIZE         = 640
CONF_THRESH_LIST = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
IOU_THRES        = 0.50
IOU_OK_THRESH    = 0.50

CLASS_NAMES = {
    1: "Meningioma",
    2: "Glioma",
    3: "Pituitary"
}

pattern = re.compile(r'^(\d+)_image_([^_]+)_(\d+)\.(?:jpg|png)$', re.IGNORECASE)

def load_mask_bool(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError("Cannot read mask: " + path)
    return m > 0

def dim_code(dim):
    return dim.split("_")[-1][0]

def train_code(train_name):
    return train_name.replace("train","")

def process_combo(dim_idx, train_idx, conf_thres):
    dim    = DIMENSIONS[dim_idx]
    train  = MODEL_TRAINS[train_idx]
    combo  = f"{dim_code(dim)}{train_code(train)}_c{int(conf_thres*100)}"

    input_dir   = os.path.join(FOLD_BASE, f"round_{train_idx}", "Test", "images")
    gt_mask_dir = os.path.join(FOLD_BASE, f"round_{train_idx}", "Test", "masks")

    thresh_folder = f"conf{conf_thres:.2f}_iou{IOU_THRES:.2f}_ok{IOU_OK_THRESH:.2f}".replace("0.", "")
    out_dir       = os.path.join(BASE_DIR, "B_Y11_inference", thresh_folder, dim, train)
    masks_dir     = os.path.join(out_dir, "masks")
    overlay_dir   = os.path.join(out_dir, "prediction")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    print(f"[DIM={dim} TRAIN={train} CONF={conf_thres}] Masks output dir: {masks_dir}")

    weights = os.path.join(BASE_DIR, "A_Y11_Models", dim, train, "weights", "best.pt")
    if not os.path.isfile(weights):
        raise FileNotFoundError("Weights not found: " + weights)
    model = YOLO(weights)

    files = sorted(os.listdir(input_dir))
    per_image = []

    for fname in files:
        m = pattern.match(fname)
        if not m:
            continue
        img_id_s, pat_id, true_cls_s = m.groups()
        img_id, true_cls = int(img_id_s), int(true_cls_s)

        img_path = os.path.join(input_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print("Cannot load image", img_path)
            continue

        res = model(img, imgsz=IMG_SIZE, conf=conf_thres, iou=IOU_THRES, task="segment")[0]
        if res.masks is None or len(res.masks.data) == 0:
            mask     = np.zeros(img.shape[:2], dtype=np.uint8)
            pred_cls = 0
            prob     = 0.0
        else:
            confs    = res.boxes.conf.cpu().numpy()
            idx      = int(np.argmax(confs))
            prob     = float(confs[idx])
            cls_i    = int(res.boxes.cls.cpu().numpy()[idx])
            pred_cls = cls_i + 1
            md       = res.masks.data.cpu().numpy()[idx]
            mask     = (md > 0.5).astype(np.uint8) * 255

        gt_name = f"{img_id}_mask_{pat_id}_{true_cls}.jpg"
        gt_p    = os.path.join(gt_mask_dir, gt_name)
        if os.path.isfile(gt_p):
            h, w = cv2.imread(gt_p, cv2.IMREAD_GRAYSCALE).shape
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        mask_name = f"{img_id}_maskY11{combo}_{pat_id}_{true_cls}.jpg"
        save_path = os.path.join(masks_dir, mask_name)
        if cv2.imwrite(save_path, mask):
            print("Saved mask for", fname)
        else:
            print("Failed to save mask for", fname)

        overlay = img.copy()
        h_img, w_img = img.shape[:2]
        overlay_mask_pred = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

        if os.path.isfile(gt_p):
            gt_mask_img     = cv2.imread(gt_p, cv2.IMREAD_GRAYSCALE)
            overlay_mask_gt = cv2.resize(gt_mask_img, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        else:
            overlay_mask_gt = np.zeros((h_img, w_img), dtype=np.uint8)

        colored_mask = np.zeros_like(img)
        colored_mask[:, :, 1] = overlay_mask_pred
        colored_mask[:, :, 0] = overlay_mask_gt
        overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.4, 0)

        cls_ok     = (pred_cls == true_cls)
        class_name = CLASS_NAMES.get(pred_cls, f"Cls{pred_cls}")
        text       = f"Pred: {class_name} ({prob:.2f})"
        text_color = (0, 255, 0) if cls_ok else (0, 0, 255)
        cv2.putText(
            overlay, text,
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=text_color,
            thickness=2,
            lineType=cv2.LINE_AA
        )

        overlay_name = f"{img_id}_overlayY11{combo}_{pat_id}_{true_cls}.jpg"
        overlay_path = os.path.join(overlay_dir, overlay_name)
        cv2.imwrite(overlay_path, overlay)

        if os.path.isfile(gt_p):
            gt_mask   = load_mask_bool(gt_p)
            pred_mask = load_mask_bool(save_path)
            y_true = gt_mask.flatten()
            y_pred = pred_mask.flatten()
            tp = int(np.logical_and(pred_mask, gt_mask).sum())
            fp = int(np.logical_and(pred_mask, ~gt_mask).sum())
            fn = int(np.logical_and(~pred_mask, gt_mask).sum())
            iou  = jaccard_score(y_true, y_pred, zero_division=0)
            dice = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0
            seg_ok = (iou >= IOU_OK_THRESH)

            per_image.append({
                "image_id":         img_id,
                "patient_id":       pat_id,
                "true_class":       true_cls,
                "Y11_pred_class":   pred_cls,
                "Y11_probability":  prob,
                "Y11_IoU":          iou,
                "Y11_Dice":         dice,
                "Seg_OK":           seg_ok,
                "Cls_OK":           cls_ok
            })

    metrics_csv  = f"metrics_Y11{combo}.csv"
    metrics_path = os.path.join(out_dir, metrics_csv)
    df_met = pd.DataFrame(per_image)
    df_met.to_csv(metrics_path, sep=";", index=False)
    print(f"Saved metrics at {metrics_path}")

def main():
    for dim_idx in range(len(DIMENSIONS)):
        for train_idx in range(1, len(MODEL_TRAINS)):
            for conf_thres in CONF_THRESH_LIST:
                process_combo(dim_idx, train_idx, conf_thres)

if __name__ == "__main__":
    main()
