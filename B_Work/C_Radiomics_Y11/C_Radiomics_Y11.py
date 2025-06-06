import os
import re
import cv2
import numpy as np
import SimpleITK as sitk
import pandas as pd
from radiomics import featureextractor

# 1. Configuration
CONF_THRES = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
IOU_THRES = 0.50
IOU_OK_THRESH = 0.50

# 2. Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(".", "A_Datasets", "A_Train_Yolo", "fold_data"))
HUMAN_EXCEL = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "A_Datasets", "B_Train_Fuzzy", "Human_tagging.xlsx"))
OUTPUT_DIR = os.path.abspath(os.path.join(".", "B_WORK", "C_Radiomics_Y11"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3. Filename pattern
pattern = re.compile(r"^(\d+)_image_([^_]+)_(\d+)\.(?:jpg|png)$", re.IGNORECASE)

# 4. Extract metadata from filename
def extract_metadata(filename):
    m = pattern.match(filename)
    return (int(m.group(1)), m.group(2), int(m.group(3))) if m else None

# 5. Collect image and mask file pairs for a given fold
def collect_test_files_fold(fold_idx, mask_base, conf_thres, model_suffix):
    img_dir = os.path.join(BASE_DIR, f"round_{fold_idx}", "Test", "images")
    mask_dir = os.path.join(mask_base, f"train{fold_idx}", "masks")
    mask_pref = f"Y11{model_suffix}{fold_idx}"
    conf_code = f"c{int(conf_thres * 100):02d}"
    pairs = []

    for fn in os.listdir(img_dir):
        if not fn.lower().endswith((".jpg", ".png")):
            continue
        meta = extract_metadata(fn)
        if not meta:
            continue

        ID, patient, slice_idx = meta
        img_path = os.path.join(img_dir, fn)

        mask_candidates = []
        for ext in ("png", "jpg", "JPG", "PNG"):
            mask_candidates.append(os.path.join(mask_dir, f"{ID}_mask_{patient}_{slice_idx}.{ext}"))
            mask_candidates.append(os.path.join(mask_dir, f"{ID}_mask{mask_pref}_{patient}_{slice_idx}.{ext}"))
            mask_candidates.append(os.path.join(mask_dir, f"{ID}_mask{mask_pref}_{conf_code}_{patient}_{slice_idx}.{ext}"))

        msk_path = next((p for p in mask_candidates if os.path.exists(p)), None)

        if not msk_path:
            print(f"\n[ERROR] Mask missing for image: {fn}")
            print("Checked paths:")
            for p in mask_candidates:
                print(f"  - {p}")
            raise FileNotFoundError(f"Mask not found for {fn}")

        pairs.append((img_path, msk_path))

    return pairs

# 6. Extract radiomic features
def extract_test_features(pairs):
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    for cls in ("firstorder", "glcm", "glrlm", "glszm", "ngtdm", "shape2D"):
        extractor.enableFeatureClassByName(cls)

    records = []
    feature_names = None

    for idx, (img_path, msk_path) in enumerate(pairs, 1):
        fn = os.path.basename(img_path)
        ID, patient, slice_idx = extract_metadata(fn)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask_raw = cv2.imread(msk_path, cv2.IMREAD_UNCHANGED)
        if img is None or mask_raw is None:
            raise FileNotFoundError(f"Failed to load image or mask for {fn}")

        if mask_raw.ndim == 2:
            msk_bool = mask_raw > 0
        else:
            msk_bool = np.any(mask_raw > 0, axis=-1)

        sitk_img = sitk.GetImageFromArray(img.astype(np.float32))
        sitk_msk = sitk.GetImageFromArray(msk_bool.astype(np.uint8))

        try:
            feats = extractor.execute(sitk_img, sitk_msk)
        except ValueError:
            print(f"Empty mask for {fn}, setting features to zero")
            if feature_names is None:
                dummy = sitk.GetImageFromArray(np.ones_like(img, dtype=np.uint8))
                feature_names = list(extractor.execute(sitk_img, dummy).keys())
            feats = {name: 0 for name in feature_names}

        if feature_names is None:
            feature_names = list(feats.keys())
        values = {name: feats[name] for name in feature_names}

        coords = np.column_stack(np.where(msk_bool))
        if coords.size:
            cy, cx = coords.mean(axis=0)
        else:
            cy = cx = np.nan

        record = {
            **values,
            "Centroid_X": float(cx),
            "Centroid_Y": float(cy),
            "image_id": ID,
            "patient_id": patient,
            "true_class": slice_idx,
        }
        records.append(record)
        print(f"[{idx}/{len(pairs)}] Processed {fn}")

    return pd.DataFrame(records)

# 7. Merge with human annotation file
def merge_human(df):
    hdf = pd.read_excel(HUMAN_EXCEL, usecols=["idfile", "category"])
    hdf = hdf.rename(columns={"idfile": "image_id", "category": "plane"})
    return df.merge(hdf, on="image_id", how="left")

# Model map
model_map = {
    'nano':   'n',
    'small':  's',
    'medium': 'm',
    'large':  'l',
    'xlarge': 'x'
}

# 8. Run experiment for one CONF_THRES value and model
def run_experiment(conf_thres, model_name, iou_thres=IOU_THRES, iou_ok_thresh=IOU_OK_THRESH):
    model_suffix = model_map[model_name]
    run_folder = f"{list(model_map.keys()).index(model_name)+1}_runs_{model_name}"
    thresh_folder = f"conf{conf_thres:.2f}_iou{iou_thres:.2f}_ok{iou_ok_thresh:.2f}".replace("0.", "")
    mask_base = os.path.abspath(
        os.path.join(SCRIPT_DIR, "..", "A_Y11", "B_Y11_inference", thresh_folder, run_folder)
    )

    for fold in range(1, 6):
        print(f"\n[INFO] Running fold {fold} | CONF {conf_thres:.2f} | MODEL {model_name}")
        try:
            pairs = collect_test_files_fold(fold, mask_base, conf_thres, model_suffix)
        except FileNotFoundError as e:
            print(e)
            continue

        df_feat = extract_test_features(pairs)
        df_final = merge_human(df_feat)

        model_idx = list(model_map.keys()).index(model_name) + 1
        model_folder = f"{model_idx}_Y11_{model_name}"
        out_dir = os.path.join(OUTPUT_DIR, thresh_folder, model_folder)
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f"{fold}_test_y11{model_suffix}.csv")

        df_final.to_csv(out_csv, index=False)
        print(f"[SAVED] CSV saved for fold {fold}: {out_csv}")

# 9. Entry point
if __name__ == "__main__":
    if not isinstance(CONF_THRES, (list, tuple)):
        CONF_THRESH_LIST = [CONF_THRES]
    else:
        CONF_THRESH_LIST = CONF_THRES

    for conf in CONF_THRESH_LIST:
        for model in model_map.keys():
            run_experiment(conf_thres=conf, model_name=model)

    print("\n[DONE] All experiments completed.")
