import os
import re
import cv2
import numpy as np
import SimpleITK as sitk
import pandas as pd
from radiomics import featureextractor

# === CONFIG ===
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "A_Train_Yolo", "fold_data")
HUMAN_EXCEL   = os.path.abspath(os.path.join(SCRIPT_DIR, "Human_tagging.xlsx"))
OUTPUT_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, "A_XY_cooridnates_v5"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

pattern = re.compile(r"^(\d+)_image_([^_]+)_(\d+)\.(?:jpg|png)$", re.IGNORECASE)

def extract_metadata(filename):
    m = pattern.match(filename)
    return (int(m.group(1)), m.group(2), int(m.group(3))) if m else None

def collect_files(fold_idx, subset):
    img_dir  = os.path.join(BASE_DIR, f"round_{fold_idx}", subset, "images")
    mask_dir = os.path.join(BASE_DIR, f"round_{fold_idx}", subset, "masks")
    mask_pref = f"Y11n{fold_idx}"
    pairs, missing = [], []

    for fn in os.listdir(img_dir):
        if not fn.lower().endswith((".jpg", ".png")):
            continue
        meta = extract_metadata(fn)
        if not meta:
            continue

        ID, patient, slice_idx = meta
        img_path = os.path.join(img_dir, fn)

        # possibili maschere
        base1 = f"{ID}_mask_{patient}_{slice_idx}"
        base2 = f"{ID}_mask{mask_pref}_{patient}_{slice_idx}"
        msk_path = None
        for base in (base1, base2):
            for ext in ("png", "jpg"):
                cand = os.path.join(mask_dir, f"{base}.{ext}")
                if os.path.exists(cand):
                    msk_path = cand
                    break
            if msk_path:
                break

        if not msk_path:
            missing.append(fn)
        pairs.append((img_path, msk_path))

    if missing:
        raise FileNotFoundError(f"Missing masks for {len(missing)} images ({subset} fold {fold_idx}): {missing}")
    return pairs

def extract_features(pairs):
    extr = featureextractor.RadiomicsFeatureExtractor()
    extr.disableAllFeatures()
    for cls in ("firstorder","glcm","glrlm","glszm","ngtdm","shape2D"):
        extr.enableFeatureClassByName(cls)

    records, feature_names = [], None

    for idx, (img_path, msk_path) in enumerate(pairs, 1):
        fn = os.path.basename(img_path)
        ID, patient, slice_idx = extract_metadata(fn)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask_raw = cv2.imread(msk_path, cv2.IMREAD_UNCHANGED)
        if img is None or mask_raw is None:
            raise FileNotFoundError(f"Failed to load {fn}")

        if mask_raw.ndim == 2:
            msk_bool = mask_raw > 0
        else:
            msk_bool = np.any(mask_raw > 0, axis=-1)

        sitk_img = sitk.GetImageFromArray(img.astype(np.float32))
        sitk_msk = sitk.GetImageFromArray(msk_bool.astype(np.uint8))

        try:
            feats = extr.execute(sitk_img, sitk_msk)
        except ValueError:
            print(f"Empty mask for {fn}, filling zeros")
            if feature_names is None:
                dummy = sitk.GetImageFromArray(np.ones_like(img, np.uint8))
                feature_names = list(extr.execute(sitk_img, dummy).keys())
            feats = {name: 0 for name in feature_names}

        if feature_names is None:
            feature_names = list(feats.keys())
        values = {name: feats[name] for name in feature_names}

        coords = np.column_stack(np.where(msk_bool))
        cy, cx = coords.mean(axis=0) if coords.size else (np.nan, np.nan)

        record = {
            **values,
            "Centroid_X": float(cx),
            "Centroid_Y": float(cy),
            "image_id": ID,
            "patient_id": patient,
            "true_class": slice_idx,
        }
        records.append(record)
        print(f"[{idx}/{len(pairs)}] DONE {fn}")

    return pd.DataFrame(records)

def merge_human(df):
    hdf = pd.read_excel(HUMAN_EXCEL, usecols=["idfile","category"])
    hdf = hdf.rename(columns={"idfile":"image_id","category":"plane"})
    return df.merge(hdf, on="image_id", how="left")

def process_test(fold_idx):
    print(f"\n=== RUN TEST fold {fold_idx} ===")
    pairs = collect_files(fold_idx, "test")
    df  = extract_features(pairs)
    df_final = merge_human(df)
    out = os.path.join(OUTPUT_DIR, f"{fold_idx}_XY_v5_test.csv")
    df_final.to_csv(out, index=False)
    print(f"Saved: {out}")

def process_trainval(fold_idx):
    print(f"\n=== RUN TRAIN+VALIDATION fold {fold_idx} ===")
    # raccogli coppie da train e da validation
    pairs_train = collect_files(fold_idx, "train")
    pairs_val   = collect_files(fold_idx, "validation")
    # unisci le liste
    all_pairs = pairs_train + pairs_val

    df  = extract_features(all_pairs)
    df_final = merge_human(df)
    out = os.path.join(OUTPUT_DIR, f"{fold_idx}_XY_v5_train.csv")
    df_final.to_csv(out, index=False)
    print(f"Saved: {out}")

if __name__ == "__main__":
    for fold in range(1, 6):
        try:
            process_test(fold)
            process_trainval(fold)
        except FileNotFoundError as e:
            print(e)
            continue

    print("\nAll runs completed.")
