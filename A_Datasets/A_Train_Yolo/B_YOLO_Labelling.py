import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

base_dir = "fold_data"
round_folders = [f"round_{i}" for i in range(1, 6)]
subsets = ["Train", "Validation", "Test"]

def get_class_from_filename(filename):
    cls_str = filename.split('_')[-1].split('.')[0]
    return int(cls_str) - 1

def contours_to_yolo_format(contours, img_shape, class_id, min_area=100, epsilon_ratio=0.001):
    h, w = img_shape
    yolo_labels = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon=epsilon_ratio * cv2.arcLength(cnt, True), closed=True)
        if len(approx) < 3:
            continue
        normalized = approx.squeeze().astype(float)
        normalized[:, 0] /= w
        normalized[:, 1] /= h
        flat = normalized.flatten()
        coords = ' '.join([f'{p:.4f}' for p in flat])
        label_line = f"{class_id} {coords}"
        yolo_labels.append(label_line)
    return yolo_labels

def process_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape
    class_id = get_class_from_filename(os.path.basename(mask_path))
    binary_mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours_to_yolo_format(contours, (h, w), class_id)

def draw_yolo_mask(yolo_lines, shape):
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for line in yolo_lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        points = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)
        points[:, 0] *= w
        points[:, 1] *= h
        poly = np.round(points).astype(np.int32)
        cv2.fillPoly(mask, [poly], color=int(parts[0]) + 1)
    return mask

def convert_masks_in_folder(mask_folder):
    mask_files = [f for f in os.listdir(mask_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    full_paths = []

    for fname in tqdm(mask_files, desc="Converting " + mask_folder, leave=False):
        fpath = os.path.join(mask_folder, fname)
        yolo_labels = process_mask(fpath)

        if yolo_labels:
            labels_dir = os.path.join(os.path.dirname(mask_folder), 'labels')
            os.makedirs(labels_dir, exist_ok=True)

            base_name = os.path.splitext(fname)[0].replace("mask", "image")  # <-- modificato qui
            out_path = os.path.join(labels_dir, base_name + ".txt")

            with open(out_path, 'w') as f:
                f.write("\n".join(yolo_labels))

            full_paths.append(fpath)
        else:
            print(f"Skipped (no valid contours): {fpath}")

    return full_paths

def visualize_sample(sample_paths):
    sample = random.sample(sample_paths, min(10, len(sample_paths)))
    for path in sample:
        original_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        h, w = original_mask.shape

        labels_dir = os.path.join(os.path.dirname(path).replace("masks", "labels"))
        txt_name = os.path.splitext(os.path.basename(path))[0].replace("mask", "image") + '.txt'
        txt_path = os.path.join(labels_dir, txt_name)

        if not os.path.exists(txt_path):
            print(f"Missing label: {txt_path}")
            continue

        with open(txt_path, 'r') as f:
            yolo_lines = f.readlines()

        reconstructed = draw_yolo_mask(yolo_lines, (h, w))

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(original_mask, cmap='gray')
        axs[0].set_title("Original Mask\n" + os.path.basename(path))
        axs[0].axis("off")

        axs[1].imshow(reconstructed, cmap='gray')
        axs[1].set_title("Reconstructed from YOLO")
        axs[1].axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    all_processed = []
    for round_folder in round_folders:
        for subset in subsets:
            mask_path = os.path.join(base_dir, round_folder, subset, "masks")
            if os.path.isdir(mask_path):
                processed_paths = convert_masks_in_folder(mask_path)
                all_processed.extend(processed_paths)

    if all_processed:
        print("Showing sample converted masks...")
        visualize_sample(all_processed)
    else:
        print("No masks found or converted.")
