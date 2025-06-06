import os
from ultralytics import YOLO
import yaml

dataset_path = '/vast/giustino/2_paper_MRI/YOLO_CrossValidation_DS'

models = [
    ('yolo11n-seg.pt', 'nano'),
    ('yolo11s-seg.pt', 'small'),
    ('yolo11m-seg.pt', 'medium'),
    ('yolo11l-seg.pt', 'large'),
    ('yolo11x-seg.pt', 'xlarge'),
]

device = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:5']
num_rounds = 5

for m_idx, (weights, m_label) in enumerate(models, start=1):
    model_folder = f"{m_idx}_runs_{m_label}"
    
    for fold in range(1, num_rounds + 1):
        round_dir = os.path.join(dataset_path, f'round_{fold}')
        
        train_images = os.path.join(round_dir, 'Train', 'images')
        val_images = os.path.join(round_dir, 'Validation', 'images')
        test_images = os.path.join(round_dir, 'Test', 'images')

        yaml_content = {
            'train': train_images,
            'val': val_images,
            'test': test_images,
            'nc': 3,
            'names': ['meningioma', 'glioma', 'pituitary']
        }

        yaml_file = os.path.join(round_dir, f'{model_folder}_fold{fold}.yaml')
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f)

        model = YOLO(weights)
        model.train(
            data=yaml_file,
            epochs=500,
            batch=128,
            imgsz=640,
            device=device,
            patience=15,
            project=dataset_path,
            name=os.path.join(model_folder, f"train{fold}")
        )

print("Training completed for all models and all folds.")
