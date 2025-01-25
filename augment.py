import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os

def augment_image(image_path, save_dir):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=15, p=0.5),
        ToTensorV2(),
    ])
    image = cv2.imread(image_path)
    augmented = transform(image=image)["image"]
    cv2.imwrite(os.path.join(save_dir, os.path.basename(image_path)), augmented)

input_dir = "data/original_images"
output_dir = "data/augmented_images"
os.makedirs(output_dir, exist_ok=True)

for img_file in os.listdir(input_dir):
    augment_image(os.path.join(input_dir, img_file), output_dir)
