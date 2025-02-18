import os
import shutil
import re
import random
from tqdm import tqdm

# Define the dataset path
SOURCE_DIR = "data\original\images"  # Change this to your folder with images
DEST_DIR = "data\classification_data"  # Output directory for train/val/test splits

# Split ratios
TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.1

# Ensure ratios sum to 1
assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0, "Ratios must sum to 1."

# Function to extract class name from filename
def extract_class_name(filename):
    # Extract everything until the first digit appears
    match = re.match(r"([a-zA-Z_]+?)(\d+)", filename)
    if match:
        class_name = match.group(1).rstrip("_")  # Remove trailing "_"
        return class_name
    return None

# Get all images and extract class names
all_images = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.jpg', '.png'))]
class_map = {}

for img in all_images:
    class_name = extract_class_name(img)
    if class_name:
        if class_name not in class_map:
            class_map[class_name] = []
        class_map[class_name].append(img)

# Create train, val, test directories
for split in ["train", "val", "test"]:
    for class_name in class_map.keys():
        os.makedirs(os.path.join(DEST_DIR, split, class_name), exist_ok=True)

# Split data and save
for class_name, images in class_map.items():
    random.shuffle(images)  # Shuffle images for randomness
    
    train_count = int(len(images) * TRAIN_RATIO)
    val_count = int(len(images) * VAL_RATIO)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    # Copy files to respective folders
    for img in tqdm(train_images, desc=f"Processing {class_name} - Train"):
        shutil.copy(os.path.join(SOURCE_DIR, img), os.path.join(DEST_DIR, "train", class_name, img))
    for img in tqdm(val_images, desc=f"Processing {class_name} - Val"):
        shutil.copy(os.path.join(SOURCE_DIR, img), os.path.join(DEST_DIR, "val", class_name, img))
    for img in tqdm(test_images, desc=f"Processing {class_name} - Test"):
        shutil.copy(os.path.join(SOURCE_DIR, img), os.path.join(DEST_DIR, "test", class_name, img))

print("Dataset split completed successfully!")
