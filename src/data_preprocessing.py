import os
import re
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import json

class PetDataPreprocessor:
    def __init__(self, data_dir, processed_dir, split_dir):
        """
        Initialize preprocessor with directory paths
        
        Args:
            data_dir (str): Path to original dataset
            processed_dir (str): Path to save processed images and masks
            split_dir (str): Path to save train/val/test splits
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.split_dir = Path(split_dir)
        
        # Create necessary directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.split_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for processed data
        self.processed_images_dir = self.processed_dir / 'images'
        self.processed_masks_dir = self.processed_dir / 'masks'
        self.processed_images_dir.mkdir(exist_ok=True)
        self.processed_masks_dir.mkdir(exist_ok=True)
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            (self.split_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.split_dir / split / 'masks').mkdir(parents=True, exist_ok=True)
        
        # Image processing parameters
        self.target_size = (224, 224)
        self.class_to_idx = {}

    
        
    def process_images(self):
        """Process and save all images and masks"""
        print("Processing images and masks...")
        
        # Get all image files
        image_files = list(self.data_dir.glob('images/*.jpg'))
        total_files = len(image_files)
        
        # Create class mapping
        classes = set()
        for img_path in image_files:
            class_name = self.extract_class_name(img_path.stem)
            classes.add(class_name)
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(classes))}
        
        # Save class mapping
        with open(self.processed_dir / 'class_mapping.json', 'w') as f:
            json.dump(self.class_to_idx, f)
        
        # Process each image and its mask
        for idx, img_path in enumerate(image_files):
            if idx % 100 == 0:
                print(f"Processing {idx}/{total_files} images...")
            
            # Get corresponding mask path
            mask_path = self.data_dir / 'annotations' / f"{img_path.stem}.png"
            
            # Process and save image
            img = Image.open(img_path).convert('RGB')
            img = img.resize(self.target_size, Image.Resampling.LANCZOS)
            img.save(self.processed_images_dir / img_path.name)
            
            # Process and save mask
            if mask_path.exists():
                mask = Image.open(mask_path).convert('L')
                mask = mask.resize(self.target_size, Image.Resampling.NEAREST)
                # Convert trimap to binary mask (1 for pet, 0 for background)
                mask_arr = np.array(mask)
                binary_mask = np.where(mask_arr == 2, 0, 1).astype(np.uint8)
                mask = Image.fromarray(binary_mask * 255)
                mask.save(self.processed_masks_dir / f"{img_path.stem}.png")
        
        print("Processing completed!")
    

    def extract_class_name(self, filename):
        print("Extracting class name from:", filename)
        match = re.match(r"([^\d]+)", filename)
        return match.group(1).strip().rstrip('_') if match else filename.rstrip('_')


    def create_splits(self, train_ratio=0.7, val_ratio=0.15):
        """Create train/val/test splits"""
        print("Creating dataset splits...")
        
        # Get all processed images
        image_files = list(self.processed_images_dir.glob('*.jpg'))
        total_files = len(image_files)
        
        # Calculate split sizes
        train_size = int(total_files * train_ratio)
        val_size = int(total_files * val_ratio)
        test_size = total_files - train_size - val_size
        
        # Randomly shuffle files
        np.random.shuffle(image_files)
        
        # Split files
        train_files = image_files[:train_size]
        val_files = image_files[train_size:train_size + val_size]
        test_files = image_files[train_size + val_size:]
        
        # Create splits
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        # Save splits
        for split_name, files in splits.items():
            print(f"Processing {split_name} split...")
            split_images_dir = self.split_dir / split_name / 'images'
            split_masks_dir = self.split_dir / split_name / 'masks'
            
            for img_path in files:
                # Copy image
                shutil.copy2(
                    img_path,
                    split_images_dir / img_path.name
                )
                
                # Copy mask if exists
                mask_path = self.processed_masks_dir / f"{img_path.stem}.png"
                if mask_path.exists():
                    shutil.copy2(
                        mask_path,
                        split_masks_dir / mask_path.name
                    )
        
        # Save split information
        split_info = {
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size
        }
        with open(self.split_dir / 'split_info.json', 'w') as f:
            json.dump(split_info, f)
        
        print("Split creation completed!")
    
    def process_dataset(self):
        """Run complete preprocessing pipeline"""
        self.process_images()
        self.create_splits()

class PetDataset(Dataset):
    """Custom Dataset for loading processed pet images"""
    def __init__(self, data_dir, transform=None, target_transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_transform = target_transform
        
        # Load class mapping
        with open(self.data_dir.parent.parent / 'processed' / 'class_mapping.json', 'r') as f:
            self.class_to_idx = json.load(f)
        
        # Get all image files
        self.image_files = list(self.data_dir.glob('images/*.jpg'))
        self.mask_files = {
            path.stem: path for path in self.data_dir.glob('masks/*.png')
        }
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        class_name = img_path.stem.split('_')[0]
        class_idx = self.class_to_idx[class_name]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Load mask if exists
        mask_path = self.mask_files.get(img_path.stem)
        mask = None
        if mask_path:
            mask = Image.open(mask_path).convert('L')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform and mask:
            mask = self.target_transform(mask)
        
        return image, mask, class_idx

if __name__ == "__main__":
    # Example usage
    preprocessor = PetDataPreprocessor(
        data_dir="data\original",
        processed_dir="data\output\processed",
        split_dir="data\output\splits"
    )
    preprocessor.process_dataset()