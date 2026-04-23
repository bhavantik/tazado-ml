import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List
import tensorflow as tf
from sklearn.model_selection import train_test_split

class MangoDataLoader:
    """Data loader for mango detection dataset"""
    
    MANGO_VARIETIES = ['alphonso', 'kesar', 'langda', 'other']
    IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG'}
    
    def __init__(self, data_dir: str, input_size: int = 224):
        """
        Initialize data loader
        
        Args:
            data_dir: Path to data directory (containing raw/mango/)
            input_size: Target image size
        """
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        self.raw_dir = self.data_dir / 'raw' / 'mango'
        self.processed_dir = self.data_dir / 'processed'
        self.splits_dir = self.data_dir / 'splits'
    
    def load_images_and_labels(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load all images and labels from mango directory
        
        Returns:
            (images, variety_labels, is_mango_labels)
        """
        images = []
        variety_labels = []
        
        print(f"Loading images from {self.raw_dir}...")
        
        # Load mango variety images
        for variety_idx, variety in enumerate(self.MANGO_VARIETIES):
            variety_dir = self.raw_dir / variety
            
            if not variety_dir.exists():
                print(f"⚠️  Directory not found: {variety_dir}")
                continue
            
            image_files = [f for f in variety_dir.iterdir() 
                          if f.suffix in self.IMG_EXTENSIONS]
            
            print(f"Found {len(image_files)} images in {variety}/")
            
            for img_path in image_files:
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    # Resize
                    img = cv2.resize(img, (self.input_size, self.input_size))
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Normalize to [-1, 1] for MobileNetV3
                    img = img.astype(np.float32)
                    img = (img / 127.5) - 1.0
                    
                    images.append(img)
                    variety_labels.append(variety_idx)
                
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        images = np.array(images)
        variety_labels = np.array(variety_labels)
        
        # All loaded images are mango (label = 1)
        is_mango_labels = np.ones(len(images))
        
        print(f"\n✅ Loaded {len(images)} mango images")
        print(f"Shape: {images.shape}")
        
        return images, variety_labels, is_mango_labels
    
    def create_train_val_test_split(self, test_size: float = 0.15, 
                                    val_size: float = 0.15) -> dict:
        """
        Create train/val/test splits
        
        Returns:
            Dictionary with split indices
        """
        images, variety_labels, is_mango_labels = self.load_images_and_labels()
        
        # First split: test set
        indices = np.arange(len(images))
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=42
        )
        
        # Second split: validation from train set
        train_val_size = val_size / (1 - test_size)
        train_indices, val_indices = train_test_split(
            train_indices, test_size=train_val_size, random_state=42
        )
        
        splits = {
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'images': images,
            'variety_labels': variety_labels,
            'is_mango_labels': is_mango_labels
        }
        
        print(f"\nData Split:")
        print(f"  Train: {len(train_indices)} samples")
        print(f"  Valid: {len(val_indices)} samples")
        print(f"  Test:  {len(test_indices)} samples")
        
        return splits
    
    def get_train_dataset(self, splits: dict, batch_size: int = 32) -> tf.data.Dataset:
        """Get training dataset with augmentation"""
        images = splits['images'][splits['train_indices']]
        variety_labels = splits['variety_labels'][splits['train_indices']]
        is_mango_labels = splits['is_mango_labels'][splits['train_indices']]
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            images, 
            (is_mango_labels, tf.keras.utils.to_categorical(variety_labels, num_classes=4))
        ))
        
        # Augmentation
        def augment(image, labels):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.2)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            return image, labels
        
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_val_dataset(self, splits: dict, batch_size: int = 32) -> tf.data.Dataset:
        """Get validation dataset"""
        images = splits['images'][splits['val_indices']]
        variety_labels = splits['variety_labels'][splits['val_indices']]
        is_mango_labels = splits['is_mango_labels'][splits['val_indices']]
        
        dataset = tf.data.Dataset.from_tensor_slices((
            images,
            (is_mango_labels, tf.keras.utils.to_categorical(variety_labels, num_classes=4))
        ))
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_test_dataset(self, splits: dict, batch_size: int = 32) -> tf.data.Dataset:
        """Get test dataset"""
        images = splits['images'][splits['test_indices']]
        variety_labels = splits['variety_labels'][splits['test_indices']]
        is_mango_labels = splits['is_mango_labels'][splits['test_indices']]
        
        dataset = tf.data.Dataset.from_tensor_slices((
            images,
            (is_mango_labels, tf.keras.utils.to_categorical(variety_labels, num_classes=4))
        ))
        
        dataset = dataset.batch(batch_size)
        
        return dataset
