import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass
import json

@dataclass
class PredictionResult:
    """Structure for prediction results"""
    is_mango: bool
    mango_confidence: float
    variety: str = None
    variety_confidence: float = None
    message: str = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'is_mango': self.is_mango,
            'mango_confidence': float(self.mango_confidence),
            'variety': self.variety,
            'variety_confidence': float(self.variety_confidence) if self.variety_confidence else None,
            'message': self.message
        }
    
    def __str__(self) -> str:
        if self.is_mango:
            return f"Mango detected! Variety: {self.variety} ({self.variety_confidence:.2%} confidence)"
        else:
            return f"Not a mango (confidence: {self.mango_confidence:.2%})"

class MangoPredictor:
    """Two-stage mango detector and variety classifier"""
    
    MANGO_VARIETIES = ['alphonso', 'kesar', 'langda', 'other']
    MANGO_THRESHOLD = 0.5
    
    def __init__(self, stage1_model_path: str, stage2_model_path: str, input_size: int = 224):
        """
        Initialize predictor with two models
        
        Args:
            stage1_model_path: Path to mango detection model
            stage2_model_path: Path to variety classification model
            input_size: Input image size
        """
        self.input_size = input_size
        self.stage1_model = tf.keras.models.load_model(stage1_model_path, compile=False)
        self.stage2_model = tf.keras.models.load_model(stage2_model_path, compile=False)
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for prediction
        
        Args:
            image_path: Path to image
            
        Returns:
            Preprocessed image array
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, (self.input_size, self.input_size))
        
        # Normalize to [-1, 1] for MobileNetV3
        img = img.astype(np.float32)
        img = (img / 127.5) - 1.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, image_path: str) -> PredictionResult:
        """
        Predict if image is mango and classify variety if it is
        
        Args:
            image_path: Path to image
            
        Returns:
            PredictionResult object
        """
        # Preprocess
        preprocessed_img = self.preprocess_image(image_path)
        
        # Stage 1: Detect mango
        stage1_pred = self.stage1_model.predict(preprocessed_img, verbose=0)
        is_mango_prob = stage1_pred[0][0]
        is_mango = is_mango_prob >= self.MANGO_THRESHOLD
        
        if not is_mango:
            return PredictionResult(
                is_mango=False,
                mango_confidence=1 - is_mango_prob,
                message="Model not trained for this object. Please upload mango image."
            )
        
        # Stage 2: Classify variety
        stage2_pred = self.stage2_model.predict(preprocessed_img, verbose=0)
        variety_idx = np.argmax(stage2_pred[0])
        variety = self.MANGO_VARIETIES[variety_idx]
        variety_confidence = stage2_pred[0][variety_idx]
        
        return PredictionResult(
            is_mango=True,
            mango_confidence=is_mango_prob,
            variety=variety,
            variety_confidence=variety_confidence,
            message=f"Detected: {variety.upper()} mango"
        )
    
    def predict_batch(self, image_dir: str) -> list:
        """
        Predict on multiple images
        
        Args:
            image_dir: Directory containing images
            
        Returns:
            List of PredictionResult objects
        """
        image_dir = Path(image_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        results = []
        image_files = [f for f in image_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        for image_path in image_files:
            try:
                result = self.predict(str(image_path))
                result.image_path = str(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        return results
    
    def export_to_tflite(self, model_path: str, output_path: str):
        """
        Export model to TensorFlow Lite format
        
        Args:
            model_path: Path to h5 model
            output_path: Path to save tflite model
        """
        model = tf.keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Model exported to {output_path}")
