import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple
from pathlib import Path

class TwoStageDetectionModel:
    """
    Two-stage mango detection model:
    Stage 1: Mango vs Not Mango
    Stage 2: Mango Variety Classification (Kesar, Alphonso, Langda, Other)
    """
    
    MANGO_VARIETIES = ['alphonso', 'kesar', 'langda', 'other']
    
    def __init__(self, input_size: int = 224):
        """
        Initialize two-stage model
        Args:
            input_size: Input image size (224 for MobileNetV3)
        """
        self.input_size = input_size
        self.stage1_model = None
        self.stage2_model = None
        
    def build_stage1_model(self, name: str = 'fruit_detection') -> keras.Model:
        """
        Build Stage 1 model: Mango vs Not Mango (Binary Classification)
        Uses MobileNetV3Small pretrained backbone
        """
        # Load pretrained MobileNetV3Small
        backbone = keras.applications.MobileNetV3Small(
            input_shape=(self.input_size, self.input_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze backbone layers
        backbone.trainable = False
        
        # Build classification head
        inputs = keras.Input(shape=(self.input_size, self.input_size, 3))
        x = backbone(inputs)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Binary classification output
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name=name)
        return model
    
    def build_stage2_model(self, name: str = 'fruit_variety') -> keras.Model:
        """
        Build Stage 2 model: Mango Variety Classification
        (Kesar, Alphonso, Langda, Other)
        Uses MobileNetV3Small pretrained backbone
        """
        # Load pretrained MobileNetV3Small
        backbone = keras.applications.MobileNetV3Small(
            input_shape=(self.input_size, self.input_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze backbone layers for initial training
        backbone.trainable = False
        
        # Build classification head
        inputs = keras.Input(shape=(self.input_size, self.input_size, 3))
        x = backbone(inputs)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Multi-class classification output
        outputs = layers.Dense(len(self.MANGO_VARIETIES), activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name=name)
        return model
    
    def compile_models(self, learning_rate: float = 1e-4):
        """Compile both models"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Stage 1: Binary cross-entropy
        self.stage1_model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )
        
        # Stage 2: Categorical cross-entropy
        self.stage2_model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
        )
    
    def unfreeze_backbone(self, num_layers_to_unfreeze: int = 50):
        """Unfreeze last N layers for fine-tuning"""
        if self.stage1_model:
            backbone = self.stage1_model.layers[1]  # Backbone is second layer after input
            backbone.trainable = True
            for layer in backbone.layers[:-num_layers_to_unfreeze]:
                layer.trainable = False
        
        if self.stage2_model:
            backbone = self.stage2_model.layers[1]
            backbone.trainable = True
            for layer in backbone.layers[:-num_layers_to_unfreeze]:
                layer.trainable = False
    
    def save_models(self, model_dir: str):
        """Save both models"""
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True)
        
        if self.stage1_model:
            stage1_path = model_path / 'fruit_detection.keras'
            self.stage1_model.save(str(stage1_path))
            print(f"Stage 1 model saved to {stage1_path}")
        
        if self.stage2_model:
            stage2_path = model_path / 'fruit_variety.keras'
            self.stage2_model.save(str(stage2_path))
            print(f"Stage 2 model saved to {stage2_path}")
    
    def load_models(self, model_dir: str):
        """Load both models"""
        model_path = Path(model_dir)
        
        stage1_path = model_path / 'fruit_detection.keras'
        stage2_path = model_path / 'fruit_variety.keras'
        
        if stage1_path.exists():
            self.stage1_model = keras.models.load_model(str(stage1_path))
            print(f"Stage 1 model loaded from {stage1_path}")
        
        if stage2_path.exists():
            self.stage2_model = keras.models.load_model(str(stage2_path))
            print(f"Stage 2 model loaded from {stage2_path}")
    
    def get_model_summary(self):
        """Print model summaries"""
        if self.stage1_model:
            print("\n" + "="*50)
            print("STAGE 1 MODEL: Mango Detection")
            print("="*50)
            self.stage1_model.summary()
        
        if self.stage2_model:
            print("\n" + "="*50)
            print("STAGE 2 MODEL: Variety Classification")
            print("="*50)
            self.stage2_model.summary()
