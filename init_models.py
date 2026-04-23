import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.mobilenet import TwoStageDetectionModel
from utils.logger import setup_logger
import json

logger = setup_logger('model_init')

def initialize_pretrained_models():
    """Initialize and setup pretrained models"""
    
    logger.info("Starting model initialization...")
    
    # Create model instance
    model_builder = TwoStageDetectionModel(input_size=224)
    
    # Build Stage 1: Mango Detection
    logger.info("Building Stage 1 model (Mango Detection)...")
    model_builder.stage1_model = model_builder.build_stage1_model()
    logger.info("Stage 1 model built with MobileNetV3 backbone and imagenet weights")
    
    # Build Stage 2: Variety Classification
    logger.info("Building Stage 2 model (Variety Classification)...")
    model_builder.stage2_model = model_builder.build_stage2_model()
    logger.info("Stage 2 model built with MobileNetV3 backbone and imagenet weights")
    
    # Compile both models
    model_builder.compile_models(learning_rate=1e-4)
    
    # Print model summaries
    model_builder.get_model_summary()
    
    # Save models
    models_dir = Path(__file__).parent / 'models'
    logger.info(f"Saving models to {models_dir}...")
    model_builder.save_models(str(models_dir))
    
    # Create model registry
    registry = {
        'models': {
            'fruit_detection': {
                'name': 'Mango Detection (Binary Classification)',
                'path': str(models_dir / 'fruit_detection.keras'),
                'framework': 'TensorFlow',
                'backbone': 'MobileNetV3Small',
                'weights': 'imagenet',
                'input_size': 224,
                'output_classes': ['not_mango', 'mango'],
                'description': 'Detects if image contains mango or not'
            },
            'fruit_variety': {
                'name': 'Mango Variety Classification',
                'path': str(models_dir / 'fruit_variety.keras'),
                'framework': 'TensorFlow',
                'backbone': 'MobileNetV3Small',
                'weights': 'imagenet',
                'input_size': 224,
                'output_classes': ['alphonso', 'kesar', 'langda', 'other'],
                'description': 'Classifies variety of mango (Alphonso, Kesar, Langda, Other)'
            }
        },
        'pipeline': {
            'inference_flow': [
                {'stage': 1, 'model': 'fruit_detection', 'threshold': 0.5},
                {'stage': 2, 'model': 'fruit_variety', 'condition': 'if_mango'}
            ],
            'not_mango_message': 'Model not trained for this object. Please upload mango image.'
        }
    }
    
    # Save registry
    registry_path = models_dir / 'registry.json'
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    logger.info(f"Model registry saved to {registry_path}")
    logger.info("✅ Model initialization complete!")
    logger.info(f"\nModels ready for training with data from: data/raw/mango/")
    logger.info(f"Available varieties: {', '.join(registry['models']['fruit_variety']['output_classes'])}")
    
    return registry

if __name__ == '__main__':
    initialize_pretrained_models()
