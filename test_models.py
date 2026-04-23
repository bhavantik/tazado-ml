import sys
from pathlib import Path
import numpy as np
from tensorflow import keras

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from inference.predict import MangoPredictor

def test_pretrained_models():
    """Test pretrained models with sample images"""
    
    models_dir = Path(__file__).parent / 'models'
    
    # Check if models exist
    stage1_path = models_dir / 'fruit_detection.keras'
    stage2_path = models_dir / 'fruit_variety.keras'
    
    if not stage1_path.exists() or not stage2_path.exists():
        print("❌ Models not found. Run `python init_models.py` first.")
        return
    
    print("✅ Models found!")
    print(f"\nStage 1 (Detection): {stage1_path}")
    print(f"Stage 2 (Variety): {stage2_path}")
    
    # Load models
    print("\nLoading models...")
    predictor = MangoPredictor(str(stage1_path), str(stage2_path))
    print("✅ Models loaded successfully!")
    
    # Load and print model info
    print("\n" + "="*60)
    print("STAGE 1 MODEL: Mango vs Not Mango Detection")
    print("="*60)
    predictor.stage1_model.summary()
    
    print("\n" + "="*60)
    print("STAGE 2 MODEL: Mango Variety Classification")
    print("="*60)
    predictor.stage2_model.summary()
    
    print("\n" + "="*60)
    print("MODEL CONFIGURATION")
    print("="*60)
    print(f"Input Size: 224x224x3")
    print(f"Backbone: MobileNetV3Small (pretrained on ImageNet)")
    print(f"Stage 1 Output: Binary (Mango / Not Mango)")
    print(f"Stage 2 Output: 4-class (Alphonso, Kesar, Langda, Other)")
    print(f"Mango Detection Threshold: 0.5")
    
    print("\n" + "="*60)
    print("READY FOR TRAINING!")
    print("="*60)
    print("\nNext steps:")
    print("1. Organize training data in data/raw/mango/")
    print("2. Run: python src/training/train.py")
    print("3. For inference: python src/inference/predict.py --image_path <path>")

if __name__ == '__main__':
    test_pretrained_models()
