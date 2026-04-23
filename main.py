#!/usr/bin/env python
"""
Tazado ML Mango Detection System - CLI Interface
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.logger import setup_logger
from models.mobilenet import TwoStageDetectionModel
from inference.predict import MangoPredictor
from data.loader import MangoDataLoader

logger = setup_logger('tazado_cli')

def cmd_init_models(args):
    """Initialize pretrained models"""
    logger.info("🔧 Initializing pretrained models...")
    
    from init_models import initialize_pretrained_models
    registry = initialize_pretrained_models()
    
    logger.info("\n📋 Model Registry:")
    for model_name, model_info in registry['models'].items():
        logger.info(f"\n  {model_name}:")
        logger.info(f"    Name: {model_info['name']}")
        logger.info(f"    Classes: {', '.join(model_info['output_classes'])}")
        logger.info(f"    Path: {model_info['path']}")

def cmd_train(args):
    """Train models"""
    logger.info("🎓 Starting model training...")
    
    from src.training.train import train_models
    train_models(epochs=args.epochs, batch_size=args.batch_size)

def cmd_predict(args):
    """Run inference on image"""
    logger.info(f"🔍 Predicting on: {args.image}")
    
    models_dir = Path(__file__).parent / 'models'
    stage1_path = models_dir / 'fruit_detection.keras'
    stage2_path = models_dir / 'fruit_variety.keras'
    
    if not stage1_path.exists() or not stage2_path.exists():
        logger.error("❌ Models not found. Run 'python main.py init_models' first.")
        return
    
    predictor = MangoPredictor(str(stage1_path), str(stage2_path))
    result = predictor.predict(args.image)
    
    logger.info("\n" + "="*60)
    logger.info("PREDICTION RESULT")
    logger.info("="*60)
    logger.info(result)
    logger.info(f"\nDetailed Results:")
    logger.info(f"  Image: {args.image}")
    logger.info(f"  Is Mango: {result.is_mango}")
    logger.info(f"  Mango Confidence: {result.mango_confidence:.4f}")
    if result.is_mango:
        logger.info(f"  Variety: {result.variety}")
        logger.info(f"  Variety Confidence: {result.variety_confidence:.4f}")
    else:
        logger.info(f"  Message: {result.message}")

def cmd_predict_batch(args):
    """Run inference on multiple images"""
    logger.info(f"🔍 Predicting on images in: {args.dir}")
    
    models_dir = Path(__file__).parent / 'models'
    stage1_path = models_dir / 'fruit_detection.keras'
    stage2_path = models_dir / 'fruit_variety.keras'
    
    if not stage1_path.exists() or not stage2_path.exists():
        logger.error("❌ Models not found. Run 'python main.py init_models' first.")
        return
    
    predictor = MangoPredictor(str(stage1_path), str(stage2_path))
    results = predictor.predict_batch(args.dir)
    
    logger.info(f"\n✅ Processed {len(results)} images")
    logger.info("\n" + "="*60)
    logger.info("BATCH PREDICTION RESULTS")
    logger.info("="*60)
    
    for i, result in enumerate(results, 1):
        logger.info(f"\n{i}. {result.image_path}")
        logger.info(f"   {result}")

def cmd_test_models(args):
    """Test loaded models"""
    logger.info("🧪 Testing pretrained models...")
    
    from test_models import test_pretrained_models
    test_pretrained_models()

def main():
    parser = argparse.ArgumentParser(
        description='Tazado ML - Mango Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize pretrained models
  python main.py init_models
  
  # Train models on data
  python main.py train --epochs 50 --batch_size 32
  
  # Make prediction on single image
  python main.py predict --image path/to/image.jpg
  
  # Make predictions on batch
  python main.py predict_batch --dir path/to/images/
  
  # Test models
  python main.py test_models
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # init_models
    init_parser = subparsers.add_parser('init_models', help='Initialize pretrained models')
    init_parser.set_defaults(func=cmd_init_models)
    
    # train
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    train_parser.set_defaults(func=cmd_train)
    
    # predict
    predict_parser = subparsers.add_parser('predict', help='Predict on single image')
    predict_parser.add_argument('--image', type=str, required=True, help='Path to image')
    predict_parser.set_defaults(func=cmd_predict)
    
    # predict_batch
    batch_parser = subparsers.add_parser('predict_batch', help='Predict on multiple images')
    batch_parser.add_argument('--dir', type=str, required=True, help='Directory with images')
    batch_parser.set_defaults(func=cmd_predict_batch)
    
    # test_models
    test_parser = subparsers.add_parser('test_models', help='Test loaded models')
    test_parser.set_defaults(func=cmd_test_models)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)

if __name__ == '__main__':
    main()
