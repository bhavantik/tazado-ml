import sys
from pathlib import Path
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.logger import setup_logger
from inference.predict import MangoPredictor

logger = setup_logger('evaluation')

def evaluate_models(test_image_dir: str = 'data/processed', threshold: float = 0.5):
    """
    Evaluate models on test images
    
    Args:
        test_image_dir: Directory with test images
        threshold: Mango detection threshold
    """
    
    logger.info("="*60)
    logger.info("MODEL EVALUATION")
    logger.info("="*60)
    
    models_dir = Path(__file__).parent / 'models'
    stage1_path = models_dir / 'fruit_detection.h5'
    stage2_path = models_dir / 'fruit_variety.h5'
    
    if not stage1_path.exists() or not stage2_path.exists():
        logger.error("❌ Models not found. Run 'python init_models.py' first.")
        return
    
    logger.info(f"\n🔍 Evaluating models...")
    predictor = MangoPredictor(str(stage1_path), str(stage2_path))
    
    # Get test images
    test_dir = Path(test_image_dir)
    if not test_dir.exists():
        logger.warning(f"Test directory {test_image_dir} not found. Creating demo evaluation...")
        return
    
    # Batch predict
    results = predictor.predict_batch(str(test_dir))
    
    logger.info(f"\n✅ Processed {len(results)} test images")
    
    # Statistics
    mango_count = sum(1 for r in results if r.is_mango)
    not_mango_count = len(results) - mango_count
    
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    logger.info(f"\nDetection Statistics:")
    logger.info(f"  Total Images: {len(results)}")
    logger.info(f"  Mango Detected: {mango_count} ({100*mango_count/len(results):.1f}%)")
    logger.info(f"  Not Mango: {not_mango_count} ({100*not_mango_count/len(results):.1f}%)")
    
    if mango_count > 0:
        variety_counts = {}
        for r in results:
            if r.is_mango:
                variety_counts[r.variety] = variety_counts.get(r.variety, 0) + 1
        
        logger.info(f"\nVariety Distribution:")
        for variety, count in sorted(variety_counts.items()):
            logger.info(f"  {variety}: {count} ({100*count/mango_count:.1f}%)")
    
    logger.info(f"\nAverage Confidence Scores:")
    avg_mango_conf = np.mean([r.mango_confidence for r in results])
    logger.info(f"  Mango Detection: {avg_mango_conf:.4f}")
    
    if mango_count > 0:
        avg_variety_conf = np.mean([r.variety_confidence for r in results if r.is_mango])
        logger.info(f"  Variety Classification: {avg_variety_conf:.4f}")
    
    return results

if __name__ == '__main__':
    evaluate_models()
