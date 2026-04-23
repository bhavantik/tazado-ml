import sys
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import tensorflow as tf
from models.mobilenet import TwoStageDetectionModel
from data.loader import MangoDataLoader
from utils.logger import setup_logger

logger = setup_logger('training')

def train_models(config_path: str = None, epochs: int = 50, batch_size: int = 32):
    """
    Train both models
    
    Args:
        config_path: Path to config file (optional)
        epochs: Number of training epochs
        batch_size: Training batch size
    """
    
    logger.info("="*60)
    logger.info("STARTING MANGO DETECTION MODEL TRAINING")
    logger.info("="*60)
    
    # Data loading
    logger.info("\n📊 Loading data...")
    data_loader = MangoDataLoader('data/', input_size=224)
    splits = data_loader.create_train_val_test_split()
    
    train_dataset = data_loader.get_train_dataset(splits, batch_size=batch_size)
    val_dataset = data_loader.get_val_dataset(splits, batch_size=batch_size)
    
    # Model building
    logger.info("\n🔨 Building models...")
    model_builder = TwoStageDetectionModel(input_size=224)
    model_builder.stage1_model = model_builder.build_stage1_model()
    model_builder.stage2_model = model_builder.build_stage2_model()
    model_builder.compile_models(learning_rate=1e-4)
    
    logger.info("✅ Models compiled")
    
    # Callbacks
    checkpoint_dir = Path('models/checkpoints')
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    stage1_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(checkpoint_dir / 'stage1_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    stage2_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(checkpoint_dir / 'stage2_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1
    )
    
    # Training Stage 1
    logger.info("\n" + "="*60)
    logger.info("TRAINING STAGE 1: Mango Detection")
    logger.info("="*60)
    
    history1 = model_builder.stage1_model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[stage1_checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    
    # Training Stage 2
    logger.info("\n" + "="*60)
    logger.info("TRAINING STAGE 2: Variety Classification")
    logger.info("="*60)
    
    history2 = model_builder.stage2_model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[stage2_checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    
    # Save final models
    logger.info("\n💾 Saving models...")
    model_builder.save_models('models')
    
    # Save training history
    history_file = Path('models/training_history.json')
    with open(history_file, 'w') as f:
        json.dump({
            'stage1': {k: [float(v) for v in vals] for k, vals in history1.history.items()},
            'stage2': {k: [float(v) for v in vals] for k, vals in history2.history.items()}
        }, f, indent=2)
    
    logger.info(f"✅ Training complete!")
    logger.info(f"Models saved to: models/")
    logger.info(f"History saved to: {history_file}")
    
    return model_builder

if __name__ == '__main__':
    train_models(epochs=50, batch_size=32)
