import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str
    backbone: str
    input_size: int
    num_classes: int
    dropout_rate: float
    
@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int
    epochs: int
    learning_rate: float
    optimizer: str
    loss_fn: str
    validation_split: float
    
@dataclass
class DataConfig:
    """Data configuration"""
    data_dir: str
    raw_dir: str
    processed_dir: str
    splits_dir: str
    train_split: float
    val_split: float
    test_split: float
    
@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    
    @staticmethod
    def load(config_path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        return Config(
            model=ModelConfig(**cfg['model']),
            training=TrainingConfig(**cfg['training']),
            data=DataConfig(**cfg['data'])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__
        }
