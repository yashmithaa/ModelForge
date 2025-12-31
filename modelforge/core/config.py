"""
Configuration schema and validation for ModelForge.

Uses Pydantic for robust config validation with sensible defaults.
"""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum


class OptimizerType(str, Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"


class SchedulerType(str, Enum):
    STEP = "step"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    NONE = "none"


class CombinerType(str, Enum):
    CONCAT = "concat"
    SUM = "sum"
    SEQUENCE_CONCAT = "sequence_concat"


class FeatureConfig(BaseModel):
    """Configuration for a single feature."""
    name: str = Field(..., description="Name of the feature column")
    type: str = Field(..., description="Feature type (text, number, category, binary)")
    encoder: str = Field(..., description="Encoder type to use")
    preprocessing: Dict[str, Any] = Field(default_factory=dict, description="Preprocessing options")
    encoder_params: Dict[str, Any] = Field(default_factory=dict, description="Encoder-specific parameters")
    
    class Config:
        extra = "allow"  # Allow additional fields


class OutputFeatureConfig(FeatureConfig):
    decoder: str = Field(..., description="Decoder type to use")
    decoder_params: Dict[str, Any] = Field(default_factory=dict, description="Decoder-specific parameters")
    loss: Optional[str] = Field(None, description="Loss function (auto-detected if not specified)")
    metrics: List[str] = Field(default_factory=list, description="Metrics to compute")


class DatasetConfig(BaseModel):
    path: str = Field(..., description="Path to the dataset file")
    format: str = Field("csv", description="Dataset format (csv, json, parquet)")
    train_split: float = Field(0.7, ge=0.0, le=1.0, description="Training split ratio")
    validation_split: float = Field(0.15, ge=0.0, le=1.0, description="Validation split ratio")
    test_split: float = Field(0.15, ge=0.0, le=1.0, description="Test split ratio")
    cache: bool = Field(True, description="Whether to cache preprocessed data")
    cache_path: Optional[str] = Field(None, description="Path for cache files")
    
    @validator('train_split', 'validation_split', 'test_split')
    def validate_splits(cls, v, values):
        # Ensure splits sum to 1.0.
        return v
    
    @root_validator
    def validate_split_sum(cls, values):
        # Ensure all splits sum to 1.0.
        total = values.get('train_split', 0) + values.get('validation_split', 0) + values.get('test_split', 0)
        if not 0.99 <= total <= 1.01:  # Allow small floating point errors
            raise ValueError(f"Train, validation, and test splits must sum to 1.0, got {total}")
        return values


class SchedulerConfig(BaseModel):
    # Configuration for learning rate scheduler
    type: SchedulerType = Field(SchedulerType.NONE, description="Scheduler type")
    step_size: Optional[int] = Field(None, description="Step size for step scheduler")
    gamma: float = Field(0.1, description="Multiplicative factor for decay")
    patience: Optional[int] = Field(None, description="Patience for reduce_on_plateau")
    warmup_steps: int = Field(0, description="Number of warmup steps")


class TrainingConfig(BaseModel):
    """Configuration for training."""
    epochs: int = Field(10, ge=1, description="Number of training epochs")
    batch_size: int = Field(32, ge=1, description="Batch size for training")
    learning_rate: float = Field(0.001, gt=0.0, description="Learning rate")
    optimizer: OptimizerType = Field(OptimizerType.ADAM, description="Optimizer type")
    scheduler: Optional[SchedulerConfig] = Field(None, description="Learning rate scheduler config")
    early_stopping: bool = Field(True, description="Whether to use early stopping")
    early_stopping_patience: int = Field(5, ge=1, description="Patience for early stopping")
    early_stopping_metric: str = Field("loss", description="Metric to monitor for early stopping")
    gradient_clip_norm: Optional[float] = Field(None, gt=0.0, description="Gradient clipping norm")
    gradient_accumulation_steps: int = Field(1, ge=1, description="Gradient accumulation steps")
    mixed_precision: bool = Field(False, description="Use mixed precision training")
    save_checkpoints: bool = Field(True, description="Save model checkpoints")
    checkpoint_frequency: int = Field(1, ge=1, description="Save checkpoint every N epochs")
    validation_frequency: int = Field(1, ge=1, description="Validate every N epochs")
    device: Optional[str] = Field(None, description="Device to train on (auto-detect if None)")
    random_seed: int = Field(42, description="Random seed for reproducibility")


class CombinerConfig(BaseModel):
    """Configuration for the combiner."""
    type: CombinerType = Field(CombinerType.CONCAT, description="Combiner type")
    params: Dict[str, Any] = Field(default_factory=dict, description="Combiner-specific parameters")
    fc_layers: Optional[List[int]] = Field(None, description="Fully connected layers after combining")
    dropout: float = Field(0.0, ge=0.0, le=1.0, description="Dropout rate")


class ModelForgeConfig(BaseModel):
    
    # Dataset configuration
    dataset: DatasetConfig = Field(..., description="Dataset configuration")
    
    # Features
    input_features: List[FeatureConfig] = Field(..., min_items=1, description="Input feature configurations")
    output_features: List[OutputFeatureConfig] = Field(..., min_items=1, description="Output feature configurations")
    
    # Model architecture
    combiner: CombinerConfig = Field(default_factory=CombinerConfig, description="Combiner configuration")
    
    # Training
    training: TrainingConfig = Field(default_factory=TrainingConfig, description="Training configuration")
    
    # Preprocessing
    preprocessing: Dict[str, Any] = Field(default_factory=dict, description="Global preprocessing options")
    
    # Experiment tracking
    experiment_name: Optional[str] = Field(None, description="Name for this experiment")
    output_directory: str = Field("./results", description="Directory to save results")
    
    class Config:
        extra = "forbid"  # Don't allow unknown fields
        use_enum_values = True


def validate_config(config_dict: Dict[str, Any]) -> ModelForgeConfig:
    """Validate a configuration dictionary.
    
    Args:
        config_dict: Configuration as a dictionary
        
    Returns:
        Validated ModelForgeConfig object
        
    Raises:
        ValidationError: If configuration is invalid
    """
    return ModelForgeConfig(**config_dict)


def load_config_from_yaml(yaml_path: str) -> ModelForgeConfig:
    """Load and validate configuration from a YAML file.
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Returns:
        Validated ModelForgeConfig object
    """
    import yaml
    
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return validate_config(config_dict)
