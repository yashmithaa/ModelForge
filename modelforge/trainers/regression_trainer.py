from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import (
    _LRScheduler,
    StepLR,
    CosineAnnealingLR,
    ExponentialLR,
    ReduceLROnPlateau
)
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from modelforge.trainers.base_trainer import BaseTrainer
from modelforge.core.config import ModelForgeConfig, OptimizerType, SchedulerType


class RegressionTrainer(BaseTrainer):
    """Trainer for regression tasks.
    
    Supports:
    - Single and multi-output regression
    - MSE, MAE, Huber loss
    - R², RMSE, MAE metrics
    - Mixed precision training
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ModelForgeConfig,
        loss_type: str = 'mse',
        output_dir: str = None
    ):
        """Initialize regression trainer.
        
        Args:
            model: PyTorch model for regression
            config: ModelForge configuration
            loss_type: Loss function type ('mse', 'mae', 'huber')
            output_dir: Directory to save outputs
        """
        super().__init__(model, config, output_dir)
        
        # Setup loss function
        self.loss_type = loss_type
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'mae':
            self.criterion = nn.L1Loss()
        elif loss_type == 'huber':
            self.criterion = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # For collecting predictions and labels
        self.val_preds = []
        self.val_labels = []
    
    def _setup_optimizer(self) -> Optimizer:
        """Setup optimizer based on configuration."""
        opt_config = self.training_config
        
        # Get optimizer class
        optimizer_map = {
            OptimizerType.ADAM: Adam,
            OptimizerType.ADAMW: AdamW,
            OptimizerType.SGD: SGD,
            OptimizerType.RMSPROP: RMSprop
        }
        
        optimizer_class = optimizer_map.get(opt_config.optimizer, Adam)
        
        # Build optimizer kwargs
        kwargs = {'lr': opt_config.learning_rate}
        
        if opt_config.optimizer in [OptimizerType.ADAM, OptimizerType.ADAMW]:
            kwargs['betas'] = (0.9, 0.999)
            kwargs['eps'] = 1e-8
            if opt_config.optimizer == OptimizerType.ADAMW:
                kwargs['weight_decay'] = 0.01
        elif opt_config.optimizer == OptimizerType.SGD:
            kwargs['momentum'] = 0.9
            kwargs['weight_decay'] = 0.0001
        
        optimizer = optimizer_class(self.model.parameters(), **kwargs)
        self.logger.info(f"Using optimizer: {opt_config.optimizer}")
        
        return optimizer
    
    def _setup_scheduler(self, optimizer: Optimizer) -> _LRScheduler:
        """Setup learning rate scheduler."""
        if not self.training_config.scheduler:
            return None
        
        sched_config = self.training_config.scheduler
        
        if sched_config.type == SchedulerType.NONE:
            return None
        
        # Create scheduler based on type
        if sched_config.type == SchedulerType.STEP:
            scheduler = StepLR(
                optimizer,
                step_size=sched_config.step_size or 10,
                gamma=sched_config.gamma
            )
        elif sched_config.type == SchedulerType.COSINE:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.training_config.epochs,
                eta_min=self.training_config.learning_rate * 0.01
            )
        elif sched_config.type == SchedulerType.EXPONENTIAL:
            scheduler = ExponentialLR(
                optimizer,
                gamma=sched_config.gamma
            )
        elif sched_config.type == SchedulerType.REDUCE_ON_PLATEAU:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=sched_config.gamma,
                patience=sched_config.patience or 5,
                verbose=True
            )
        else:
            self.logger.warning(f"Unknown scheduler type: {sched_config.type}")
            return None
        
        self.logger.info(f"Using scheduler: {sched_config.type}")
        return scheduler
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Perform a single training step.
        
        Args:
            batch: Dictionary containing input features and targets
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Move batch to device
        inputs = {k: v.to(self.device) for k, v in batch.items() if k not in ['labels', 'targets']}
        targets = batch.get('targets', batch.get('labels')).to(self.device)
        
        # Forward pass with mixed precision if enabled
        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(**inputs)
                predictions = outputs if isinstance(outputs, torch.Tensor) else outputs.predictions
                loss = self.criterion(predictions.squeeze(), targets.squeeze())
        else:
            outputs = self.model(**inputs)
            predictions = outputs if isinstance(outputs, torch.Tensor) else outputs.predictions
            loss = self.criterion(predictions.squeeze(), targets.squeeze())
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Compute MAE as additional metric
        with torch.no_grad():
            mae = torch.abs(predictions.squeeze() - targets.squeeze()).mean().item()
        
        metrics = {
            'mae': mae
        }
        
        return loss, metrics
    
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Perform a single validation step.
        
        Args:
            batch: Dictionary containing input data and targets
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Move batch to device
        inputs = {k: v.to(self.device) for k, v in batch.items() if k not in ['labels', 'targets']}
        targets = batch.get('targets', batch.get('labels')).to(self.device)
        
        # Forward pass
        outputs = self.model(**inputs)
        predictions = outputs if isinstance(outputs, torch.Tensor) else outputs.predictions
        loss = self.criterion(predictions.squeeze(), targets.squeeze())
        
        # Store for epoch-level metrics
        self.val_preds.extend(predictions.squeeze().cpu().numpy())
        self.val_labels.extend(targets.squeeze().cpu().numpy())
        
        # Compute MAE
        mae = torch.abs(predictions.squeeze() - targets.squeeze()).mean().item()
        
        metrics = {
            'mae': mae
        }
        
        return loss, metrics
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate the model with detailed metrics.
        
        Extends base validation to compute R², RMSE, and MAE.
        """
        # Reset prediction storage
        self.val_preds = []
        self.val_labels = []
        
        # Run base validation
        avg_loss, avg_metrics = super().validate(val_loader)
        
        # Compute detailed metrics if we have predictions
        if self.val_preds and self.val_labels:
            preds_array = np.array(self.val_preds)
            labels_array = np.array(self.val_labels)
            
            # Compute metrics
            avg_metrics['mse'] = mean_squared_error(labels_array, preds_array)
            avg_metrics['rmse'] = np.sqrt(avg_metrics['mse'])
            avg_metrics['mae'] = mean_absolute_error(labels_array, preds_array)
            avg_metrics['r2'] = r2_score(labels_array, preds_array)
        
        return avg_loss, avg_metrics
    
    def _predict_batch(self, batch: Dict[str, torch.Tensor]) -> List[float]:
        """Generate predictions for a batch.
        
        Args:
            batch: Batch of data
            
        Returns:
            List of predicted values
        """
        # Move batch to device
        inputs = {k: v.to(self.device) for k, v in batch.items() if k not in ['labels', 'targets']}
        
        # Forward pass
        outputs = self.model(**inputs)
        predictions = outputs if isinstance(outputs, torch.Tensor) else outputs.predictions
        
        return predictions.squeeze().cpu().tolist()
