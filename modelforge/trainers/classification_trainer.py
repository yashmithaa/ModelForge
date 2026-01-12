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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from modelforge.trainers.base_trainer import BaseTrainer
from modelforge.core.config import ModelForgeConfig, OptimizerType, SchedulerType


class ClassificationTrainer(BaseTrainer):
    """
    Supports:
    - Binary and multi-class classification
    - Various optimizers and schedulers
    - Automatic metrics computation (accuracy, precision, recall, f1)
    - Mixed precision training
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ModelForgeConfig,
        num_classes: int,
        output_dir: str = None
    ):
        """Initialize classification trainer.
        
        Args:
            model: PyTorch model for classification
            config: ModelForge configuration
            num_classes: Number of classes
            output_dir: Directory to save outputs
        """
        super().__init__(model, config, output_dir)
        self.num_classes = num_classes
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # For collecting predictions and labels
        self.train_preds = []
        self.train_labels = []
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
            batch: Dictionary containing 'input_ids', 'attention_mask', 'labels', etc.
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Move batch to device
        inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(self.device)
        
        # Forward pass with mixed precision if enabled
        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(**inputs)
                logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
                loss = self.criterion(logits, labels)
        else:
            outputs = self.model(**inputs)
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
            loss = self.criterion(logits, labels)
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Compute metrics
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == labels).float().mean().item()
        
        metrics = {
            'accuracy': accuracy
        }
        
        return loss, metrics
    
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Perform a single validation step.
        
        Args:
            batch: Dictionary containing input data and labels
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Move batch to device
        inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        outputs = self.model(**inputs)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
        loss = self.criterion(logits, labels)
        
        # Compute metrics
        predictions = torch.argmax(logits, dim=-1)
        
        # Store for epoch-level metrics
        self.val_preds.extend(predictions.cpu().numpy())
        self.val_labels.extend(labels.cpu().numpy())
        
        accuracy = (predictions == labels).float().mean().item()
        
        metrics = {
            'accuracy': accuracy
        }
        
        return loss, metrics
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate the model with detailed metrics.
        
        Extends base validation to compute precision, recall, and F1.
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
            avg_metrics['accuracy'] = accuracy_score(labels_array, preds_array)
            
            # For multi-class, use weighted average
            average = 'binary' if self.num_classes == 2 else 'weighted'
            
            avg_metrics['precision'] = precision_score(
                labels_array, preds_array, average=average, zero_division=0
            )
            avg_metrics['recall'] = recall_score(
                labels_array, preds_array, average=average, zero_division=0
            )
            avg_metrics['f1'] = f1_score(
                labels_array, preds_array, average=average, zero_division=0
            )
        
        return avg_loss, avg_metrics
    
    def _predict_batch(self, batch: Dict[str, torch.Tensor]) -> List[int]:
        """Generate predictions for a batch.
        
        Args:
            batch: Batch of data
            
        Returns:
            List of predicted class indices
        """
        # Move batch to device
        inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
        
        # Forward pass
        outputs = self.model(**inputs)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        
        return predictions.cpu().tolist()
    
    def predict_proba(self, data_loader: DataLoader) -> np.ndarray:
        """Generate probability predictions.
        
        Args:
            data_loader: DataLoader for inference
            
        Returns:
            Array of shape (n_samples, n_classes) with class probabilities
        """
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for batch in data_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                
                outputs = self.model(**inputs)
                logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
                
                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())
        
        return np.vstack(all_probs)
