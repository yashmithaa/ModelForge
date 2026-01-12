from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, List
from pathlib import Path
import time
import json
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table

from modelforge.core.config import TrainingConfig, ModelForgeConfig
from modelforge.utils.logger import get_logger


class EarlyStopping:
    """Early stopping handler to stop training when metric stops improving."""
    
    def __init__(self, patience: int = 5, mode: str = 'min', min_delta: float = 0.0):
        """
        Args:
            patience: Number of epochs to wait before stopping
            mode: 'min' for loss or 'max' for accuracy 
            min_delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, metric: float) -> bool:
        """Check if should stop training.
        
        Args:
            metric: Current metric value
            
        Returns:
            True if should stop training, False otherwise
        """
        score = -metric if self.mode == 'min' else metric
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            
        return self.early_stop


class TrainingStats:
    """Container for tracking training statistics."""
    
    def __init__(self):
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_metrics: List[Dict[str, float]] = []
        self.val_metrics: List[Dict[str, float]] = []
        self.learning_rates: List[float] = []
        self.epoch_times: List[float] = []
        
    def add_epoch(
        self,
        train_loss: float,
        val_loss: Optional[float] = None,
        train_metrics: Optional[Dict[str, float]] = None,
        val_metrics: Optional[Dict[str, float]] = None,
        learning_rate: Optional[float] = None,
        epoch_time: Optional[float] = None
    ):
        """Add statistics for an epoch."""
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if train_metrics is not None:
            self.train_metrics.append(train_metrics)
        if val_metrics is not None:
            self.val_metrics.append(val_metrics)
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
        if epoch_time is not None:
            self.epoch_times.append(epoch_time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times
        }
    
    def save(self, path: Path):
        """Save statistics to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class BaseTrainer(ABC):
    """Abstract base class for all trainers.
    
    Handles the core training loop, validation, checkpointing, and logging.
    Subclasses need to implement model-specific training and validation steps.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ModelForgeConfig,
        output_dir: Optional[Path] = None
    ):
        """Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            config: ModelForge configuration
            output_dir: Directory to save outputs (checkpoints, logs, etc.)
        """
        self.model = model
        self.config = config
        self.training_config = config.training
        
        # Setup output directory
        self.output_dir = Path(output_dir) if output_dir else Path(config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = get_logger(__name__)
        self.console = Console()
        
        # Setup device
        self.device = self._setup_device()
        self.model.to(self.device)
        
        # Training components (to be initialized)
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[_LRScheduler] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.stats = TrainingStats()
        
        # Early stopping
        self.early_stopping = None
        if self.training_config.early_stopping:
            mode = 'min' if 'loss' in self.training_config.early_stopping_metric else 'max'
            self.early_stopping = EarlyStopping(
                patience=self.training_config.early_stopping_patience,
                mode=mode
            )
    
    def _setup_device(self) -> torch.device:
        """Setup compute device (CPU/GPU)."""
        if self.training_config.device:
            device = torch.device(self.training_config.device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.logger.info(f"Using device: {device}")
        if device.type == 'cuda':
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"CUDA version: {torch.version.cuda}")
        
        return device
    
    @abstractmethod
    def _setup_optimizer(self) -> Optimizer:
        """Setup optimizer for training.
        
        Returns:
            Configured optimizer
        """
        pass
    
    @abstractmethod
    def _setup_scheduler(self, optimizer: Optimizer) -> Optional[_LRScheduler]:
        """Setup learning rate scheduler.
        
        Args:
            optimizer: The optimizer to schedule
            
        Returns:
            Configured scheduler or None
        """
        pass
    
    @abstractmethod
    def train_step(self, batch: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Perform a single training step.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Tuple of (loss tensor, metrics dict)
        """
        pass
    
    @abstractmethod
    def validation_step(self, batch: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Perform a single validation step.
        
        Args:
            batch: Batch of validation data
            
        Returns:
            Tuple of (loss tensor, metrics dict)
        """
        pass
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Tuple of (average loss, metrics dict)
        """
        self.model.train()
        total_loss = 0.0
        total_metrics = {}
        num_batches = len(train_loader)
        
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task(
                f"[cyan]Training Epoch {self.current_epoch + 1}",
                total=num_batches
            )
            
            for batch_idx, batch in enumerate(train_loader):
                # Perform training step
                loss, metrics = self.train_step(batch)
                
                # Accumulate loss and metrics
                total_loss += loss.item()
                for key, value in metrics.items():
                    total_metrics[key] = total_metrics.get(key, 0.0) + value
                
                # Gradient accumulation
                if (batch_idx + 1) % self.training_config.gradient_accumulation_steps == 0:
                    self._optimizer_step()
                
                self.global_step += 1
                progress.update(task, advance=1)
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Tuple of (average loss, metrics dict)
        """
        self.model.eval()
        total_loss = 0.0
        total_metrics = {}
        num_batches = len(val_loader)
        
        with torch.no_grad():
            with Progress(
                SpinnerColumn(),
                *Progress.get_default_columns(),
                console=self.console
            ) as progress:
                task = progress.add_task(
                    "[yellow]Validating",
                    total=num_batches
                )
                
                for batch in val_loader:
                    loss, metrics = self.validation_step(batch)
                    
                    total_loss += loss.item()
                    for key, value in metrics.items():
                        total_metrics[key] = total_metrics.get(key, 0.0) + value
                    
                    progress.update(task, advance=1)
        
        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def _optimizer_step(self):
        """Perform optimizer step with gradient clipping if configured."""
        if self.training_config.gradient_clip_norm:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.gradient_clip_norm
            )
        
        if self.scaler:  # Mixed precision
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> TrainingStats:
        """Main training loop.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            
        Returns:
            Training statistics
        """
        # Setup training components
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler(self.optimizer)
        
        if self.training_config.mixed_precision and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.logger.info("Starting training...")
        self.logger.info(f"Total epochs: {self.training_config.epochs}")
        self.logger.info(f"Batch size: {self.training_config.batch_size}")
        self.logger.info(f"Learning rate: {self.training_config.learning_rate}")
        
        # Training loop
        for epoch in range(self.training_config.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validate if configured
            val_loss, val_metrics = None, None
            if val_loader and (epoch + 1) % self.training_config.validation_frequency == 0:
                val_loss, val_metrics = self.validate(val_loader)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loss else train_loss)
                else:
                    self.scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record statistics
            self.stats.add_epoch(
                train_loss=train_loss,
                val_loss=val_loss,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                learning_rate=current_lr,
                epoch_time=epoch_time
            )
            
            # Log epoch results
            self._log_epoch_results(
                epoch, train_loss, val_loss, train_metrics, val_metrics, current_lr, epoch_time
            )
            
            # Save checkpoint
            if self.training_config.save_checkpoints and \
               (epoch + 1) % self.training_config.checkpoint_frequency == 0:
                self.save_checkpoint(epoch, val_loss if val_loss else train_loss)
            
            # Check early stopping
            if self.early_stopping and val_loss is not None:
                if self.early_stopping(val_loss):
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
        
        # Save final checkpoint
        if self.training_config.save_checkpoints:
            self.save_checkpoint(self.current_epoch, val_loss if val_loss else train_loss, is_final=True)
        
        # Save training statistics
        self.stats.save(self.output_dir / "training_stats.json")
        
        self.logger.info("Training completed!")
        return self.stats
    
    def _log_epoch_results(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float],
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
        learning_rate: float,
        epoch_time: float
    ):
        """Log results for an epoch."""
        table = Table(title=f"Epoch {epoch + 1}/{self.training_config.epochs}")
        table.add_column("Metric", style="cyan")
        table.add_column("Train", style="green")
        if val_loss is not None:
            table.add_column("Validation", style="yellow")
        
        # Loss
        if val_loss is not None:
            table.add_row("Loss", f"{train_loss:.4f}", f"{val_loss:.4f}")
        else:
            table.add_row("Loss", f"{train_loss:.4f}", "-")
        
        # Other metrics
        all_metric_keys = set(train_metrics.keys())
        if val_metrics:
            all_metric_keys.update(val_metrics.keys())
        
        for key in sorted(all_metric_keys):
            train_val = f"{train_metrics.get(key, 0.0):.4f}"
            val_val = f"{val_metrics.get(key, 0.0):.4f}" if val_metrics else "-"
            if val_loss is not None:
                table.add_row(key.replace('_', ' ').title(), train_val, val_val)
            else:
                table.add_row(key.replace('_', ' ').title(), train_val, "-")
        
        # Learning rate and time
        table.add_row("Learning Rate", f"{learning_rate:.2e}", "-" if val_loss is not None else "")
        table.add_row("Epoch Time", f"{epoch_time:.2f}s", "-" if val_loss is not None else "")
        
        self.console.print(table)
    
    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_final: bool = False
    ):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            val_loss: Validation loss
            is_final: Whether this is the final checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config.model_dump() if hasattr(self.config, 'model_dump') else self.config.dict(),
            'stats': self.stats.to_dict()
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        if is_final:
            path = self.checkpoint_dir / "final_checkpoint.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model (val_loss={val_loss:.4f}) to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def predict(self, data_loader: DataLoader) -> List[Any]:
        """Generate predictions on a dataset.
        
        Args:
            data_loader: DataLoader for inference
            
        Returns:
            List of predictions
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch_preds = self._predict_batch(batch)
                predictions.extend(batch_preds)
        
        return predictions
    
    @abstractmethod
    def _predict_batch(self, batch: Any) -> List[Any]:
        """Generate predictions for a batch.
        
        Args:
            batch: Batch of data
            
        Returns:
            List of predictions for the batch
        """
        pass
