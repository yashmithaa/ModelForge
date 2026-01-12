"""
Trainers for different task types.
"""

from modelforge.trainers.base_trainer import BaseTrainer, EarlyStopping, TrainingStats
from modelforge.trainers.classification_trainer import ClassificationTrainer
from modelforge.trainers.regression_trainer import RegressionTrainer

__all__ = [
    'BaseTrainer',
    'EarlyStopping',
    'TrainingStats',
    'ClassificationTrainer',
    'RegressionTrainer',
]
