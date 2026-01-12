"""
Example: Using the BaseTrainer with a simple classification model.

This example demonstrates how to use the ClassificationTrainer
with a simple sentiment analysis task.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import yaml

from modelforge.trainers import ClassificationTrainer
from modelforge.core.config import ModelForgeConfig


# Example: Simple text classification model
class SimpleClassifier(nn.Module):
    """Simple LSTM-based text classifier."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Mask for padding (optional)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        embedded = self.embedding(input_ids)  # (batch, seq, embed)
        lstm_out, (hidden, _) = self.lstm(embedded)  # hidden: (2, batch, hidden)
        
        # Concatenate forward and backward hidden states
        hidden = torch.cat([hidden[0], hidden[1]], dim=1)  # (batch, hidden*2)
        
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)  # (batch, num_classes)
        
        return logits


# Example dataset
class SimpleTextDataset(Dataset):
    """Simple dataset for demonstration."""
    
    def __init__(self, texts, labels, vocab_size=10000, max_length=128):
        self.texts = texts
        self.labels = labels
        self.vocab_size = vocab_size
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Simulate tokenization (in reality, you'd use a proper tokenizer)
        text = self.texts[idx]
        # Simple hash-based "tokenization"
        tokens = [hash(word) % self.vocab_size for word in text.split()][:self.max_length]
        
        # Pad to max_length
        input_ids = tokens + [0] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids[:self.max_length], dtype=torch.long)
        
        # Attention mask
        attention_mask = torch.tensor(
            [1] * len(tokens) + [0] * (self.max_length - len(tokens)),
            dtype=torch.long
        )[:self.max_length]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def create_sample_config() -> ModelForgeConfig:
    """Create a sample configuration for demonstration."""
    config_dict = {
        'dataset': {
            'path': 'dummy.csv',
            'format': 'csv',
            'train_split': 0.7,
            'validation_split': 0.15,
            'test_split': 0.15,
        },
        'input_features': [
            {
                'name': 'text',
                'type': 'text',
                'encoder': 'rnn',
                'encoder_params': {
                    'vocab_size': 10000,
                    'embedding_size': 128,
                    'state_size': 256,
                }
            }
        ],
        'output_features': [
            {
                'name': 'sentiment',
                'type': 'category',
                'encoder': 'passthrough',
                'decoder': 'classifier',
                'decoder_params': {
                    'num_classes': 3
                }
            }
        ],
        'training': {
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'early_stopping': True,
            'early_stopping_patience': 3,
            'early_stopping_metric': 'loss',
            'gradient_clip_norm': 1.0,
            'save_checkpoints': True,
            'checkpoint_frequency': 2,
            'validation_frequency': 1,
            'random_seed': 42,
            'scheduler': {
                'type': 'step',
                'step_size': 5,
                'gamma': 0.1
            }
        },
        'experiment_name': 'sentiment_analysis_example',
        'output_directory': './results/example'
    }
    
    return ModelForgeConfig(**config_dict)


def main():
    """Main example function."""
    print("=" * 80)
    print("ModelForge Trainer Example: Sentiment Analysis")
    print("=" * 80)
    
    # Create sample data
    train_texts = [
        "I love this product, it's amazing!",
        "Terrible experience, waste of money.",
        "It's okay, nothing special.",
        "Best purchase ever, highly recommend!",
        "Disappointed with the quality.",
        "Average product for the price.",
    ] * 20  # Repeat for more samples
    
    train_labels = [2, 0, 1, 2, 0, 1] * 20  # 0=negative, 1=neutral, 2=positive
    
    val_texts = [
        "Great quality and fast shipping!",
        "Not worth the money.",
        "It's decent.",
    ] * 10
    
    val_labels = [2, 0, 1] * 10
    
    # Create datasets
    train_dataset = SimpleTextDataset(train_texts, train_labels)
    val_dataset = SimpleTextDataset(val_texts, val_labels)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = SimpleClassifier(
        vocab_size=10000,
        embedding_dim=128,
        hidden_dim=256,
        num_classes=3
    )
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create configuration
    config = create_sample_config()
    
    # Create trainer
    trainer = ClassificationTrainer(
        model=model,
        config=config,
        num_classes=3,
        output_dir='./results/example'
    )
    
    print(f"\nTrainer: {trainer.__class__.__name__}")
    print(f"Device: {trainer.device}")
    print(f"Output directory: {trainer.output_dir}")
    
    # Train the model
    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80 + "\n")
    
    stats = trainer.train(train_loader, val_loader)
    
    # Print final results
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    
    print(f"\nFinal Training Loss: {stats.train_losses[-1]:.4f}")
    if stats.val_losses:
        print(f"Final Validation Loss: {stats.val_losses[-1]:.4f}")
    
    if stats.val_metrics:
        final_metrics = stats.val_metrics[-1]
        print("\nFinal Validation Metrics:")
        for metric_name, value in final_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    print(f"\nCheckpoints saved to: {trainer.checkpoint_dir}")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    
    # Example: Making predictions
    print("\n" + "=" * 80)
    print("Example Predictions")
    print("=" * 80 + "\n")
    
    test_texts = [
        "This is absolutely wonderful!",
        "Very disappointing product.",
        "It's just okay, nothing more.",
    ]
    test_labels = [2, 0, 1]
    test_dataset = SimpleTextDataset(test_texts, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=3)
    
    predictions = trainer.predict(test_loader)
    probs = trainer.predict_proba(test_loader)
    
    label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    for i, (text, pred, prob) in enumerate(zip(test_texts, predictions, probs)):
        print(f"Text: {text}")
        print(f"Predicted: {label_map[pred]}")
        print(f"Probabilities: {prob}")
        print(f"True label: {label_map[test_labels[i]]}")
        print()


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    
    main()
