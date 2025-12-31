from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """Base class for all encoders.
    
    Encoders transform input features into hidden representations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the encoder.
        
        Args:
            config: Configuration dictionary for the encoder
        """
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the encoder.
        
        Args:
            inputs: Input tensor
            mask: Optional mask tensor
            
        Returns:
            Tuple of (hidden_state, output)
        """
        pass
    
    @property
    @abstractmethod
    def output_shape(self) -> Tuple[int, ...]:
        """Returns the shape of the encoder output."""
        pass


class BaseDecoder(nn.Module, ABC):
    """Base class for all decoders.
    
    Decoders transform combined hidden representations into predictions.
    """
    
    def __init__(self, input_size: int, config: Dict[str, Any]):
        """Initialize the decoder.
        
        Args:
            input_size: Size of the input from combiner
            config: Configuration dictionary for the decoder
        """
        super().__init__()
        self.input_size = input_size
        self.config = config
    
    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder.
        
        Args:
            inputs: Input tensor from combiner
            
        Returns:
            Output predictions
        """
        pass
    
    @abstractmethod
    def get_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss for this decoder.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Loss tensor
        """
        pass


class BaseCombiner(nn.Module, ABC):
    """Base class for all combiners.
    
    Combiners merge multiple encoder outputs into a single representation.
    """
    
    def __init__(self, input_sizes: Dict[str, int], config: Dict[str, Any]):
        """Initialize the combiner.
        
        Args:
            input_sizes: Dictionary mapping feature names to their output sizes
            config: Configuration dictionary for the combiner
        """
        super().__init__()
        self.input_sizes = input_sizes
        self.config = config
    
    @abstractmethod
    def forward(self, encoder_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the combiner.
        
        Args:
            encoder_outputs: Dictionary mapping feature names to encoder outputs
            
        Returns:
            Combined representation tensor
        """
        pass
    
    @property
    @abstractmethod
    def output_size(self) -> int:
        """Returns the size of the combiner output."""
        pass


class BaseFeature(ABC):
    """Base class for feature type definitions.
    
    Features define how to preprocess, encode, and decode specific data types.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize the feature.
        
        Args:
            name: Name of the feature column
            config: Configuration dictionary for the feature
        """
        self.name = name
        self.config = config
    
    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """Preprocess raw data for this feature.
        
        Args:
            data: Raw input data
            
        Returns:
            Preprocessed data
        """
        pass
    
    @abstractmethod
    def get_encoder(self) -> BaseEncoder:
        """Get the encoder for this feature type.
        
        Returns:
            Encoder instance
        """
        pass
    
    @abstractmethod
    def get_decoder(self, input_size: int) -> Optional[BaseDecoder]:
        """Get the decoder for this feature type (if output feature).
        
        Args:
            input_size: Size of the input from combiner
            
        Returns:
            Decoder instance or None if input-only feature
        """
        pass
