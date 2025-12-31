"""
Component registry system for ModelForge.

Provides decorator-based registration and auto-discovery of encoders, decoders, 
combiners, and other components.
"""

from typing import Dict, Type, Callable, Any
from functools import wraps


class Registry:
    """Registry for ModelForge components.
    
    Maintains mappings of component names to their implementation classes.
    """
    
    def __init__(self, name: str):
        """Initialize the registry.
        
        Args:
            name: Name of the registry (e.g., 'encoders', 'decoders')
        """
        self.name = name
        self._registry: Dict[str, Type] = {}
    
    def register(self, name: str) -> Callable:
        """Decorator to register a component.
        
        Args:
            name: Name to register the component under
            
        Returns:
            Decorator function
            
        Example:
            @encoder_registry.register("rnn")
            class RNNEncoder(BaseEncoder):
                pass
        """
        def decorator(cls: Type) -> Type:
            if name in self._registry:
                raise ValueError(
                    f"Component '{name}' already registered in {self.name} registry"
                )
            self._registry[name] = cls
            return cls
        return decorator
    
    def get(self, name: str) -> Type:
        """Get a registered component by name.
        
        Args:
            name: Name of the component
            
        Returns:
            Component class
            
        Raises:
            KeyError: If component not found
        """
        if name not in self._registry:
            raise KeyError(
                f"Component '{name}' not found in {self.name} registry. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name]
    
    def list(self) -> list:
        """List all registered component names.
        
        Returns:
            List of registered component names
        """
        return list(self._registry.keys())
    
    def __contains__(self, name: str) -> bool:
        """Check if a component is registered.
        
        Args:
            name: Component name
            
        Returns:
            True if registered, False otherwise
        """
        return name in self._registry


# Global registries
encoder_registry = Registry("encoders")
decoder_registry = Registry("decoders")
combiner_registry = Registry("combiners")
feature_registry = Registry("features")
metric_registry = Registry("metrics")


# Convenience decorator functions
def register_encoder(name: str) -> Callable:
    """Register an encoder.
    
    Args:
        name: Name for the encoder (e.g., 'rnn', 'transformer', 'bert')
        
    Returns:
        Decorator function
    """
    return encoder_registry.register(name)


def register_decoder(name: str) -> Callable:
    """Register a decoder.
    
    Args:
        name: Name for the decoder (e.g., 'classifier', 'regressor')
        
    Returns:
        Decorator function
    """
    return decoder_registry.register(name)


def register_combiner(name: str) -> Callable:
    """Register a combiner.
    
    Args:
        name: Name for the combiner (e.g., 'concat', 'sum')
        
    Returns:
        Decorator function
    """
    return combiner_registry.register(name)


def register_feature(name: str) -> Callable:
    """Register a feature type.
    
    Args:
        name: Name for the feature type (e.g., 'text', 'number', 'category')
        
    Returns:
        Decorator function
    """
    return feature_registry.register(name)


def register_metric(name: str) -> Callable:
    """Register a metric.
    
    Args:
        name: Name for the metric (e.g., 'accuracy', 'f1')
        
    Returns:
        Decorator function
    """
    return metric_registry.register(name)
