"""
Core abstractions for ModelForge.

This module contains the foundational classes and utilities:
- Config schema and validation
- Component registry system
- Base classes for all components
"""

from modelforge.core.base import BaseEncoder, BaseDecoder, BaseCombiner, BaseFeature
from modelforge.core.registry import Registry, register_encoder, register_decoder, register_combiner
from modelforge.core.config import ModelForgeConfig, validate_config

__all__ = [
    "BaseEncoder",
    "BaseDecoder", 
    "BaseCombiner",
    "BaseFeature",
    "Registry",
    "register_encoder",
    "register_decoder",
    "register_combiner",
    "ModelForgeConfig",
    "validate_config",
]
