"""
MVT - Modèle à Variance Topologique.
"""
from .config import MVTConfig
from .model import MVT
from .agent import MVSAgent
from .creativity import CreativityEngine

__version__ = "2.0.0"
__all__ = ["MVTConfig", "MVT", "MVSAgent", "CreativityEngine"]
