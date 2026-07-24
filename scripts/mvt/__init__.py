"""
MVT - Modèle à Variance Topologique.
"""
from .config import MVTConfig
from .model import MVT
from .agent import MVSAgent
from .creativity import CreativityEngine
from .kuramoto_metric import KuramotoMetricCoupler
from .natural_gradient_spd import NaturalGradientSPD

__version__ = "3.0.0"
__all__ = ["MVTConfig", "MVT", "MVSAgent", "CreativityEngine",
          "KuramotoMetricCoupler", "NaturalGradientSPD"]
