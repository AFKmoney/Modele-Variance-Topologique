"""
MVT - Modèle à Variance Topologique (Topological Variance Model)
=================================================================

Une architecture d'IA fondée sur la mécanique géométrique et la
géométrie différentielle, remplaçant les transformers par un champ
vectoriel continu régi par un lagrangien sémantique.

Auteurs: Architecture conceptuelle originale
Licence: MIT
"""

__version__ = "1.0.0"
__author__ = "MVT Project"

from .config import MVTConfig
from .model import MVT

__all__ = ["MVTConfig", "MVT"]
