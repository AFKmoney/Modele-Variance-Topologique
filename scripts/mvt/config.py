"""
Configuration globale du MVT.
===============================
Définit les hyperparamètres de l'espace topologique, du lagrangien,
de la plasticité et de l'encodeur.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class MVTConfig:
    """
    Configuration complète du Modèle à Variance Topologique.

    Attributes:
        ambient_dim: Dimensionnalité N de l'espace ambiant R^N.
                      Chaque concept est un point dans cet espace.
        intrinsic_dim: Dimension intrinsèque de la variété sémantique.
        dt: Pas de temps pour l'intégration RK4.
        num_rk4_steps: Nombre de pas d'intégration par génération.
        alpha_erosion: Taux d'érosion topologique (barrières aux erreurs).
        beta_sedimentation: Taux de sédimentation (renforcement des succès).
        syntopy_strength: Force de l'opérateur de syntopie (fusion ★).
        potential_stiffness: Raideur du potentiel de contrainte V.
        kinetic_coupling: Couplage de l'énergie cinétique sémantique T.
        vocabulary_size: Taille du vocabulaire de projection.
        embedding_dim: Dimension des embeddings de base pour l'encodage.
        curvature_threshold: Seuil de courbure pour détecter les singularités.
        divergence_threshold: Seuil de divergence pour arrêter la génération.
        seed: Graine aléatoire pour la reproductibilité.
    """

    # === Dimensions de l'espace ===
    ambient_dim: int = 128
    intrinsic_dim: int = 64

    # === Paramètres d'intégration ===
    dt: float = 0.01
    num_rk4_steps: int = 200

    # === Paramètres de plasticité topologique ===
    alpha_erosion: float = 0.05
    beta_sedimentation: float = 0.03
    learning_rate_geom: float = 0.001

    # === Paramètres du lagrangien ===
    syntopy_strength: float = 1.0
    potential_stiffness: float = 2.0
    kinetic_coupling: float = 1.0

    # === Paramètres de l'encodeur ===
    vocabulary_size: int = 10000
    embedding_dim: int = 128
    max_seq_length: int = 1024

    # === Seuils de stabilité ===
    curvature_threshold: float = 10.0
    divergence_threshold: float = 1e6

    # === Reproductibilité ===
    seed: Optional[int] = 42
