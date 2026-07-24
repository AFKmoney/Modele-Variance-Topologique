"""
Configuration globale du MVT.
===============================
Définit les hyperparamètres de l'espace topologique, du lagrangien,
de la plasticité, de l'encodeur, et NOUVEAU : de l'agentivité et de la créativité.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class MVTConfig:
    """
    Configuration complète du Modèle à Variance Topologique.
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
    damping: float = 0.05  # Amortissement (baissé = plus fluide/creatif)

    # === Paramètres de l'encodeur ===
    vocabulary_size: int = 10000
    embedding_dim: int = 128
    max_seq_length: int = 1024

    # === Seuils de stabilité (RELÂCHÉS pour créativité) ===
    curvature_threshold: float = 100.0       # Très élevé = peu de blocages
    divergence_threshold: float = 1e8         # Très élevé = exploration libre

    # === CRÉATIVITÉ ===
    temperature: float = 0.8       # Température stochastique (0=déterministe, 1=très créatif)
    novelty_bias: float = 0.4     # Biais vers la nouveauté (0=safe, 1=explorateur)
    metaphor_jump_prob: float = 0.1   # Probabilité de saut métaphorique
    metaphor_jump_strength: float = 0.3  # Amplitude du saut dans l'espace
    creativity_noise_scale: float = 0.15  # Bruit injecté dans le lagrangien

    # === AGENTIC ===
    max_agent_iterations: int = 5   # Itérations max de la boucle d'agent
    self_reflection_threshold: float = 0.5  # Seuil de satisfaction de l'agent
    goal_steering_strength: float = 0.3  # Force d'attraction vers le but
    branching_factor: int = 3  # Nombre de trajectoires parallèles à explorer
    replan_probability: float = 0.3  # Probabilité de replanification

    # === Projection créative ===
    projection_temperature: float = 0.7  # Température de sélection de mots (soft-max like)
    diversity_penalty: float = 0.3  # Pénalité pour les répétitions
    max_consecutive_repeat: int = 2  # Max répétitions consécutives autorisées
    sample_rate: int = 3  # Échantillonner tous les N pas

    # === KURAMOTO ↔ MÉTRIQUE COUPLÉE ===
    kuramoto_enabled: bool = False  # Activer la boucle Kuramoto-G (mémoire morphologique)
    kuramoto_coupling_K: float = 1.0  # Force de couplage Kuramoto
    kuramoto_n_oscillators: Optional[int] = None  # Nombre d'oscillateurs (None = ambient_dim)
    kuramoto_metric_lr: float = 0.001  # Taux d'apprentissage NatGrad SPD
    kuramoto_retraction: str = "approx2"  # Méthode de retraction: 'exp', 'approx2', 'cholesky'
    kuramoto_phase_coupling: float = 0.1  # ε dans G_ij *= (1 + ε·cos(φ_i - φ_j))
    kuramoto_phase_init: str = "random"  # Initialisation des phases: 'random', 'sync', 'cluster'

    # === Reproductibilité ===
    seed: Optional[int] = 42
