"""
MVT Plasticity - Moteur d'Autopoïèse Topologique.
====================================================

Implémente l'évolution dynamique du tenseur métrique G(t) :
    dG_ij/dt = -alpha * Courbure(G_ij) + beta * Flux_ij

L'IA s'auto-corrige et développe de nouvelles capacités cognitives
en temps réel, simplement en modifiant la géométrie de son propre
espace de réflexion.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, List

from .config import MVTConfig
from .core.metric_tensor import MetricTensor


class TopologicalPlasticityEngine:
    """
    Moteur de plasticité topologique : érosion et sédimentation.

    Quand l'IA génère une trajectoire qui mène à une impasse logique,
    l'espace s'érige une "barrière" topologique (érosion).
    Quand l'IA génère une trajectoire réussie, le chemin s'approfondit
    (sédimentation), rendant la génération de ce type d'idée plus fluide.

    Pas de rétropropagation. Pas de gradient. L'évolution est purement
    géométrique.
    """

    def __init__(self, config: MVTConfig, metric: MetricTensor):
        self.config = config
        self.metric = metric
        self.N = config.ambient_dim

        # Taux de plasticité
        self.alpha = config.alpha_erosion
        self.beta = config.beta_sedimentation
        self.learning_rate = config.learning_rate_geom

        # Historique des trajectoires pour le suivi
        self._trajectory_history: List[np.ndarray] = []
        self._success_flags: List[bool] = []

        # Compteurs de plasticité
        self._total_erosions = 0
        self._total_sedimentations = 0
        self._barriers: List[np.ndarray] = []  # Positions des barrières créées
        self._channels: List[np.ndarray] = []   # Positions des canaux approfondis

    def compute_curvature_deformation(
        self, trajectory: np.ndarray
    ) -> np.ndarray:
        """
        Calcule la déformation due à la courbure le long d'une trajectoire.

        Si la trajectoire passe par des zones de forte courbure positive
        (singularités/impasses), cela génère une force d'érosion.

        Args:
            trajectory: Trajectoire (T, N)

        Returns:
            Tenseur de déformation (N, N)
        """
        T_len = len(trajectory)
        deformation = np.zeros((self.N, self.N), dtype=np.float64)

        for t_idx in range(T_len):
            q = trajectory[t_idx]
            curvature = self.metric.scalar_curvature(q)

            if curvature > self.config.curvature_threshold * 0.5:
                # Zone de forte courbure positive → barrière
                # La déformation éloigne les futures trajectoires
                q_normalized = q / (np.linalg.norm(q) + 1e-8)
                barrier = np.outer(q_normalized, q_normalized)
                deformation -= self.alpha * curvature * barrier

        return deformation

    def compute_flux_deformation(
        self, trajectory: np.ndarray, success: bool = True
    ) -> np.ndarray:
        """
        Calcule la déformation par sédimentation le long d'une trajectoire.

        Si la trajectoire est un succès, les chemins sont approfondis
        (la métrique se renforce dans ces directions).

        Args:
            trajectory: Trajectoire (T, N)
            success: La trajectoire est-elle réussie ?

        Returns:
            Tenseur de déformation (N, N)
        """
        if not success:
            return np.zeros((self.N, self.N), dtype=np.float64)

        T_len = len(trajectory)
        flux = np.zeros((self.N, self.N), dtype=np.float64)

        # Calculer les directions principales du flux sémantique
        if T_len < 2:
            return flux

        for t_idx in range(T_len - 1):
            dq = trajectory[t_idx + 1] - trajectory[t_idx]
            dq_norm = dq / (np.linalg.norm(dq) + 1e-8)

            # Sédimentation : renforcer la métrique dans la direction du flux
            flux += self.beta * np.outer(dq_norm, dq_norm)

        # Normaliser par la longueur de la trajectoire
        flux /= T_len

        return flux

    def update_metric(
        self,
        trajectory: np.ndarray,
        success: bool = True,
    ):
        """
        Met à jour le tenseur métrique G(t) après une génération.

        dG_ij/dt = -alpha * Courbure(G_ij) + beta * Flux_ij

        Args:
            trajectory: Trajectoire générée (T, N)
            success: La génération est-elle jugée réussie ?
        """
        # Déformation de courbure (érosion)
        curvature_def = self.compute_curvature_deformation(trajectory)

        # Déformation de flux (sédimentation)
        flux_def = self.compute_flux_deformation(trajectory, success)

        # Mise à jour totale
        dG = self.learning_rate * (curvature_def + flux_def)

        # Appliquer la mise à jour
        self.metric.update(dG)

        # Stocker l'historique
        self._trajectory_history.append(trajectory)
        self._success_flags.append(success)

        if not success:
            self._total_erosions += 1
            # Enregistrer les barrières créées
            barrier_pos = trajectory[len(trajectory) // 2]  # Point médian
            self._barriers.append(barrier_pos.copy())
        else:
            self._total_sedimentations += 1
            channel_center = np.mean(trajectory, axis=0)
            self._channels.append(channel_center.copy())

        # Limiter l'historique
        max_history = 100
        if len(self._trajectory_history) > max_history:
            self._trajectory_history = self._trajectory_history[-max_history:]
            self._success_flags = self._success_flags[-max_history:]

    def detect_singularity(self, trajectory: np.ndarray) -> Tuple[bool, int]:
        """
        Détecte si une trajectoire passe par une singularité.

        Une singularité est une zone de courbure infinie ou divergente
        où l'IA "ne sait pas".

        Args:
            trajectory: Trajectoire (T, N)

        Returns:
            (has_singularity, singular_step)
        """
        for t_idx in range(len(trajectory)):
            q = trajectory[t_idx]
            curvature = self.metric.scalar_curvature(q)

            if abs(curvature) > self.config.curvature_threshold:
                return True, t_idx

        # Vérifier aussi la divergence de la trajectoire
        for t_idx in range(1, len(trajectory)):
            step = np.linalg.norm(trajectory[t_idx] - trajectory[t_idx - 1])
            if step > self.config.divergence_threshold:
                return True, t_idx

        return False, -1

    def get_barriers(self) -> List[np.ndarray]:
        """Retourne les positions des barrières topologiques."""
        return self._barriers

    def get_channels(self) -> List[np.ndarray]:
        """Retourne les positions des canaux sédimentés."""
        return self._channels

    def get_plasticity_stats(self) -> dict:
        """
        Retourne les statistiques de plasticité.

        Returns:
            Dict avec les compteurs et métriques.
        """
        return {
            "total_erosions": self._total_erosions,
            "total_sedimentations": self._total_sedimentations,
            "total_barriers": len(self._barriers),
            "total_channels": len(self._channels),
            "total_trajectories": len(self._trajectory_history),
            "success_rate": (
                sum(self._success_flags) / len(self._success_flags)
                if self._success_flags
                else 0.0
            ),
            "metric_det": float(np.linalg.det(self.metric.G)),
            "metric_trace": float(np.trace(self.metric.G)),
            "metric_condition": float(np.linalg.cond(self.metric.G)),
        }

    def __repr__(self) -> str:
        stats = self.get_plasticity_stats()
        return (
            f"TopologicalPlasticityEngine("
            f"erosions={stats['total_erosions']}, "
            f"sedimentations={stats['total_sedimentations']}, "
            f"barriers={stats['total_barriers']}, "
            f"channels={stats['total_channels']})"
        )
