"""
MVT Syntopy - Opérateur de Syntopie et Couche One-Shot.
==========================================================

Implémente l'opérateur ★ (syntopie) qui fusionne les topologies
de l'exemple et de la requête pour permettre le one-shot absolu.

Pas de fine-tuning, pas de gradient. La topologie de l'exemple
s'imprime instantanément sur celle de la requête via déformation
géométrique.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, List

from .config import MVTConfig
from .core.metric_tensor import MetricTensor
from .core.vector_field import VectorField
from .encoder import InputEncoder


class SyntopicLayer:
    """
    Couche de Syntopie : fusion topologique One-Shot.

    L'opérateur ★ fusionne deux variétés sémantiques :
        tau = nabla * (M_requete ⊕ M_exemple)

    La topologie de l'exemple s'imprime sur la requête, contraignant
    la génération future sans aucun apprentissage.
    """

    def __init__(
        self,
        config: MVTConfig,
        metric: MetricTensor,
        encoder: InputEncoder,
    ):
        self.config = config
        self.metric = metric
        self.encoder = encoder
        self.N = config.ambient_dim

        # Exemple stocké (pour le one-shot)
        self._example_field: Optional[VectorField] = None
        self._example_center: Optional[np.ndarray] = None
        self._example_text: Optional[str] = None

        # Force de l'opérateur ★
        self.syntopy_strength = config.syntopy_strength

    def set_example(self, example_text: str):
        """
        Définit l'exemple pour le one-shot.

        L'exemple est encodé une seule fois et stocké. Toute requête
        ultérieure sera automatiquement "déformée" par cette topologie.

        Args:
            example_text: Texte de l'exemple
        """
        self._example_text = example_text
        self._example_field, self._example_center = self.encoder.encode_prompt(
            example_text
        )

    def clear_example(self):
        """Supprime l'exemple courant."""
        self._example_field = None
        self._example_center = None
        self._example_text = None

    def _compute_tension_field(
        self, request_field: VectorField
    ) -> np.ndarray:
        """
        Calcule le champ de tension tau entre la requête et l'exemple.

        tau = nabla * (M_requete ⊕ M_exemple)

        Le champ de tension représente la "force" avec laquelle l'exemple
        tire la requête vers sa topologie.

        Returns:
            Champ de tension comme déformation du tenseur métrique (N, N)
        """
        if self._example_field is None:
            return np.zeros((self.N, self.N), dtype=np.float64)

        # ⊕ (fusion) : concaténer les points de contrôle
        request_positions = np.array(request_field.control_points)
        example_positions = np.array(self._example_field.control_points)
        example_strengths = np.array(self._example_field.strengths)

        if len(request_positions) == 0 or len(example_positions) == 0:
            return np.zeros((self.N, self.N), dtype=np.float64)

        # Calcul du "centre de tension" entre les deux variétés
        request_center = np.mean(request_positions, axis=0)
        example_center = self._example_center

        # Direction de la tension
        tension_dir = example_center - request_center
        tension_norm = np.linalg.norm(tension_dir)
        if tension_norm > 1e-8:
            tension_dir /= tension_norm

        # Construction du champ de tension comme tenseur
        # Le tenseur de tension déforme l'espace pour "tirer" la requête
        # vers la topologie de l'exemple
        tau = np.outer(tension_dir, tension_dir) * self.syntopy_strength

        # Ajouter les effets des points de contrôle de l'exemple
        for pos, strength in zip(example_positions, example_strengths):
            delta = pos - request_center
            dist = np.linalg.norm(delta)
            if dist > 1e-8:
                weight = strength / (dist + 1.0)
                tau += weight * np.outer(delta, delta) / (dist ** 2)

        # Symétriser et normaliser
        tau = 0.5 * (tau + tau.T)

        # Normaliser pour éviter les déformations excessives
        max_eigenvalue = np.max(np.abs(np.linalg.eigvalsh(tau)))
        if max_eigenvalue > 1.0:
            tau /= max_eigenvalue

        return tau

    def apply_syntopy(
        self, request_field: VectorField, query_text: str
    ) -> Tuple[VectorField, np.ndarray]:
        """
        Applique l'opérateur de syntopie ★ à une requête.

        Si un exemple a été défini, fusionne sa topologie avec celle
        de la requête pour contraindre la génération.

        Args:
            request_field: Champ de la requête
            query_text: Texte de la requête (pour le logging)

        Returns:
            (champ_fusionné, déformation_métrique)
        """
        tau = self._compute_tension_field(request_field)

        if np.allclose(tau, 0):
            return request_field, tau

        # Appliquer la déformation : ajouter les points de contrôle
        # de l'exemple pondérés au champ de la requête
        if self._example_field is not None:
            for pos, strength in zip(
                self._example_field.control_points,
                self._example_field.strengths,
            ):
                # Projection : on ne copie pas l'exemple, on déforme
                # la requête vers la topologie de l'exemple
                deformed_pos = pos * self.syntopy_strength + (
                    request_field.control_points[0]
                    if request_field.control_points
                    else np.zeros(self.N)
                ) * (1 - self.syntopy_strength)
                request_field.add_source(deformed_pos, strength * 0.5)

        return request_field, tau

    def get_example_trajectory(self) -> Optional[np.ndarray]:
        """
        Retourne la trajectoire de l'exemple si disponible.

        La trajectoire de l'exemple est le chemin à travers les points
        de contrôle dans l'ordre.
        """
        if self._example_field is None:
            return None

        positions = self._example_field.control_points
        if len(positions) < 2:
            return None

        return np.array(positions)

    def compute_syntopy_score(
        self, generated_trajectory: np.ndarray
    ) -> float:
        """
        Calcule un score de "syntopie" : à quel point la trajectoire
        générée respecte la topologie de l'exemple.

        Score entre 0 (pas de correspondance) et 1 (parfait alignement).

        Args:
            generated_trajectory: Trajectoire générée (T, N)

        Returns:
            Score de syntopie [0, 1]
        """
        if self._example_field is None:
            return 0.0

        example_traj = self.get_example_trajectory()
        if example_traj is None:
            return 0.0

        # Comparer les directions principales des deux trajectoires
        gen_mean = np.mean(generated_trajectory, axis=0)
        ex_mean = np.mean(example_traj, axis=0)

        # Distance normalisée
        dist = self.metric.distance(gen_mean, ex_mean)
        similarity = 1.0 / (1.0 + dist)

        # Comparer les covariance (structure topologique)
        if len(generated_trajectory) > 1:
            gen_cov = np.cov(generated_trajectory.T)
            ex_cov = np.cov(example_traj.T)

            # Distance de Frobenius entre les covariances
            cov_dist = np.linalg.norm(gen_cov - ex_cov, 'fro')
            cov_sim = 1.0 / (1.0 + cov_dist * 0.1)
        else:
            cov_sim = 0.0

        return 0.6 * similarity + 0.4 * cov_sim

    def __repr__(self) -> str:
        has_example = self._example_field is not None
        return (
            f"SyntopicLayer(N={self.N}, "
            f"has_example={has_example}, "
            f"strength={self.syntopy_strength:.2f})"
        )
