"""
MVT Core - Champ Vectoriel Continu.
=====================================

Implémente les champs de force sémantiques dans R^N et les opérations
de déformation topologique.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Callable

from ..config import MVTConfig


class VectorField:
    """
    Champ vectoriel continu F: R^N -> R^N représentant un état sémantique.

    Un prompt ou un concept est modélisé comme un champ de force dans
    l'espace ambiant. La "particule d'idée" se déplace sous l'influence
    de ce champ.
    """

    def __init__(self, config: MVTConfig):
        self.config = config
        self.N = config.ambient_dim

        # Points de contrôle du champ (sources et puits)
        self.control_points: list[np.ndarray] = []
        self.strengths: list[float] = []

        # Grille discrétisée pour l'évaluation rapide (lazy)
        self._grid = None
        self._grid_res = 16

    def add_source(self, position: np.ndarray, strength: float = 1.0):
        """
        Ajoute un point source (attracteur) au champ.

        Un attracteur crée un "puits" dans le paysage énergétique,
        guidant la particule d'idée vers ce concept.
        """
        assert len(position) == self.N
        self.control_points.append(position.copy())
        self.strengths.append(strength)
        self._invalidate_cache()

    def add_sink(self, position: np.ndarray, strength: float = 1.0):
        """
        Ajoute un point puits (répulseur) au champ.

        Un répulseur crée une "barrière" dans le paysage énergétique.
        """
        self.add_source(position, -strength)

    def evaluate(self, q: np.ndarray) -> np.ndarray:
        """
        Évalue le champ vectoriel au point q.

        F(q) = sum_i (strength_i * (control_i - q) / |control_i - q|^3)

        C'est un champ de type Coulomb/gravité en N dimensions.

        Args:
            q: Point dans l'espace (N,)

        Returns:
            Force au point q (N,)
        """
        F = np.zeros(self.N, dtype=np.float64)

        for pos, strength in zip(self.control_points, self.strengths):
            delta = pos - q
            dist = np.linalg.norm(delta)
            if dist < 1e-8:
                dist = 1e-8

            # Champ en 1/r^(N-1) (gravité en N dimensions)
            F += strength * delta / (dist ** (self.N - 1) + 1e-6)

        return F

    def potential(self, q: np.ndarray) -> float:
        """
        Calcule le potentiel scalaire au point q.

        V(q) = -sum_i (strength_i / |q - control_i|^(N-2))

        (pour N > 2, sinon logarithmique)
        """
        V = 0.0
        for pos, strength in zip(self.control_points, self.strengths):
            dist = np.linalg.norm(q - pos)
            if dist < 1e-8:
                dist = 1e-8
            if self.N > 2:
                V -= strength / (dist ** (self.N - 2))
            else:
                V -= strength * np.log(dist + 1e-8)
        return V

    def gradient(self, q: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        """
        Calcule le gradient du potentiel numériquement.

        grad V ≈ (V(q + eps*e_i) - V(q - eps*e_i)) / (2*eps) pour chaque i
        """
        grad = np.zeros(self.N, dtype=np.float64)
        for i in range(self.N):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[i] += epsilon
            q_minus[i] -= epsilon
            grad[i] = (self.potential(q_plus) - self.potential(q_minus)) / (2 * epsilon)
        return grad

    def divergence(self, q: np.ndarray, epsilon: float = 1e-5) -> float:
        """
        Calcule la divergence du champ vectoriel au point q.

        div F = sum_i dF^i/dq^i
        """
        div = 0.0
        for i in range(self.N):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[i] += epsilon
            q_minus[i] -= epsilon
            F_plus = self.evaluate(q_plus)
            F_minus = self.evaluate(q_minus)
            div += (F_plus[i] - F_minus[i]) / (2 * epsilon)
        return float(div)

    def curl_magnitude(self, q: np.ndarray, epsilon: float = 1e-5) -> float:
        """
        Calcule la magnitude du rotationnel (pour N=3, ou généralisé).

        En N dimensions, le rotationnel est un tenseur antisymétrique d'ordre 2.
        On retourne la norme de Frobenius.
        """
        N = self.N
        curl = np.zeros((N, N), dtype=np.float64)

        for i in range(N):
            for j in range(N):
                if i != j:
                    # dF^j/dq^i - dF^i/dq^j
                    q_plus_i = q.copy(); q_plus_i[i] += epsilon
                    q_minus_i = q.copy(); q_minus_i[i] -= epsilon
                    q_plus_j = q.copy(); q_plus_j[j] += epsilon
                    q_minus_j = q.copy(); q_minus_j[j] -= epsilon

                    dFj_di = (self.evaluate(q_plus_i)[j] - self.evaluate(q_minus_i)[j]) / (2 * epsilon)
                    dFi_dj = (self.evaluate(q_plus_j)[i] - self.evaluate(q_minus_j)[i]) / (2 * epsilon)
                    curl[i, j] = dFj_di - dFi_dj

        return float(np.linalg.norm(curl))

    def deform(self, center: np.ndarray, radius: float, intensity: float):
        """
        Applique une déformation topologique locale (colline ou vallée).

        Args:
            center: Centre de la déformation (N,)
            radius: Rayon d'influence
            intensity: Intensité (positive = attracteur, négative = barrière)
        """
        # On ajoute un point de contrôle pondéré
        self.add_source(center, intensity * radius)

    def _invalidate_cache(self):
        """Invalide le cache de la grille discrétisée."""
        self._grid = None

    def to_grid(self, resolution: int = 16) -> tuple:
        """
        Convertit le champ en grille discrétisée pour visualisation.

        Returns:
            (grid_points, grid_forces, grid_potentials)
        """
        # Limiter aux 2 premières dimensions pour la visualisation
        points = []
        forces = []
        potentials = []

        lin = np.linspace(-2, 2, resolution)
        for x in lin:
            for y in lin:
                q = np.zeros(self.N)
                q[0] = x
                q[1] = y
                points.append(q[:2].copy())
                forces.append(self.evaluate(q)[:2].copy())
                potentials.append(self.potential(q))

        return np.array(points), np.array(forces), np.array(potentials)

    def __repr__(self) -> str:
        return (
            f"VectorField(N={self.N}, "
            f"control_points={len(self.control_points)})"
        )
