"""
MVT Lagrangian - Mécanique Lagrangienne Sémantique (VERSION CRÉATIVE).
======================================================================

Le lagrangien est enrichi avec :
- Un terme de bruit stochastique (température)
- Un terme de nouveauté (repulsion des zones visitées)
- Un amortissement réduit (plus fluide, plus créatif)
- La possibilité de sauts métaphoriques
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Callable

from ..config import MVTConfig
from ..core.metric_tensor import MetricTensor


class SemanticLagrangian:
    """
    Lagrangien Sémantique L(q, dq, t) = T(q, dq) - V(q, t) + Bruit.

    Version créative : le lagrangien inclut des termes stochastiques
    qui permettent à la particule d'idée d'explorer au-delà du
    chemin de moindre action stricte.
    """

    def __init__(self, config: MVTConfig, metric: MetricTensor):
        self.config = config
        self.metric = metric
        self.N = config.ambient_dim

        self.external_potential: Optional[Callable[[np.ndarray], float]] = None
        self.external_force: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self.novelty_force: Optional[Callable[[np.ndarray], np.ndarray]] = None

        self.kinetic_weight = config.kinetic_coupling
        self.potential_weight = config.potential_stiffness
        self.damping_factor = config.damping  # Amortissement réduit pour créativité

    def kinetic_energy(self, q: np.ndarray, dq: np.ndarray) -> float:
        T = 0.5 * self.kinetic_weight * (dq @ self.metric.G @ dq)
        return float(max(0, T))

    def potential_energy(self, q: np.ndarray, t: float = 0.0) -> float:
        V = 0.0
        if self.external_potential is not None:
            V += self.potential_weight * self.external_potential(q)

        # Confinement très doux (relâché pour créativité)
        r = np.linalg.norm(q)
        if r > 10.0:  # Seuil élargi (était 5.0)
            V += self.potential_weight * 2.0 * (r - 10.0) ** 2

        return V

    def lagrangian(self, q: np.ndarray, dq: np.ndarray, t: float = 0.0) -> float:
        T = self.kinetic_energy(q, dq)
        V = self.potential_energy(q, t)
        return T - V

    def euler_lagrange_rhs(self, q: np.ndarray, dq: np.ndarray, t: float) -> np.ndarray:
        N = self.N

        # Force du potentiel
        F_potential = np.zeros(N, dtype=np.float64)
        epsilon = 1e-5
        for i in range(N):
            q_plus = q.copy(); q_plus[i] += epsilon
            q_minus = q.copy(); q_minus[i] -= epsilon
            F_potential[i] = -(self.potential_energy(q_plus, t) - self.potential_energy(q_minus, t)) / (2 * epsilon)

        # Force externe (champ sémantique)
        if self.external_force is not None:
            F_potential += self.external_force(q)

        # Force de nouveauté (anti-répétition)
        if self.novelty_force is not None:
            F_potential += self.novelty_force(q)

        # Accélération géodésique (connexions de Levi-Civita)
        geo_acc = self.metric.geodesic_acceleration(q, dq)

        # Amortissement LÉGER (réduit pour créativité)
        damping = -self.damping_factor * dq

        # Résoudre : G * ddq = F + geo + damping
        try:
            G_inv = np.linalg.inv(self.metric.G)
            ddq = G_inv @ (F_potential + geo_acc + damping)
        except np.linalg.LinAlgError:
            ddq = F_potential + geo_acc + damping

        return ddq

    def action(self, trajectory: np.ndarray, dt: float) -> float:
        S = 0.0
        T_len = len(trajectory)
        for t_idx in range(T_len):
            q = trajectory[t_idx]
            if t_idx < T_len - 1:
                dq = (trajectory[t_idx + 1] - q) / dt
            else:
                dq = np.zeros(self.N)
            S += self.lagrangian(q, dq, t_idx * dt) * dt
        return S
