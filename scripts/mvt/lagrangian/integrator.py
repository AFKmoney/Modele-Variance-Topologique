"""
MVT Lagrangian - Intégrateur Euler-Lagrange (Runge-Kutta 4).
==============================================================

Résout les équations du mouvement par pas de temps différentiels.
La particule d'idée suit la courbe de moindre action dans l'espace
sémantique, guidée par le lagrangien et les contraintes topologiques.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Callable

from ..config import MVTConfig
from .semantic_lagrangian import SemanticLagrangian


class LagrangianIntegrator:
    """
    Intégrateur RK4 pour les équations d'Euler-Lagrange.

    Résout le système :
        dq/dt = dq
        d(dq)/dt = euler_lagrange_rhs(q, dq, t)

    par la méthode de Runge-Kutta d'ordre 4.
    """

    def __init__(self, config: MVTConfig, lagrangian: SemanticLagrangian):
        self.config = config
        self.lagrangian = lagrangian
        self.N = config.ambient_dim
        self.dt = config.dt

    def _derivatives(self, q: np.ndarray, dq: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule les dérivées (dq/dt, ddq/dt).

        Returns:
            (dq, ddq) - vitesse et accélération
        """
        ddq = self.lagrangian.euler_lagrange_rhs(q, dq, t)
        return dq.copy(), ddq

    def step_rk4(
        self, q: np.ndarray, dq: np.ndarray, t: float, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Un pas de Runge-Kutta d'ordre 4.

        Args:
            q: Position actuelle (N,)
            dq: Vitesse actuelle (N,)
            t: Temps actuel
            dt: Pas de temps

        Returns:
            (q_new, dq_new) - nouvelle position et vitesse
        """
        # k1
        dq1, ddq1 = self._derivatives(q, dq, t)

        # k2
        q2 = q + 0.5 * dt * dq1
        dq2 = dq + 0.5 * dt * ddq1
        dq_k2, ddq2 = self._derivatives(q2, dq2, t + 0.5 * dt)

        # k3
        q3 = q + 0.5 * dt * dq_k2
        dq3 = dq + 0.5 * dt * ddq2
        dq_k3, ddq3 = self._derivatives(q3, dq3, t + 0.5 * dt)

        # k4
        q4 = q + dt * dq_k3
        dq4 = dq + dt * ddq3
        dq_k4, ddq4 = self._derivatives(q4, dq4, t + dt)

        # Mise à jour
        q_new = q + (dt / 6.0) * (dq1 + 2 * dq_k2 + 2 * dq_k3 + dq_k4)
        dq_new = dq + (dt / 6.0) * (ddq1 + 2 * ddq2 + 2 * ddq3 + ddq4)

        return q_new, dq_new

    def integrate(
        self,
        q0: np.ndarray,
        dq0: Optional[np.ndarray] = None,
        num_steps: Optional[int] = None,
        callback: Optional[Callable[[int, np.ndarray, np.ndarray, float], bool]] = None,
    ) -> np.ndarray:
        """
        Intègre les équations d'Euler-Lagrange sur plusieurs pas.

        Génère la trajectoire complète de la particule d'idée dans
        l'espace sémantique.

        Args:
            q0: Position initiale (N,)
            dq0: Vitesse initiale (N,), ou None pour démarrage au repos
            num_steps: Nombre de pas, ou None pour utiliser la config
            callback: Fonction appelée à chaque pas.
                      Retourne True pour continuer, False pour arrêter.
                      Signature: (step, q, dq, t) -> bool

        Returns:
            Trajectoire de shape (num_steps+1, N)
        """
        if num_steps is None:
            num_steps = self.config.num_rk4_steps

        if dq0 is None:
            dq0 = np.random.randn(self.N) * 0.01

        trajectory = np.zeros((num_steps + 1, self.N), dtype=np.float64)
        trajectory[0] = q0.copy()

        q = q0.copy()
        dq = dq0.copy()
        t = 0.0
        dt = self.dt

        for step in range(num_steps):
            # Vérification de divergence
            if np.linalg.norm(q) > self.config.divergence_threshold:
                trajectory = trajectory[: step + 1]
                break

            # Vérification de courbure (singularité)
            curvature = self.lagrangian.metric.scalar_curvature(q)
            if abs(curvature) > self.config.curvature_threshold:
                # Singularité détectée - on arrête
                trajectory = trajectory[: step + 1]
                break

            # Pas RK4
            q, dq = self.step_rk4(q, dq, t, dt)
            t += dt
            trajectory[step + 1] = q.copy()

            # Callback utilisateur
            if callback is not None:
                if not callback(step, q, dq, t):
                    trajectory = trajectory[: step + 2]
                    break

        return trajectory

    def compute_action(self, trajectory: np.ndarray) -> float:
        """
        Calcule l'action S le long de la trajectoire.

        S = sum_t L(q(t), dq(t), t) * dt

        Lower action = more optimal path.
        """
        return self.lagrangian.action(trajectory, self.dt)

    def find_optimal_trajectory(
        self,
        q0: np.ndarray,
        q_target: np.ndarray,
        num_attempts: int = 5,
    ) -> Tuple[np.ndarray, float]:
        """
        Trouve la trajectoire de moindre action entre q0 et q_target.

        Essaie plusieurs conditions initiales de vitesse et retourne
        celle avec l'action minimale.

        Args:
            q0: Point de départ (N,)
            q_target: Point cible (N,)
            num_attempts: Nombre de tentatives

        Returns:
            (best_trajectory, best_action)
        """
        best_trajectory = None
        best_action = float('inf')

        direction = q_target - q0
        direction_norm = direction / (np.linalg.norm(direction) + 1e-8)

        for attempt in range(num_attempts):
            # Vitesse initiale dirigée vers la cible avec variation
            speed = 0.5 + attempt * 0.3
            dq0 = speed * direction_norm + np.random.randn(self.N) * 0.1

            trajectory = self.integrate(q0, dq0)
            action = self.compute_action(trajectory)

            if action < best_action:
                best_action = action
                best_trajectory = trajectory

        return best_trajectory, best_action
