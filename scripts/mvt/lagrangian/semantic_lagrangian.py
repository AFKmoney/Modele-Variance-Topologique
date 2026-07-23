"""
MVT Lagrangian - Mécanique Lagrangienne Sémantique.
=====================================================

Implémente le Lagrangien Sémantique L = T - V et les équations
du mouvement d'Euler-Lagrange qui gouvernent la génération de texte.

L'énergie cinétique sémantique T mesure la "fluidité" du discours.
L'énergie potentielle V mesure la cohérence logique et grammaticale.
La particule d'idée suit la courbe de moindre action.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Callable

from ..config import MVTConfig
from ..core.metric_tensor import MetricTensor
from ..core.vector_field import VectorField


class SemanticLagrangian:
    """
    Lagrangien Sémantique L(q, dq, t) = T(q, dq) - V(q, t).

    La génération de texte est formulée comme un problème de
    mécanique classique : trouver la trajectoire q(t) qui minimise
    l'action S = int L dt.
    """

    def __init__(self, config: MVTConfig, metric: MetricTensor):
        self.config = config
        self.metric = metric
        self.N = config.ambient_dim

        # Champ de potentiel externe (contraintes de cohérence)
        self.external_potential: Optional[Callable[[np.ndarray], float]] = None

        # Champ de force externe (déformations topologiques du prompt)
        self.external_force: Optional[Callable[[np.ndarray], np.ndarray]] = None

        # Paramètres de pondération
        self.kinetic_weight = config.kinetic_coupling
        self.potential_weight = config.potential_stiffness

    def kinetic_energy(self, q: np.ndarray, dq: np.ndarray) -> float:
        """
        Énergie cinétique sémantique T.

        T = (1/2) G_ij(q) dq^i dq^j

        Mesure la "vitesse conceptuelle" : un discours fluide a une
        énergie cinétique stable, un discours chaotique a une énergie
        cinétique erratique.

        Args:
            q: Position dans l'espace sémantique (N,)
            dq: Vitesse (dérivée temporelle) (N,)

        Returns:
            Énergie cinétique scalaire
        """
        # T = (1/2) * dq^T * G * dq
        T = 0.5 * self.kinetic_weight * (dq @ self.metric.G @ dq)
        return float(max(0, T))

    def potential_energy(self, q: np.ndarray, t: float = 0.0) -> float:
        """
        Énergie potentielle de contrainte V.

        V(q) = V_ext(q) + V_field(q)

        V_ext : Potentiel externe (cohérence logique, grammaticale).
        V_field : Potentiel du champ de force sémantique.

        Si V augmente, l'idée s'éloigne de la logique (comme un ressort
        qui s'étire).

        Args:
            q: Position (N,)
            t: Temps (pour potentiels dépendants du temps)

        Returns:
            Énergie potentielle scalaire
        """
        V = 0.0

        # Potentiel externe (s'il existe)
        if self.external_potential is not None:
            V += self.potential_weight * self.external_potential(q)

        # Terme de confinement : empêche la particule de s'échapper
        r = np.linalg.norm(q)
        if r > 5.0:
            V += self.potential_weight * 10.0 * (r - 5.0) ** 2

        return V

    def lagrangian(self, q: np.ndarray, dq: np.ndarray, t: float = 0.0) -> float:
        """
        Lagrangien total L = T - V.

        Args:
            q: Position (N,)
            dq: Vitesse (N,)
            t: Temps

        Returns:
            Valeur du lagrangien
        """
        T = self.kinetic_energy(q, dq)
        V = self.potential_energy(q, t)
        return T - V

    def euler_lagrange_rhs(self, q: np.ndarray, dq: np.ndarray, t: float) -> np.ndarray:
        """
        Second membre des équations d'Euler-Lagrange.

        d/dt(dL/d(dq^i)) - dL/dq^i = 0

        Avec G_ij constant spatialement (approximation), cela donne :
            G_ij d²q^j/dt² = -dV/dq^i + termes géodésiques

        On retourne d²q/dt² (l'accélération).

        Args:
            q: Position (N,)
            dq: Vitesse (N,)
            t: Temps

        Returns:
            Accélération ddq (N,)
        """
        N = self.N

        # Force du potentiel : F_i = -dV/dq^i
        F_potential = np.zeros(N, dtype=np.float64)
        epsilon = 1e-5

        for i in range(N):
            q_plus = q.copy(); q_plus[i] += epsilon
            q_minus = q.copy(); q_minus[i] -= epsilon
            F_potential[i] = -(self.potential_energy(q_plus, t) - self.potential_energy(q_minus, t)) / (2 * epsilon)

        # Force externe (champ de force du prompt)
        if self.external_force is not None:
            F_potential += self.external_force(q)

        # Accélération géodésique (connexion de Levi-Civita)
        geo_acc = self.metric.geodesic_acceleration(q, dq)

        # Force de friction sémantique (amortissement pour la stabilité)
        damping = -0.1 * dq

        # Résoudre pour ddq : G * ddq = F_potential + geo_acc + damping
        try:
            G_inv = np.linalg.inv(self.metric.G)
            ddq = G_inv @ (F_potential + geo_acc + damping)
        except np.linalg.LinAlgError:
            ddq = F_potential + geo_acc + damping

        return ddq

    def action(self, trajectory: np.ndarray, dt: float) -> float:
        """
        Calcule l'action S le long d'une trajectoire.

        S = sum_t L(q(t), dq(t), t) * dt

        Args:
            trajectory: Array de shape (T, N) avec les positions
            dt: Pas de temps

        Returns:
            Action totale
        """
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
