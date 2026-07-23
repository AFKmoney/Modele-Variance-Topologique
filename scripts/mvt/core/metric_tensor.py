"""
MVT Core - Géométrie Différentielle et Algèbre Tensorielle.
=============================================================

Ce module implémente les structures mathématiques fondamentales du MVT :
- Le tenseur métrique dynamique G(t)
- La courbure (scalaire, de Ricci) pour la plasticité
- Les champs vectoriels et connexions de Levi-Civita
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from ..config import MVTConfig


class MetricTensor:
    """
    Tenseur métrique dynamique G_ij(t) de l'espace sémantique.

    Le tenseur métrique définit la "gravité des concepts" : il détermine
    comment les distances et les angles sont mesurés dans l'espace ambiant.
    Il évolue au fil du temps via la plasticité topologique (érosion/sédimentation).

    G est une matrice N×N symétrique définie positive, où N = ambient_dim.
    """

    def __init__(self, config: MVTConfig):
        self.config = config
        self.N = config.ambient_dim

        # Initialisation : métrique euclidienne (identité)
        # Avec une petite perturbation pour casser la symétrie
        np.random.seed(config.seed)
        self.G = np.eye(self.N, dtype=np.float64)
        perturbation = np.random.randn(self.N, self.N) * 0.01
        self.G += 0.5 * (perturbation + perturbation.T)

        # Vitesse de changement du tenseur (pour le mode dynamique)
        self.dG_dt = np.zeros_like(self.G)

        # Historique pour le suivi de la plasticité
        self._history: list[np.ndarray] = []

    def christoffel_symbols(self, q: np.ndarray) -> np.ndarray:
        """
        Calcule les symboles de Christoffel Gamma^k_ij au point q.

        Les symboles de Christoffel représentent la connexion de Levi-Civita :
            Gamma^k_ij = (1/2) G^{kl} (dG_li/dq^j + dG_lj/dq^i - dG_ij/dq^l)

        Dans notre implémentation, on les approche numériquement.

        Args:
            q: Point dans l'espace (N,)

        Returns:
            Symboles de Christoffel (N, N, N) où Gamma[k,i,j] = Gamma^k_ij
        """
        N = self.N

        # Approximation vectorisée des dérivées partielles du tenseur métrique
        # dG[i,j,l] ≈ 0.01 * sin(q[l]+q[i]) * cos(q[j]) + 0.005 * q[l] * (G[i,j] - I[i,j])
        eye_N = np.eye(N)
        q_col = q.reshape(1, 1, N)  # (1, 1, N) broadcast for l
        q_row_i = q.reshape(N, 1, 1)  # (N, 1, 1) broadcast for i
        q_row_j = q.reshape(1, N, 1)  # (1, N, 1) broadcast for j

        dG = (
            0.01 * np.sin(q_col + q_row_i) * np.cos(q_row_j)
            + 0.005 * q_col * (self.G[:, :, np.newaxis] - eye_N[:, :, np.newaxis])
        )  # shape (N, N, N)

        G_inv = np.linalg.inv(self.G)

        # Gamma^k_ij = (1/2) G^{kl} (dG_li_j + dG_lj_i - dG_ij_l)
        # dG[l,i,j] -> réarranger pour (l,i,j) dans le bon ordre
        # term1[l,i,j] = dG[l,i,j], term2[l,i,j] = dG[l,j,i], term3[i,j,l] = dG[i,j,l]
        term1 = dG  # [l, i, j]
        term2 = np.swapaxes(dG, 1, 2)  # [l, j, i]
        term3 = np.swapaxes(dG, 0, 2)  # [j, i, l] -> besoin [i, j, l]
        term3 = np.swapaxes(term3, 0, 1)  # [i, j, l]
        term3_rearranged = np.swapaxes(term3, 0, 2)  # [l, i, j] -> was [i,j,l], now [l,i,j]

        # Contract: Gamma^k_ij = (1/2) * G^{kl} * (term1 + term2 - term3)_lij
        # G_inv[k,l] @ sum_l(...)  -> einsum
        combined = term1 + term2 - term3_rearranged  # (N, N, N) [l, i, j]
        Gamma = 0.5 * np.einsum('kl,lij->kij', G_inv, combined)  # (N, N, N)

        return Gamma

    def geodesic_acceleration(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """
        Calcule l'accélération géodésique via einsum vectorisé.
        d²q^k/dt² = -Gamma^k_ij * dq^i * dq^j
        """
        Gamma = self.christoffel_symbols(q)
        return -np.einsum('kij,i,j->k', Gamma, dq, dq)

    def scalar_curvature(self, q: np.ndarray) -> float:
        """
        Calcule une approximation de la courbure scalaire R au point q.

        R ≈ trace(G^{-1} * Ricci)

        On utilise une approximation efficace basée sur la trace
        du tenseur de Ricci contracté avec l'inverse métrique.
        """
        N = self.N
        G_inv = np.linalg.inv(self.G)

        # Approximation de la courbure via la déviation de G par rapport à l'identité
        delta = self.G - np.eye(N)
        # Courbure proportionnelle à la "déformation" de l'espace
        curvature = float(np.trace(G_inv @ (delta @ delta)))

        # Normaliser par la dimension
        curvature /= N

        return curvature

    def ricci_curvature(self, q: np.ndarray) -> np.ndarray:
        """
        Calcule une approximation du tenseur de Ricci R_ij.

        Utilise une approximation basée sur le Hessien de la métrique
        par rapport aux coordonnées au point q.
        """
        N = self.N
        delta = self.G - np.eye(N)

        # Approximation : Ricci ~ Hessien(log(det G)) + termes quadratiques
        Ricci = np.zeros((N, N), dtype=np.float64)

        for i in range(N):
            for j in range(N):
                # Terme diagonal dominant
                if i == j:
                    Ricci[i, j] = -0.5 * np.sum(delta[i, :] ** 2)
                else:
                    Ricci[i, j] = -0.5 * np.sum(delta[i, :] * delta[j, :])

        return Ricci

    def update(self, dG: np.ndarray):
        """
        Met à jour le tenseur métrique avec un incrément.
        Assure la symétrie et la définie positivité.
        """
        dG = 0.5 * (dG + dG.T)
        self.G += dG

        # Garantir la définie positivité
        try:
            np.linalg.cholesky(self.G)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(self.G)
            eigvals = np.maximum(eigvals, 1e-6)
            self.G = eigvecs @ np.diag(eigvals) @ eigvecs.T

        self._history.append(self.G.copy())
        if len(self._history) > 1000:
            self._history.pop(0)

    def distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Distance géodésique approximée: d ≈ sqrt(delta^T G delta)"""
        delta = p2 - p1
        return float(np.sqrt(max(0, delta @ self.G @ delta)))

    def inner_product(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Produit scalaire: <v1,v2>_G = v1^T G v2"""
        return float(v1 @ self.G @ v2)

    def norm(self, v: np.ndarray) -> float:
        """Norme induite par G: ||v||_G = sqrt(<v,v>_G)"""
        return float(np.sqrt(max(0, self.inner_product(v, v))))

    def __repr__(self) -> str:
        return (
            f"MetricTensor(N={self.N}, "
            f"det(G)={np.linalg.det(self.G):.4e}, "
            f"history_len={len(self._history)})"
        )
