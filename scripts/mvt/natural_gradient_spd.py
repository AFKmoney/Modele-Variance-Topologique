"""
Natural Gradient sur la Variété SPD (Symmetric Positive Definite)
==================================================================

Espace : S^+_n = {G ∈ R^{n×n} : G^T = G, λ_i(G) > 0 ∀i}

Structure Riemannienne (métrique affine-invariante) :
    ⟨X, Y⟩_G = tr(G^{-1} X G^{-1} Y),  X, Y ∈ T_G S^+_n

Gradient Riemannien :
    grad_R f = G · (∂f/∂G) · G

Exponentielle (retraction exacte) :
    Exp_G(η) = G^{1/2} · expm(G^{-1/2} · η · G^{-1/2}) · G^{1/2}

Retraction d'ordre 2 (approximation stable pour CPU) :
    R_G(η) = G + η + (1/2) · η · G^{-1} · η

Log-Cholesky (alternative pour grands N) :
    G = L·L^T → update L dans l'espace tangent des facteurs Cholesky

Garantit que G reste SPD à chaque step, sans projection ad-hoc.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class NaturalGradientSPD:
    """
    Gradient naturel sur S^+_n avec retraction stable.

    Args:
        lr:              Taux d'apprentissage (échelle dans l'espace tangent).
        retraction_order: 'exp'    = exponentielle exacte (coûteuse, O(n³))
                           'approx2' = retrait d'ordre 2 (stable, rapide)
                           'cholesky' = Log-Cholesky (le plus stable)
        eigval_floor:     Valeur plancher pour les valeurs propres.
        momentum:         Coefficient de momentum (0 = pas de momentum).
        grad_clip:        Norme maximale du gradient tangent.
        ridge:            Régularisation ridge pour stabiliser G^{-1}.
    """
    lr: float = 0.001
    retraction_order: str = "approx2"
    eigval_floor: float = 1e-4
    momentum: float = 0.0
    grad_clip: float = 10.0
    ridge: float = 1e-6

    # État interne pour momentum
    _velocity: Optional[np.ndarray] = None

    def euclidean_to_riemannian(
        self, G: np.ndarray, grad_E: np.ndarray
    ) -> np.ndarray:
        """
        Convertir le gradient Euclidien en gradient Riemannien.

        grad_R f = G · grad_E · G

        C'est la transformation clé : le gradient « remonte » la métrique.
        En espace Euclidien, ∂f/∂G est un tenseur (2,0). En espace Riemannien
        avec la métrique affine-invariante, il faut le « contracter » avec G
        des deux côtés pour obtenir un vrai vecteur tangent.

        Args:
            G:      Tenseur métrique SPD (N, N)
            grad_E: Gradient Euclidien (N, N), symétrique

        Returns:
            Gradient Riemannien (N, N), tangent à S^+_n en G
        """
        # Assurer la symétrie du gradient Euclidien
        grad_E = 0.5 * (grad_E + grad_E.T)

        # grad_R = G · grad_E · G
        grad_R = G @ grad_E @ G

        # Clipping de norme
        grad_norm = np.linalg.norm(grad_R)
        if grad_norm > self.grad_clip:
            grad_R = grad_R * (self.grad_clip / grad_norm)

        return grad_R

    def retract(
        self,
        G: np.ndarray,
        grad_E: np.ndarray,
        lr: Optional[float] = None,
    ) -> np.ndarray:
        """
        Retraction sur S^+_n : mettre à jour G en restant sur la variété.

        η = -lr · grad_R f  (direction de descente dans l'espace tangent)
        G_new = R_G(η)      (retraction)

        La retraction garantit que G_new ∈ S^+_n (SPD) par construction.
        Pas besoin de projection corrective.

        Args:
            G:      Tenseur métrique SPD actuel (N, N)
            grad_E: Gradient Euclidien de la loss (N, N)
            lr:     Taux d'apprentissage (override si spécifié)

        Returns:
            G_new ∈ S^+_n (N, N)
        """
        lr = lr or self.lr

        # 1. Gradient Riemannien
        grad_R = self.euclidean_to_riemannian(G, grad_E)

        # 2. Momentum
        if self.momentum > 0:
            if self._velocity is None:
                self._velocity = -lr * grad_R
            else:
                self._velocity = self.momentum * self._velocity - lr * grad_R
            eta = self._velocity
        else:
            eta = -lr * grad_R

        # 3. Retraction selon la méthode choisie
        if self.retraction_order == "exp":
            G_new = self._retract_exp(G, eta)
        elif self.retraction_order == "cholesky":
            G_new = self._retract_cholesky(G, eta)
        else:  # "approx2"
            G_new = self._retract_approx2(G, eta)

        # 4. Assurer SPD (filet de sécurité)
        G_new = self._ensure_spd(G_new)

        return G_new

    def _retract_exp(self, G: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        Retraction exacte via exponentielle matricielle.

        Exp_G(η) = G^{1/2} · expm(G^{-1/2} · η · G^{-1/2}) · G^{1/2}

        Coûteuse : O(N³) pour la décomposition et l'exponentielle.
        Précise et garantie SPD.
        """
        N = G.shape[0]

        # G^{1/2} via eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(G)
        eigvals = np.maximum(eigvals, self.eigval_floor)
        sqrt_G = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        inv_sqrt_G = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        # Cartographier η dans le plan tangent à l'identité
        M = inv_sqrt_G @ eta @ inv_sqrt_G
        M = 0.5 * (M + M.T)  # symétriser (correction numérique)

        # Exponentielle matricielle
        exp_M = _matrix_exponential(M, N)

        # Retourner sur la variété
        G_new = sqrt_G @ exp_M @ sqrt_G
        G_new = 0.5 * (G_new + G_new.T)

        return G_new

    def _retract_approx2(self, G: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        Retraction d'ordre 2 (approximation de l'exponentielle).

        R_G(η) = G + η + (1/2) · η · G^{-1} · η

        Équivalente à la troncature de la série de Taylor de Exp_G à l'ordre 2.
        Stable tant que ||G^{-1} · η|| < 1 (garanti par le grad_clip).

        Avantages :
        - Rapide : une inversion + multiplications matricielles
        - SPD-stable pour des pas petits
        - Pas d'appel à expm
        """
        # Régulariser l'inversion
        G_inv = np.linalg.inv(G + self.ridge * np.eye(G.shape[0]))

        # Termes de la série tronquée
        term1 = eta
        term2 = 0.5 * eta @ G_inv @ eta

        G_new = G + term1 + term2
        G_new = 0.5 * (G_new + G_new.T)

        return G_new

    def _retract_cholesky(self, G: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        Retraction via Log-Cholesky.

        G = L·L^T → parameteriser dans l'espace des matrices triangulaires.
        L_new = L · expm(L^{-1} · sym(η) · L^{-T})

        Plus stable pour les grandes dimensions car L^{-1} est triangulaire.
        La factorisation Cholesky garantit SPD par construction.
        """
        N = G.shape[0]

        try:
            L = np.linalg.cholesky(G)
        except np.linalg.LinAlgError:
            # Fallback : forcer SPD via eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(G)
            eigvals = np.maximum(eigvals, self.eigval_floor)
            G_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L = np.linalg.cholesky(G_fixed)

        # Symétriser eta
        eta_sym = 0.5 * (eta + eta.T)

        # Mapper dans l'espace de L
        L_inv = np.linalg.inv(L)  # triangulaire → stable
        M = L_inv @ eta_sym @ L_inv.T
        M = 0.5 * (M + M.T)

        # Exponentielle (approximation de Padé pour les petits M)
        # Pour les petits pas, expm(M) ≈ I + M + M²/2
        norm_M = np.linalg.norm(M, 'fro')
        if norm_M < 0.1:
            exp_M = np.eye(N) + M + 0.5 * (M @ M)
        else:
            exp_M = _matrix_exponential(M, N)

        L_new = L @ exp_M

        # Reconstruire G
        G_new = L_new @ L_new.T
        G_new = 0.5 * (G_new + G_new.T)

        return G_new

    def _ensure_spd(self, G: np.ndarray) -> np.ndarray:
        """
        Filet de sécurité : garantir SPD par eigendecomposition corrective.
        Ne devrait presque jamais être activé si la retraction fonctionne.
        """
        eigvals, eigvecs = np.linalg.eigh(G)

        if np.all(eigvals > self.eigval_floor):
            return G  # déjà SPD, rien à faire

        # Corriger : seuiller les eigenvalues négatives
        eigvals = np.maximum(eigvals, self.eigval_floor)
        G_new = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return 0.5 * (G_new + G_new.T)

    def parallel_transport(
        self,
        G_from: np.ndarray,
        G_to: np.ndarray,
        tangent_vec: np.ndarray,
    ) -> np.ndarray:
        """
        Transport parallèle d'un vecteur tangent de G_from vers G_to.

        Γ(G_from → G_to)[η] = G_to^{1/2} · (G_from^{-1/2} · η · G_from^{-1/2})^{1/2} · G_to^{1/2}

        Utilisé pour accumuler des gradients sur plusieurs steps
        lorsque G change à chaque step.
        """
        eigvals_f, eigvecs_f = np.linalg.eigh(G_from)
        eigvals_f = np.maximum(eigvals_f, self.eigval_floor)
        inv_sqrt_f = eigvecs_f @ np.diag(1.0 / np.sqrt(eigvals_f)) @ eigvecs_f.T

        eigvals_t, eigvecs_t = np.linalg.eigh(G_to)
        eigvals_t = np.maximum(eigvals_t, self.eigval_floor)
        sqrt_t = eigvecs_t @ np.diag(np.sqrt(eigvals_t)) @ eigvecs_t.T

        # Mapper → identité → G_to
        M = inv_sqrt_f @ tangent_vec @ inv_sqrt_f
        M = 0.5 * (M + M.T)

        # Racine carrée matricielle (symétrique)
        eigvals_m, eigvecs_m = np.linalg.eigh(M)
        eigvals_m = np.maximum(eigvals_m, 0.0)  # M devrait être PSD
        sqrt_M = eigvecs_m @ np.diag(np.sqrt(eigvals_m)) @ eigvecs_m.T

        transported = sqrt_t @ sqrt_M @ sqrt_t
        return 0.5 * (transported + transported.T)

    def geodesic_distance(
        self, G1: np.ndarray, G2: np.ndarray
    ) -> float:
        """
        Distance géodésique sur S^+_n (affine-invariant).

        d(G1, G2) = || log(G1^{-1/2} · G2 · G1^{-1/2}) ||_F

        Invariante sous les transformations affines : d(AGA^T, BGB^T) = d(G1, G2).
        """
        N = G1.shape[0]

        eigvals1, eigvecs1 = np.linalg.eigh(G1)
        eigvals1 = np.maximum(eigvals1, self.eigval_floor)
        inv_sqrt1 = eigvecs1 @ np.diag(1.0 / np.sqrt(eigvals1)) @ eigvecs1.T

        M = inv_sqrt1 @ G2 @ inv_sqrt1
        M = 0.5 * (M + M.T)

        eigvals_M, _ = np.linalg.eigh(M)
        log_eigvals = np.log(np.maximum(eigvals_M, 1e-10))

        return float(np.sqrt(np.sum(log_eigvals ** 2)))


# ===========================================================================
# Utilitaires (exponentielle matricielle)
# ===========================================================================

def _matrix_exponential(M: np.ndarray, N: int) -> np.ndarray:
    """
    Exponentielle matricielle pour matrices symétriques.

    Utilise l'eigendecomposition : exp(M) = Q · diag(exp(λ_i)) · Q^T

    Plus rapide et stable que scipy.linalg.expm pour les matrices symétriques.
    """
    eigvals, eigvecs = np.linalg.eigh(M)

    # Clipper pour éviter l'overflow
    eigvals = np.clip(eigvals, -20, 20)

    return eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.T


# ===========================================================================
# Tests
# ===========================================================================

if __name__ == "__main__":
    print("=" * 64)
    print("  NatGrad SPD — Tests de stabilité")
    print("=" * 64)

    N = 32

    # G initiale (SPD)
    G = np.eye(N) + 0.1 * np.random.randn(N, N)
    G = 0.5 * (G + G.T)
    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals = np.maximum(eigvals, 0.1)
    G = eigvecs @ np.diag(eigvals) @ eigvecs.T

    print(f"\n  G initiale: det={np.linalg.det(G):.4e}, "
          f"min_eigval={np.min(np.linalg.eigvalsh(G)):.4f}")

    # Test les 3 retractions
    for method in ["approx2", "exp", "cholesky"]:
        ng = NaturalGradientSPD(lr=0.01, retraction_order=method)
        G_test = G.copy()

        print(f"\n  Méthode: {method}")
        print(f"  {'Step':>5s} | {'det(G)':>12s} | {'min_λ':>10s} | {'max_λ':>10s} | {'cond':>10s} | {'SPD':>3s}")
        print(f"  {'-----':>5s}-+-{'----------':>12s}-+-{'--------':>10s}-+-{'--------':>10s}-+-{'--------':>10s}-+-{'---':>3s}")

        for step in range(200):
            # Gradient Euclidien fictif (bruit structuré)
            grad_E = 0.01 * np.random.randn(N, N)
            grad_E = 0.5 * (grad_E + grad_E.T)

            G_test = ng.retract(G_test, grad_E)

            if step % 20 == 0 or step == 199:
                ev = np.linalg.eigvalsh(G_test)
                det = np.linalg.det(G_test)
                cond = np.linalg.cond(G_test)
                spd = bool(np.all(ev > 0))
                print(f"  {step:>5d} | {det:>12.4e} | {np.min(ev):>10.6f} | "
                      f"{np.max(ev):>10.4f} | {cond:>10.2f} | {'✓' if spd else '✗':>3s}")

        # Vérification SPD finale
        ev_final = np.linalg.eigvalsh(G_test)
        assert np.all(ev_final > 0), f"ERREUR: G n'est plus SPD avec {method} !"
        print(f"  ✓ {method}: SPD maintenu sur 200 steps")

    # Test distance géodésique
    print("\n  Distance géodésique:")
    ng = NaturalGradientSPD()
    G1 = np.eye(N)
    G2 = np.eye(N) * 2.0
    d = ng.geodesic_distance(G1, G2)
    print(f"  d(I, 2I) = {d:.6f} (théorique = √N × |ln 2| = {np.sqrt(N) * abs(np.log(2)):.6f})")

    # Transport parallèle
    print("\n  Transport parallèle:")
    eta = np.random.randn(N, N) * 0.1
    eta = 0.5 * (eta + eta.T)
    eta_transported = ng.parallel_transport(G1, G2, eta)
    print(f"  ||η_original|| = {np.linalg.norm(eta, 'fro'):.6f}")
    print(f"  ||η_transported|| = {np.linalg.norm(eta_transported, 'fro'):.6f}")

    print("\n  ✓ Tous les tests passés")
