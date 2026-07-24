"""
Couplage Kuramoto ↔ Tenseur Métrique G(t)
============================================

Système dynamique joint :
    dφ_i/dt = ω_i + (K/N) Σ_j sin(φ_j - φ_i) · G_ij(t)
    dG/dt   = NatGrad_SPD( L(q, dq, G, φ) )

La métrique module la synchronisation, la synchronisation module la métrique.
Boucle fermée. G(t) est de la mémoire morphologique, pas un paramètre statique.

Troisième dimension du scaling : params × data × morphological_steps.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable

from .config import MVTConfig
from .natural_gradient_spd import NaturalGradientSPD


@dataclass
class KuramotoMetricConfig:
    """
    Configuration du système couplé Kuramoto ↔ Métrique.

    Attributs:
        N:               Dimension de l'espace (nombre d'oscillateurs = ambient_dim).
        n_oscillators:   Nombre d'oscillateurs Kuramoto (≤ N, typiquement N ou N/2).
        K_coupling:      Force de couplage Kuramoto.
        omega_range:     Plage des fréquences naturelles ω_i ~ Uniform(omega_range).
        metric_lr:       Taux d'apprentissage pour la mise à jour de G via NatGrad.
        retraction_order: Ordre de la retraction ('exp' = exponentielle exacte,
                          'approx2' = second ordre, 'cholesky' = Log-Cholesky).
        phase_init:      Initialisation des phases ('random', 'sync', 'cluster').
        eigval_floor:    Valeur minimale des valeurs propres (garantit SPD).
        dt_kuramoto:     Pas de temps pour l'intégration Kuramoto (peut différer du RK4).
    """
    N: int = 128
    n_oscillators: int = 128
    K_coupling: float = 1.0
    omega_range: Tuple[float, float] = (-1.0, 1.0)
    metric_lr: float = 0.001
    retraction_order: str = "approx2"  # 'exp', 'approx2', 'cholesky'
    phase_init: str = "random"
    eigval_floor: float = 1e-4
    dt_kuramoto: float = 0.01
    # Couplage phase → métrique
    phase_metric_coupling: float = 0.1  # ε dans G_ij *= (1 + ε·cos(φ_i - φ_j))


class KuramotoDynamics:
    """
    Oscillateurs de Kuramoto couplés par la métrique G(t).

    dφ_i/dt = ω_i + (K/N) Σ_j sin(φ_j - φ_i) · G_ij(t)

    La métrique G pondère le couplage : les paires (i,j) dans des régions
    de l'espace sémantique « proches » (G_ij élevé) se synchronisent plus vite.
    """

    def __init__(self, config: KuramotoMetricConfig, seed: Optional[int] = 42):
        self.config = config
        self.N = config.N
        self.n_osc = config.n_oscillators
        self.K = config.K_coupling
        self.dt = config.dt_kuramoto

        rng = np.random.RandomState(seed)

        # Fréquences naturelles
        self.omega = rng.uniform(
            config.omega_range[0], config.omega_range[1], self.n_osc
        )

        # Initialisation des phases
        if config.phase_init == "random":
            self.phi = rng.uniform(0, 2 * np.pi, self.n_osc)
        elif config.phase_init == "sync":
            self.phi = np.ones(self.n_osc) * rng.uniform(0, 2 * np.pi)
        elif config.phase_init == "cluster":
            self.phi = np.zeros(self.n_osc)
            n_clusters = 4
            for i in range(self.n_osc):
                self.phi[i] = (i % n_clusters) * 2 * np.pi / n_clusters
        else:
            self.phi = rng.uniform(0, 2 * np.pi, self.n_osc)

        self.phi_history: list[np.ndarray] = []

    def order_parameter(self) -> float:
        """
        Paramètre d'ordre de Kuramoto : r ∈ [0, 1].
        r ≈ 1 → synchronisation complète, r ≈ 0 → incohérence.
        """
        return float(np.abs(np.mean(np.exp(1j * self.phi))))

    def phase_coherence(self, G: np.ndarray) -> float:
        """
        Cohérence phase-pondérée par la métrique.
        Mesure combien les phases alignées sont aussi proches dans l'espace sémantique.
        """
        N_eff = min(self.n_osc, G.shape[0])
        phi_slice = self.phi[:N_eff]
        G_slice = G[:N_eff, :N_eff]

        # Matrice de cohérence de phase
        coherence_matrix = np.cos(phi_slice[:, None] - phi_slice[None, :])
        # Pondérée par G
        weighted = np.sum(G_slice * coherence_matrix)
        normalization = np.sum(np.abs(G_slice)) + 1e-10
        return float(weighted / normalization)

    def compute_dphi(
        self, phi: np.ndarray, G: np.ndarray, t: float = 0.0
    ) -> np.ndarray:
        """
        dφ_i/dt = ω_i + (K/N) Σ_j sin(φ_j - φ_i) · G_ij(t)

        Args:
            phi: Phases actuelles (n_osc,)
            G:   Tenseur métrique (N, N) — seuls les n_osc premiers sont utilisés
            t:   Temps (pour ω_i éventuellement variables)

        Returns:
            Dérivées des phases (n_osc,)
        """
        N_eff = min(self.n_osc, G.shape[0])
        phi_eff = phi[:N_eff]
        G_eff = G[:N_eff, :N_eff]

        # Matrice de couplage: sin(φ_j - φ_i) pour chaque paire
        phase_diff = phi_eff[None, :] - phi_eff[:, None]  # (N_eff, N_eff)
        coupling = np.sin(phase_diff)  # (N_eff, N_eff)

        # Couplage pondéré par G
        weighted_coupling = coupling * G_eff  # (N_eff, N_eff)

        # Somme sur j
        dphi = self.omega[:N_eff] + (self.K / N_eff) * np.sum(
            weighted_coupling, axis=1
        )

        return dphi

    def step_euler(
        self, G: np.ndarray, dt: Optional[float] = None
    ) -> np.ndarray:
        """
        Pas d'Euler pour les phases Kuramoto.
        Stable si dt << 1/K.
        """
        dt = dt or self.dt
        dphi = self.compute_dphi(self.phi, G)
        self.phi = (self.phi + dphi * dt) % (2 * np.pi)
        return self.phi.copy()

    def step_rk4(self, G: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        """
        Pas de Runge-Kutta 4 pour les phases Kuramoto.
        Plus précis, surtout près des bifurcations de synchronisation.
        """
        dt = dt or self.dt
        phi0 = self.phi.copy()

        k1 = self.compute_dphi(phi0, G)
        k2 = self.compute_dphi((phi0 + 0.5 * dt * k1) % (2 * np.pi), G)
        k3 = self.compute_dphi((phi0 + 0.5 * dt * k2) % (2 * np.pi), G)
        k4 = self.compute_dphi((phi0 + dt * k3) % (2 * np.pi), G)

        self.phi = (phi0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)) % (2 * np.pi)
        return self.phi.copy()

    def record(self):
        """Enregistrer l'état actuel des phases."""
        self.phi_history.append(self.phi.copy())
        if len(self.phi_history) > 1000:
            self.phi_history.pop(0)


class KuramotoMetricCoupler:
    """
    Système couplé Kuramoto ↔ Métrique.

    Boucle fermée :
        1. Kuramoto synchronise les phases φ(t) modulées par G(t)
        2. G(t) est mis à jour via NatGrad_SPD(L(q, dq, G, φ))
        3. La cohérence de phase module la structure de G

    La métrique devient de la mémoire morphologique : chaque step d'intégration
    modifie G en même temps que la trajectoire évolue.
    """

    def __init__(
        self,
        mvt_config: Optional[MVTConfig] = None,
        kura_config: Optional[KuramotoMetricConfig] = None,
        G: Optional[np.ndarray] = None,
    ):
        mvt_config = mvt_config or MVTConfig()
        self.mvt_config = mvt_config

        if kura_config is None:
            kura_config = KuramotoMetricConfig(N=mvt_config.ambient_dim)
        self.kura_config = kura_config

        # Initialiser G si non fourni
        if G is None:
            N = mvt_config.ambient_dim
            self.G = np.eye(N, dtype=np.float64)
            perturbation = np.random.randn(N, N) * 0.01
            self.G += 0.5 * (perturbation + perturbation.T)
        else:
            self.G = G.copy()

        # Sous-systèmes
        self.kuramoto = KuramotoDynamics(kura_config, seed=mvt_config.seed)
        self.nat_grad = NaturalGradientSPD(
            lr=kura_config.metric_lr,
            retraction_order=kura_config.retraction_order,
            eigval_floor=kura_config.eigval_floor,
        )

        # État
        self._step_count = 0
        self._metrics_history: list[dict] = []

    # ================================================================
    # Couplage phase → métrique (modulation structurelle)
    # ================================================================

    def phase_modulate_metric(
        self, G: np.ndarray, strength: Optional[float] = None
    ) -> np.ndarray:
        """
        G_ij ← G_ij × (1 + ε · cos(φ_i - φ_j))

        Les paires synchronisées (φ_i ≈ φ_j) voient leur couplage renforcé.
        Les paires désynchronisées voient leur couplage affaibli.

        C'est le « clock géométrique » : la métrique pulse avec le rythme interne.
        """
        eps = strength or self.kura_config.phase_metric_coupling
        N_eff = min(self.kura_config.n_oscillators, G.shape[0])
        phi = self.kuramoto.phi[:N_eff]

        phase_diff = phi[:, None] - phi[None, :]
        modulation = 1.0 + eps * np.cos(phase_diff)

        G_new = G.copy()
        G_new[:N_eff, :N_eff] *= modulation

        # Recentrer pour éviter la dérive
        G_new = 0.5 * (G_new + G_new.T)

        return G_new

    # ================================================================
    # Couplage métrique → phase (via Kuramoto)
    # ================================================================

    def compute_coupled_dphi(self, G: np.ndarray) -> np.ndarray:
        """
        Version couplée : les phases sont modulées par G qui elle-même
        reflète la structure sémantique.
        """
        return self.kuramoto.compute_dphi(self.kuramoto.phi, G)

    # ================================================================
    # Boucle d'intégration complète
    # ================================================================

    def coupled_step(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        loss_fn: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]] = None,
    ) -> dict:
        """
        Un step complet du système couplé :

        1. Kuramoto RK4 : φ(t+dt) ← φ(t) + dφ·dt  (modulé par G)
        2. Phase → Metric : G* ← G × (1 + ε·cos(Δφ))  (modulation structurelle)
        3. NatGrad SPD   : G(t+dt) ← Retract(G*, -lr·∇_R L)  (mémoire morphologique)

        Args:
            q: Position dans l'espace sémantique (N,)
            dq: Vitesse (N,)
            loss_fn: Fonction de perte L(q, dq, G, φ) → float.
                     Si None, utilise le résidu lagrangien par défaut.

        Returns:
            Dictionnaire avec métriques du step.
        """
        dt = self.kura_config.dt_kuramoto

        # --- Étape 1 : Mettre à jour les phases via Kuramoto ---
        self.kuramoto.step_rk4(self.G, dt)
        self.kuramoto.record()

        # --- Étape 2 : Moduler G par la cohérence de phase ---
        G_modulated = self.phase_modulate_metric(self.G)

        # --- Étape 3 : Mettre à jour G via NatGrad SPD ---
        if loss_fn is not None:
            loss_val = loss_fn(q, dq, G_modulated, self.kuramoto.phi)
            grad_E = self._compute_euclidean_gradient(
                q, dq, G_modulated, self.kuramoto.phi, loss_fn
            )
        else:
            # Loss par défaut : résidu lagrangien
            loss_val, grad_E = self._default_lagrangian_residual(
                q, dq, G_modulated
            )

        # Appliquer le gradient naturel sur SPD
        self.G = self.nat_grad.retract(G_modulated, grad_E)

        # --- Normalisation post-retraction (sécurité) ---
        # Rescaler G pour maintenir tr(G) ≈ N
        trace_G = np.trace(self.G)
        if abs(trace_G - self.mvt_config.ambient_dim) > 0.5 * self.mvt_config.ambient_dim:
            self.G = self.G * (self.mvt_config.ambient_dim / trace_G)
            self.G = 0.5 * (self.G + self.G.T)

        # --- Métriques ---
        r = self.kuramoto.order_parameter()
        coherence = self.kuramoto.phase_coherence(self.G)
        eigvals = np.linalg.eigvalsh(self.G)

        metrics = {
            "step": self._step_count,
            "order_parameter": r,
            "phase_coherence": coherence,
            "loss": loss_val,
            "grad_norm": float(np.linalg.norm(grad_E)),
            "G_det": float(np.linalg.det(self.G)),
            "G_trace": float(np.trace(self.G)),
            "G_cond": float(np.linalg.cond(self.G)),
            "G_eigval_min": float(np.min(eigvals)),
            "G_eigval_max": float(np.max(eigvals)),
            "G_eigval_spread": float(np.max(eigvals) - np.min(eigvals)),
        }

        self._step_count += 1
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > 10000:
            self._metrics_history.pop(0)

        return metrics

    def _default_lagrangian_residual(
        self, q: np.ndarray, dq: np.ndarray, G: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Résidu lagrangien avec normalisation de la trace.

        Le problème critique : le NatGrad affine-invariant a grad_R = G·∇·G.
        Si G croît, le gradient croît quadratiquement → explosion.
        Solution : normaliser G à chaque step pour maintenir tr(G) = N.

        L = α·(T - V) + β·(tr(G) - N)² + γ·||log(G) - log(I)||²_F

        Terms :
          - T = dq^T G dq (énergie cinétique)
          - V = ||q||² (potentiel harmonique)
          - tr(G) = N (contrainte de trace, empêche l'explosion)
          - ||log(G)|| (contrainte log-eigenvalue, empêche collapse)

        Returns:
            (loss_value, euclidean_gradient)
        """
        N = G.shape[0]

        # --- Normaliser G pour empêcher la divergence ---
        # Centre la trace à N avant le calcul
        trace_G = np.trace(G)
        if trace_G > 1e6 or trace_G < 1e-6:
            # Débordement : rescaler
            G_normalized = G * (N / trace_G)
        else:
            G_normalized = G

        # Énergie cinétique normalisée : T = dq^T G dq / tr(G)
        # Division par tr(G) pour rendre T invariant au scaling
        T = float(dq @ G_normalized @ dq) / N

        # Potentiel : V = ||q||²
        V = float(np.dot(q, q)) / N

        # --- Contrainte de trace (critique pour stabilité) ---
        # tr(G) doit rester ≈ N
        trace_penalty = 1.0 * (trace_G - N) ** 2

        # --- Contrainte log-eigenvalue (empêche collapse/explosion) ---
        # Penalize eigenvalues far from 1 in log-space
        eigvals = np.linalg.eigvalsh(G)
        log_eigvals = np.log(np.maximum(eigvals, 1e-10))
        log_penalty = 0.5 * np.sum(log_eigvals ** 2)

        loss = T - V + trace_penalty + log_penalty

        # --- Gradients Euclidiens ---
        # ∂T/∂G = (1/N) dq ⊗ dq  (normalisé)
        grad_T = (1.0 / N) * np.outer(dq, dq)

        # ∂(trace_penalty)/∂G = 2·α·(tr(G) - N) · I
        grad_trace = 2.0 * 1.0 * (trace_G - N) * np.eye(N)

        # ∂(log_penalty)/∂G ≈ 2·γ·log(G)·G^{-1}  (développé via eigendecomposition)
        # Approximation : gradient dans l'espace des eigenvalues
        # ∂/∂G_ii Σ (log λ_i)² ≈ 2·log(λ_i)/λ_i pour les composantes diagonales
        try:
            G_inv = np.linalg.inv(G)
            grad_log = 0.5 * (2.0 * log_eigvals[:, None] * G_inv)
            grad_log = 0.5 * (grad_log + grad_log.T)
        except np.linalg.LinAlgError:
            grad_log = np.zeros_like(G)

        grad_E = grad_T + grad_trace + grad_log

        return loss, grad_E

    def _compute_euclidean_gradient(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        G: np.ndarray,
        phi: np.ndarray,
        loss_fn: Callable,
    ) -> np.ndarray:
        """
        Calcul numérique du gradient Euclidien de L par rapport à G.
        Différences finies centrées sur chaque élément de G.

        NOTE: O(N²) évaluations de loss_fn — coûteux, à utiliser
        uniquement pour validation. En production, utiliser l'analytique.
        """
        N = G.shape[0]
        eps_fd = 1e-5

        loss0 = loss_fn(q, dq, G, phi)
        grad_E = np.zeros_like(G)

        # Ne différencier que la partie supérieure (symétrie)
        for i in range(N):
            for j in range(i, N):
                G_plus = G.copy()
                G_plus[i, j] += eps_fd
                G_plus[j, i] += eps_fd

                loss_plus = loss_fn(q, dq, G_plus, phi)
                grad_E[i, j] = (loss_plus - loss0) / eps_fd
                grad_E[j, i] = grad_E[i, j]

        return grad_E

    # ================================================================
    # API
    # ================================================================

    def integrate_coupled(
        self,
        q0: np.ndarray,
        dq0: Optional[np.ndarray] = None,
        n_steps: int = 200,
        loss_fn: Optional[Callable] = None,
        callback: Optional[Callable[[int, dict], bool]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Intégration complète du système couplé sur n_steps.

        Args:
            q0: Position initiale (N,)
            dq0: Vitesse initiale (N,), None → random small
            n_steps: Nombre de steps couplés
            loss_fn: Fonction de perte optionnelle
            callback: Fonction appelée chaque step. Retourne True pour continuer.

        Returns:
            (phi_final, G_final, metrics_list)
        """
        N = self.mvt_config.ambient_dim

        if dq0 is None:
            dq0 = np.random.randn(N) * 0.01

        q = q0.copy()
        dq = dq0.copy()

        all_metrics = []

        for step in range(n_steps):
            metrics = self.coupled_step(q, dq, loss_fn)
            all_metrics.append(metrics)

            # Mise à jour q, dq via dynamique simplifiée (géodésique)
            # En production, ceci serait remplacé par le RK4 lagrangien complet
            G_inv = np.linalg.inv(self.G)
            Gamma = np.einsum('kl,lij->kij', G_inv,
                              np.random.randn(N, N, N) * 0.001)  # stub
            accel = -np.einsum('kij,i,j->k', Gamma, dq, dq) * 0.01
            dq = dq + accel * self.kura_config.dt_kuramoto
            q = q + dq * self.kura_config.dt_kuramoto

            if callback is not None:
                if not callback(step, metrics):
                    break

        return self.kuramoto.phi.copy(), self.G.copy(), all_metrics

    def get_sync_state(self) -> dict:
        """État de synchronisation actuel."""
        return {
            "order_parameter": self.kuramoto.order_parameter(),
            "phase_coherence": self.kuramoto.phase_coherence(self.G),
            "n_oscillators": self.kura_config.n_oscillators,
            "coupling_K": self.kura_config.K_coupling,
        }

    def get_metric_state(self) -> dict:
        """État du tenseur métrique."""
        eigvals = np.linalg.eigvalsh(self.G)
        return {
            "det": float(np.linalg.det(self.G)),
            "trace": float(np.trace(self.G)),
            "condition": float(np.linalg.cond(self.G)),
            "eigval_min": float(np.min(eigvals)),
            "eigval_max": float(np.max(eigvals)),
            "eigval_spread": float(np.max(eigvals) - np.min(eigvals)),
            "spd": bool(np.all(eigvals > 0)),
        }

    def __repr__(self) -> str:
        sync = self.get_sync_state()
        metric = self.get_metric_state()
        return (
            f"KuramotoMetricCoupler(\n"
            f"  N={self.kura_config.N},\n"
            f"  oscillators={sync['n_oscillators']},\n"
            f"  K={sync['coupling_K']},\n"
            f"  r={sync['order_parameter']:.3f},\n"
            f"  coherence={sync['phase_coherence']:.3f},\n"
            f"  G_spd={metric['spd']},\n"
            f"  G_det={metric['det']:.4e},\n"
            f"  steps={self._step_count},\n"
            f")"
        )
