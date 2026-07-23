"""
MVT - Modèle à Variance Topologique (Orchestrateur Principal).
================================================================

Le MVT est le point d'entrée principal du système. Il orchestre les
5 composants de l'architecture :

1. InputEncoder : Texte → Champ de force F_0 dans R^N
2. SyntopicLayer : Opérateur ★ pour le one-shot
3. LagrangianIntegrator : Euler-Lagrange + RK4
4. TopologicalPlasticityEngine : Auto-évolution géométrique
5. ManifoldProjector : Courbe continue → Texte

Usage:
    model = MVT(MVTConfig())
    model.set_example("Exemple de format...")
    result = model.generate("Requête...")
"""

from __future__ import annotations

import time
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .config import MVTConfig
from .core.metric_tensor import MetricTensor
from .core.vector_field import VectorField
from .lagrangian.semantic_lagrangian import SemanticLagrangian
from .lagrangian.integrator import LagrangianIntegrator
from .encoder import InputEncoder
from .syntopy import SyntopicLayer
from .plasticity import TopologicalPlasticityEngine
from .projector import ManifoldProjector


@dataclass
class GenerationResult:
    """
    Résultat d'une génération du MVT.

    Attributes:
        text: Texte généré
        trajectory: Trajectoire complète (T, N)
        action: Valeur de l'action S
        kinetic_energy: Énergie cinétique totale
        potential_energy: Énergie potentielle totale
        has_singularity: Une singularité a-t-elle été détectée ?
        singularity_step: Pas où la singularité a été détectée (-1 si aucune)
        syntopy_score: Score de syntopie (0-1, 0 si pas d'exemple)
        plasticity_stats: Statistiques de plasticité
        generation_time: Temps de génération en secondes
        num_steps: Nombre de pas d'intégration
        metrics: Métriques détaillées du tenseur G
    """
    text: str
    trajectory: np.ndarray
    action: float = 0.0
    kinetic_energy: float = 0.0
    potential_energy: float = 0.0
    has_singularity: bool = False
    singularity_step: int = -1
    syntopy_score: float = 0.0
    plasticity_stats: Dict[str, Any] = None
    generation_time: float = 0.0
    num_steps: int = 0
    metrics: Dict[str, float] = None

    def __post_init__(self):
        if self.plasticity_stats is None:
            self.plasticity_stats = {}
        if self.metrics is None:
            self.metrics = {}


class MVT:
    """
    Modèle à Variance Topologique — Orchestrateur Principal.

    Le MVT remplace l'architecture transformer par un système de
    mécanique géométrique appliquée à la pensée. Le langage n'est
    plus découpé en tokens mais modélisé comme un champ vectoriel
    continu dans un espace à N dimensions, gouverné par un lagrangien
    sémantique.

    Pipeline de génération:
        1. Encodage du prompt → champ de force F_0
        2. (Optionnel) Application de la syntopie ★ pour le one-shot
        3. Intégration des équations d'Euler-Lagrange par RK4
        4. Détection de singularités
        5. Projection de la trajectoire → texte
        6. Mise à jour de la plasticité topologique (auto-évolution)
    """

    def __init__(self, config: Optional[MVTConfig] = None):
        if config is None:
            config = MVTConfig()

        self.config = config

        # 1. Tenseur métrique G(t)
        self.metric = MetricTensor(config)

        # 2. Encodeur : texte → champ de force
        self.encoder = InputEncoder(config, self.metric)

        # 3. Lagrangien sémantique L = T - V
        self.lagrangian = SemanticLagrangian(config, self.metric)

        # 4. Intégrateur Euler-Lagrange (RK4)
        self.integrator = LagrangianIntegrator(config, self.lagrangian)

        # 5. Couche de syntopie (one-shot)
        self.syntopy = SyntopicLayer(config, self.metric, self.encoder)

        # 6. Moteur de plasticité topologique
        self.plasticity = TopologicalPlasticityEngine(config, self.metric)

        # 7. Projecteur : variété → texte
        self.projector = ManifoldProjector(config, self.metric, self.encoder)

        # Historique des générations
        self._generation_history: List[GenerationResult] = []

    def set_example(self, example_text: str):
        """
        Définit un exemple pour le one-shot (syntopie).

        L'opérateur ★ fusionne instantanément la topologie de
        l'exemple avec la requête, sans fine-tuning ni gradient.

        Args:
            example_text: Texte de l'exemple
        """
        self.syntopy.set_example(example_text)

    def clear_example(self):
        """Supprime l'exemple de syntopie courant."""
        self.syntopy.clear_example()

    def generate(
        self,
        prompt: str,
        num_steps: Optional[int] = None,
        enable_plasticity: bool = True,
        return_trajectory: bool = False,
    ) -> GenerationResult:
        """
        Génère du texte à partir d'un prompt.

        Pipeline complet :
        1. Encodage du prompt en champ de force
        2. Application de la syntopie (si exemple défini)
        3. Intégration Euler-Lagrange (RK4)
        4. Détection de singularités
        5. Projection en texte
        6. Mise à jour plastique de G(t)

        Args:
            prompt: Texte du prompt
            num_steps: Nombre de pas d'intégration (None = config)
            enable_plasticity: Activer la mise à jour de G(t)
            return_trajectory: Inclure la trajectoire dans le résultat

        Returns:
            GenerationResult avec le texte et les métriques
        """
        start_time = time.time()

        # === ÉTAPE 1 : Encodage du prompt ===
        field, q0 = self.encoder.encode_prompt(prompt)

        # === ÉTAPE 2 : Syntopie (one-shot) ===
        field, tau = self.syntopy.apply_syntopy(field, prompt)

        # Configurer le lagrangien avec le champ de force externe
        self.lagrangian.external_force = field.evaluate

        # === ÉTAPE 3 : Intégration Euler-Lagrange (RK4) ===
        trajectory = self.integrator.integrate(q0, dq0=None, num_steps=num_steps)

        # Calculer l'action
        action = self.integrator.compute_action(trajectory)

        # Énergies
        T_total = sum(
            self.lagrangian.kinetic_energy(
                trajectory[i],
                (trajectory[min(i+1, len(trajectory)-1)] - trajectory[i]) / self.config.dt
            )
            for i in range(len(trajectory))
        )
        V_total = sum(
            self.lagrangian.potential_energy(trajectory[i], i * self.config.dt)
            for i in range(len(trajectory))
        )

        # === ÉTAPE 4 : Détection de singularités ===
        has_singularity, singularity_step = self.plasticity.detect_singularity(trajectory)

        # === ÉTAPE 5 : Projection en texte ===
        if has_singularity:
            text = self.projector.singularity_declaration(trajectory[singularity_step])
            success = False
        else:
            text = self.projector.project_trajectory(trajectory)
            success = True

        # === ÉTAPE 6 : Plasticité topologique ===
        if enable_plasticity:
            self.plasticity.update_metric(trajectory, success=success)

        # Score de syntopie
        syntopy_score = self.syntopy.compute_syntopy_score(trajectory)

        # Temps de génération
        gen_time = time.time() - start_time

        # Métriques du tenseur
        metrics = {
            "metric_det": float(np.linalg.det(self.metric.G)),
            "metric_trace": float(np.trace(self.metric.G)),
            "metric_condition": float(np.linalg.cond(self.metric.G)),
            "metric_eigenvalues": np.linalg.eigvalsh(self.metric.G).tolist(),
        }

        # Construire le résultat
        result = GenerationResult(
            text=text,
            trajectory=trajectory if return_trajectory else trajectory[:0],
            action=action,
            kinetic_energy=T_total,
            potential_energy=V_total,
            has_singularity=has_singularity,
            singularity_step=singularity_step,
            syntopy_score=syntopy_score,
            plasticity_stats=self.plasticity.get_plasticity_stats(),
            generation_time=gen_time,
            num_steps=len(trajectory),
            metrics=metrics,
        )

        self._generation_history.append(result)

        return result

    def generate_with_example(
        self,
        example_text: str,
        prompt: str,
        **kwargs,
    ) -> GenerationResult:
        """
        Génère avec un exemple (one-shot) en un seul appel.

        Args:
            example_text: Texte de l'exemple
            prompt: Texte de la requête
            **kwargs: Arguments passés à generate()

        Returns:
            GenerationResult
        """
        self.set_example(example_text)
        return self.generate(prompt, **kwargs)

    def batch_generate(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[GenerationResult]:
        """
        Génère du texte pour plusieurs prompts.

        Args:
            prompts: Liste de prompts
            **kwargs: Arguments passés à generate()

        Returns:
            Liste de GenerationResult
        """
        return [self.generate(prompt, **kwargs) for prompt in prompts]

    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques globales du modèle.

        Returns:
            Dict avec toutes les métriques
        """
        return {
            "config": {
                "ambient_dim": self.config.ambient_dim,
                "intrinsic_dim": self.config.intrinsic_dim,
                "dt": self.config.dt,
                "num_rk4_steps": self.config.num_rk4_steps,
            },
            "metric": {
                "det": float(np.linalg.det(self.metric.G)),
                "trace": float(np.trace(self.metric.G)),
                "condition": float(np.linalg.cond(self.metric.G)),
                "history_len": len(self.metric._history),
            },
            "plasticity": self.plasticity.get_plasticity_stats(),
            "syntopy": {
                "has_example": self.syntopy._example_field is not None,
                "strength": self.syntopy.syntopy_strength,
            },
            "encoder": {
                "vocab_size": len(self.encoder._word_cache),
                "total_words": self.encoder._total_words,
            },
            "generation": {
                "total_generations": len(self._generation_history),
                "total_singularity": sum(
                    1 for r in self._generation_history if r.has_singularity
                ),
            },
        }

    def reset(self):
        """
        Réinitialise le modèle à son état initial.

        Conserve la configuration mais réinitialise tous les
        composants (métrique, plasticité, historique).
        """
        self.metric = MetricTensor(self.config)
        self.encoder = InputEncoder(self.config, self.metric)
        self.lagrangian = SemanticLagrangian(self.config, self.metric)
        self.integrator = LagrangianIntegrator(self.config, self.lagrangian)
        self.syntopy = SyntopicLayer(self.config, self.metric, self.encoder)
        self.plasticity = TopologicalPlasticityEngine(self.config, self.metric)
        self.projector = ManifoldProjector(self.config, self.metric, self.encoder)
        self._generation_history = []

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"MVT(\n"
            f"  dim={stats['config']['ambient_dim']},\n"
            f"  generations={stats['generation']['total_generations']},\n"
            f"  singularities={stats['generation']['total_singularity']},\n"
            f"  erosions={stats['plasticity']['total_erosions']},\n"
            f"  sedimentations={stats['plasticity']['total_sedimentations']},\n"
            f"  barriers={stats['plasticity']['total_barriers']},\n"
            f"  channels={stats['plasticity']['total_channels']},\n"
            f")"
        )
