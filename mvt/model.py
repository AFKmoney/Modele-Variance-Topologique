"""
MVT - Modèle à Variance Topologique (Orchestrateur Principal v2).
=================================================================

Version 2 avec :
- Moteur de créativité intégré (bruit, nouveauté, sauts métaphoriques)
- Anti-hallucination SOUPLE (singularités contrôlées, exploration libre)
- Intégration créative dans le pipeline
- API enrichie (temperature, creativity_mode)

Usage:
    model = MVT(MVTConfig(temperature=0.8))
    result = model.generate("Prompt...", temperature=0.9)
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
from .creativity import CreativityEngine


@dataclass
class GenerationResult:
    text: str
    trajectory: np.ndarray
    action: float = 0.0
    kinetic_energy: float = 0.0
    potential_energy: float = 0.0
    has_singularity: bool = False
    singularity_step: int = -1
    syntopy_score: float = 0.0
    creativity_score: float = 0.0
    diversity_score: float = 0.0
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
    MVT v2 — Orchestrateur avec créativité intégrée.
    """

    def __init__(self, config: Optional[MVTConfig] = None):
        if config is None:
            config = MVTConfig()

        self.config = config

        # Composants de base
        self.metric = MetricTensor(config)
        self.encoder = InputEncoder(config, self.metric)
        self.lagrangian = SemanticLagrangian(config, self.metric)
        self.integrator = LagrangianIntegrator(config, self.lagrangian)
        self.syntopy = SyntopicLayer(config, self.metric, self.encoder)
        self.plasticity = TopologicalPlasticityEngine(config, self.metric)
        self.projector = ManifoldProjector(config, self.metric, self.encoder)

        # NOUVEAU : Moteur de créativité
        self.creativity = CreativityEngine(config, self.metric)

        # Connecter la créativité au lagrangien
        self.lagrangian.novelty_force = self.creativity.novelty_force

        self._generation_history: List[GenerationResult] = []

    def set_example(self, example_text: str):
        self.syntopy.set_example(example_text)

    def clear_example(self):
        self.syntopy.clear_example()

    def set_temperature(self, temperature: float):
        """Ajuste la température de créativité à la volée."""
        self.creativity.temperature = np.clip(temperature, 0.0, 2.0)

    def generate(
        self,
        prompt: str,
        num_steps: Optional[int] = None,
        enable_plasticity: bool = True,
        return_trajectory: bool = False,
        temperature: Optional[float] = None,
    ) -> GenerationResult:
        """
        Génère du texte avec créativité intégrée.
        """
        start_time = time.time()

        # Ajuster température si spécifiée
        if temperature is not None:
            self.set_temperature(temperature)

        # === ÉTAPE 1 : Encodage ===
        field, q0 = self.encoder.encode_prompt(prompt)

        # === ÉTAPE 2 : Syntopie ===
        field, tau = self.syntopy.apply_syntopy(field, prompt)

        # Configurer le lagrangien
        self.lagrangian.external_force = field.evaluate

        # === ÉTAPE 3 : Intégration créative ===
        trajectory = self._creative_integrate(q0, num_steps)

        # Métriques
        action = self.integrator.compute_action(trajectory)

        T_total = sum(
            self.lagrangian.kinetic_energy(
                trajectory[i],
                (trajectory[min(i+1, len(trajectory)-1)] - trajectory[i]) / self.config.dt
            )
            for i in range(len(trajectory))
        )

        # === ÉTAPE 4 : Singularités CONTRÔLÉES (douces) ===
        has_singularity, singularity_step = self._soft_singularity_check(trajectory)

        # === ÉTAPE 5 : Projection créative ===
        if has_singularity and singularity_step > 0:
            # Singularité partielle : générer quand même, avec un marqueur
            text_before = self._creative_project(trajectory[:singularity_step])
            text_after = self._creative_project(trajectory[singularity_step:])
            text = f"{text_before} ... {text_after}"
            success = True  # Pas de blocage ! On continue quand même
        else:
            text = self._creative_project(trajectory)
            success = True

        # Scores de créativité
        creativity_score = self.creativity.evaluate_creativity(trajectory)
        words = text.split()
        diversity_score = len(set(words)) / max(1, len(words))

        # === ÉTAPE 6 : Plasticité (douce) ===
        if enable_plasticity:
            # L'érosion est douce : on n'érige des barrières que pour les
            # trajectoires vraiment pathologiques
            self.plasticity.update_metric(
                trajectory,
                success=len(words) > 1 and diversity_score > 0.1,
            )

        # Score de syntopie
        syntopy_score = self.syntopy.compute_syntopy_score(trajectory)

        gen_time = time.time() - start_time

        metrics = {
            "metric_det": float(np.linalg.det(self.metric.G)),
            "metric_trace": float(np.trace(self.metric.G)),
            "metric_condition": float(np.linalg.cond(self.metric.G)),
        }

        result = GenerationResult(
            text=text,
            trajectory=trajectory if return_trajectory else trajectory[:0],
            action=action,
            kinetic_energy=T_total,
            potential_energy=0.0,
            has_singularity=has_singularity,
            singularity_step=singularity_step,
            syntopy_score=syntopy_score,
            creativity_score=creativity_score,
            diversity_score=diversity_score,
            plasticity_stats=self.plasticity.get_plasticity_stats(),
            generation_time=gen_time,
            num_steps=len(trajectory),
            metrics=metrics,
        )

        self._generation_history.append(result)
        return result

    def _creative_integrate(
        self, q0: np.ndarray, num_steps: Optional[int] = None
    ) -> np.ndarray:
        """
        Intégration RK4 avec injection créative complète.
        Bruit + nouveauté + sauts métaphoriques.
        """
        if num_steps is None:
            num_steps = self.config.num_rk4_steps

        N = self.config.ambient_dim
        dt = self.config.dt
        trajectory = np.zeros((num_steps + 1, N), dtype=np.float64)
        trajectory[0] = q0.copy()

        q = q0.copy()
        dq = np.random.randn(N) * 0.01 * self.creativity.temperature
        t = 0.0

        for step in range(num_steps):
            # Pas RK4 standard
            q_new, dq_new = self.integrator.step_rk4(q, dq, t, dt)

            # === INJECTION CRÉATIVE ===

            # 1. Bruit stochastique (température)
            noise = self.creativity.inject_noise(q, dq)
            q_new += noise * dt

            # 2. Force de nouveauté (anti-stagnation)
            novelty = self.creativity.novelty_force(q_new)
            q_new += novelty * dt

            # 3. Saut métaphorique (discontinuité créative)
            if self.creativity.should_metaphor_jump():
                q_new = self.creativity.metaphor_jump(q_new)

            # Enregistrer la visite
            self.creativity.record_visit(q_new)

            # Vérification de divergence (TRÈS relâchée)
            if np.linalg.norm(q_new) > self.config.divergence_threshold:
                trajectory = trajectory[:step + 1]
                break

            q = q_new
            dq = dq_new
            t += dt
            trajectory[step + 1] = q.copy()

        return trajectory

    def _soft_singularity_check(self, trajectory: np.ndarray):
        """
        Vérification DOUCE des singularités.

        Au lieu d'arrêter net la génération, on signale la singularité
        mais on laisse la trajectoire continuer. L'agent peut ensuite
        décider quoi faire avec cette information.
        """
        threshold = self.config.curvature_threshold  # Déjà très élevé (100+)

        for t_idx in range(len(trajectory)):
            q = trajectory[t_idx]
            curvature = self.metric.scalar_curvature(q)

            if abs(curvature) > threshold:
                return True, t_idx

        return False, -1

    def _creative_project(self, trajectory: np.ndarray) -> str:
        """
        Projection créative avec soft-max et diversité.
        """
        self.projector._build_inverse_vocab()
        self.projector._extend_vocabulary()
        self.projector._sample_rate = self.config.sample_rate

        T_len = len(trajectory)
        if T_len == 0:
            return ""

        indices = list(range(0, T_len, self.config.sample_rate))
        sampled = trajectory[indices]

        words: List[str] = []
        recent_words: List[str] = []

        for q in sampled:
            # Collecter candidats
            candidates = []
            for word, pos in self.encoder._word_cache.items():
                dist = self.metric.distance(q, pos)
                if dist < 10.0:  # Seuil très large
                    candidates.append((word, dist))
            for word, pos in self.projector._extended_vocab:
                dist = self.metric.distance(q, pos)
                if dist < 10.0:
                    candidates.append((word, dist))

            if not candidates:
                continue

            # Sélection créative (soft-max, pas argmax)
            selected = self.creativity.creative_word_selection(
                q, candidates, recent_words
            )

            if selected:
                # Vérifier répétitions consécutives
                consecutive = sum(
                    1 for w in reversed(recent_words) if w == selected
                )
                if consecutive < self.config.max_consecutive_repeat:
                    words.append(selected)
                    recent_words.append(selected)

            if len(words) >= self.projector._max_words:
                break

        return " ".join(words)

    def generate_with_example(
        self, example_text: str, prompt: str, **kwargs
    ) -> GenerationResult:
        self.set_example(example_text)
        return self.generate(prompt, **kwargs)

    def batch_generate(self, prompts: List[str], **kwargs) -> List[GenerationResult]:
        return [self.generate(prompt, **kwargs) for prompt in prompts]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "config": {
                "ambient_dim": self.config.ambient_dim,
                "intrinsic_dim": self.config.intrinsic_dim,
                "dt": self.config.dt,
                "num_rk4_steps": self.config.num_rk4_steps,
                "temperature": self.creativity.temperature,
                "novelty_bias": self.creativity.novelty_bias,
            },
            "metric": {
                "det": float(np.linalg.det(self.metric.G)),
                "trace": float(np.trace(self.metric.G)),
                "condition": float(np.linalg.cond(self.metric.G)),
            },
            "creativity": {
                "temperature": self.creativity.temperature,
                "novelty_bias": self.creativity.novelty_bias,
                "metaphor_prob": self.creativity.metaphor_jump_prob,
                "visited_regions": len(self.creativity._visited_regions),
            },
            "plasticity": self.plasticity.get_plasticity_stats(),
            "syntopy": {
                "has_example": self.syntopy._example_field is not None,
                "strength": self.syntopy.syntopy_strength,
            },
            "generation": {
                "total_generations": len(self._generation_history),
            },
        }

    def reset(self):
        self.metric = MetricTensor(self.config)
        self.encoder = InputEncoder(self.config, self.metric)
        self.lagrangian = SemanticLagrangian(self.config, self.metric)
        self.integrator = LagrangianIntegrator(self.config, self.lagrangian)
        self.syntopy = SyntopicLayer(self.config, self.metric, self.encoder)
        self.plasticity = TopologicalPlasticityEngine(self.config, self.metric)
        self.projector = ManifoldProjector(self.config, self.metric, self.encoder)
        self.creativity = CreativityEngine(self.config, self.metric)
        self.lagrangian.novelty_force = self.creativity.novelty_force
        self._generation_history = []

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"MVT(\n"
            f"  dim={stats['config']['ambient_dim']},\n"
            f"  temp={stats['creativity']['temperature']:.2f},\n"
            f"  gens={stats['generation']['total_generations']},\n"
            f"  barriers={stats['plasticity']['total_barriers']},\n"
            f"  channels={stats['plasticity']['total_channels']},\n"
            f")"
        )
