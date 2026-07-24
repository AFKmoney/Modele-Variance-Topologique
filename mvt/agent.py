"""
MVT Agent - Framework Agentique Autonome.
==========================================

Rend le MVT agentique par nature :
1. Boucle d'agent : observation → pensée → action → réflexion
2. Buts dirigés : l'agent vise une cible sémantique
3. Auto-réflexion : évalue et critique sa propre production
4. Replanification : adapte sa stratégie si insatisfait
5. Pensée en chaîne : décompose les problèmes en sous-buts
6. Mémoire épisodique : se souvient des trajectoires passées
"""

from __future__ import annotations

import time
import numpy as np
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from .config import MVTConfig
from .model import MVT, GenerationResult


class AgentState(Enum):
    """États de l'agent."""
    IDLE = "idle"
    OBSERVING = "observing"
    PLANNING = "planning"
    GENERATING = "generating"
    REFLECTING = "reflecting"
    REPLANNING = "replanning"
    FINISHED = "finished"


@dataclass
class Goal:
    """
    But sémantique de l'agent.
    Un but est une cible dans l'espace + des critères de succès.
    """
    description: str
    target_position: np.ndarray  # Cible dans R^N
    success_threshold: float = 0.5  # Distance max pour succès
    priority: float = 1.0
    achieved: bool = False
    attempts: int = 0

    def distance_to(self, q: np.ndarray, metric=None) -> float:
        """Distance au but."""
        delta = self.target_position - q
        if metric is not None:
            return float(np.sqrt(max(0, delta @ metric.G @ delta)))
        return float(np.linalg.norm(delta))


@dataclass
class ReflectionResult:
    """
    Résultat de l'auto-réflexion de l'agent.
    """
    satisfaction_score: float  # 0-1, à quel point l'agent est satisfait
    creativity_score: float  # 0-1, à quel point la sortie est créative
    coherence_score: float  # 0-1, à quel point c'est cohérent
    should_retry: bool  # Faut-il réessayer ?
    feedback: str  # Rétroaction textuelle
    new_goals: List[Goal] = field(default_factory=list)


@dataclass
class AgentStep:
    """Une étape de la trace de l'agent."""
    iteration: int
    state: AgentState
    prompt: str
    result: Optional[GenerationResult]
    reflection: Optional[ReflectionResult]
    goals: List[Goal]
    time_elapsed: float


class MVSAgent:
    """
    Agent Autonome basé sur le MVT.

    L'agent ne se contente pas de générer du texte : il
    OBSERVE son environnement, PLANIFIE des sous-buts,
    GÉNÈRE du texte, RÉFLÉCHIT sur sa production, et
    REPLANIFIE si nécessaire.

    Caractéristiques agentiques :
    - Autonome : prend des décisions sans supervision humaine
    - Créatif : explored'autres chemins quand insatisfait
    - Persistant : réessaie avec différentes stratégies
    - Adaptatif : modifie sa métrique interne en fonction des résultats
    """

    def __init__(self, config: Optional[MVTConfig] = None):
        if config is None:
            config = MVTConfig()

        self.config = config

        # Le MVT sous-jacent
        self.mvt = MVT(config)

        # État de l'agent
        self.state = AgentState.IDLE
        self._iteration = 0
        self._trace: List[AgentStep] = []

        # Buts courants
        self.goals: List[Goal] = []

        # Mémoire épisodique
        self._episodic_memory: List[Dict[str, Any]] = []

        # Compteurs agentiques
        self._total_reflections = 0
        self._total_replans = 0
        self._total_goals_achieved = 0

    # ================================================================
    #  BICYCLE D'AGENT : Observe → Plan → Generate → Reflect → Loop
    # ================================================================

    def observe(self, prompt: str) -> Dict[str, Any]:
        """
        Phase d'observation : analyse le prompt et l'environnement.

        Encode le prompt, analyse sa complexité, identifie les concepts clés.
        """
        self.state = AgentState.OBSERVING

        # Encoder le prompt
        field, q0 = self.mvt.encoder.encode_prompt(prompt)

        # Analyser la complexité
        words = prompt.split()
        complexity = min(1.0, len(words) / 20.0)

        # Identifier les concepts clés
        key_concepts = [
            w for w in words
            if self.mvt.encoder._word_importance(w) > 0.3
        ]

        return {
            "center": q0,
            "complexity": complexity,
            "num_words": len(words),
            "key_concepts": key_concepts,
            "control_points": len(field.control_points),
        }

    def plan(self, prompt: str) -> List[Goal]:
        """
        Phase de planification : décompose la tâche en sous-buts.

        Pour les tâches simples : un seul but.
        Pour les tâches complexes : plusieurs sous-buts enchaînés.
        """
        self.state = AgentState.PLANNING

        observation = self.observe(prompt)
        q0 = observation["center"]
        complexity = observation["complexity"]

        # Définir le but principal : générer à partir du prompt
        main_goal = Goal(
            description=prompt,
            target_position=q0 + np.random.randn(self.config.ambient_dim) * 0.1,
            success_threshold=0.5 + (1.0 - complexity) * 0.3,
            priority=1.0,
        )

        goals = [main_goal]

        # Si complexe, ajouter des sous-buts de diversité
        if complexity > 0.5:
            # Sous-but : explorer une direction créative
            creative_dir = np.random.randn(self.config.ambient_dim)
            creative_dir /= (np.linalg.norm(creative_dir) + 1e-8)
            creative_goal = Goal(
                description="Explorer une perspective créative",
                target_position=q0 + creative_dir * self.config.novelty_bias,
                success_threshold=0.7,
                priority=0.8,
            )
            goals.append(creative_goal)

        self.goals = goals
        return goals

    def generate_step(self, prompt: str, goal: Optional[Goal] = None) -> GenerationResult:
        """
        Phase de génération : génère du texte en suivant le but.

        Si un but est spécifié, steer la génération vers la cible.
        """
        self.state = AgentState.GENERATING

        # Encoder le prompt
        field, q0 = self.mvt.encoder.encode_prompt(prompt)

        # Si on a un but, ajouter un attracteur vers la cible
        if goal is not None:
            target = goal.target_position
            # Ajouter la cible comme source dans le champ de force
            field.add_source(target, strength=self.config.goal_steering_strength)
            goal.attempts += 1

        # Configurer la créativité
        creativity = self.mvt.creativity if hasattr(self.mvt, 'creativity') else None

        # Appliquer la syntopie si un exemple existe
        field, tau = self.mvt.syntopy.apply_syntopy(field, prompt)

        # Configurer le lagrangien
        self.mvt.lagrangian.external_force = field.evaluate

        # Intégrer avec créativité
        from .lagrangian.integrator import LagrangianIntegrator
        integrator = LagrangianIntegrator(self.config, self.mvt.lagrangian)

        # Intégration avec callbacks créatifs
        trajectory = self._creative_integrate(integrator, q0, goal)

        # Calculer les métriques
        action = integrator.compute_action(trajectory)

        # Énergies
        T_total = sum(
            self.mvt.lagrangian.kinetic_energy(
                trajectory[i],
                (trajectory[min(i+1, len(trajectory)-1)] - trajectory[i]) / self.config.dt
            )
            for i in range(len(trajectory))
        )

        # Projection créative
        if creativity is not None:
            text = self._creative_project(trajectory, creativity)
            creativity.record_visit(trajectory[-1])
        else:
            text = self.mvt.projector.project_trajectory(trajectory)

        # Score de syntopie
        syntopy_score = self.mvt.syntopy.compute_syntopy_score(trajectory)

        result = GenerationResult(
            text=text,
            trajectory=trajectory,
            action=action,
            kinetic_energy=T_total,
            syntopy_score=syntopy_score,
            num_steps=len(trajectory),
        )

        return result

    def _creative_integrate(
        self,
        integrator,
        q0: np.ndarray,
        goal: Optional[Goal],
    ) -> np.ndarray:
        """
        Intégration RK4 avec injection créative.
        Injecte du bruit, des sauts métaphoriques, et la force de nouveauté.
        """
        creativity = self.mvt.creativity if hasattr(self.mvt, 'creativity') else None

        num_steps = self.config.num_rk4_steps
        N = self.config.ambient_dim
        dt = self.config.dt

        trajectory = np.zeros((num_steps + 1, N), dtype=np.float64)
        trajectory[0] = q0.copy()

        q = q0.copy()
        dq = np.random.randn(N) * 0.01 * self.config.temperature
        t = 0.0

        for step in range(num_steps):
            # Pas RK4 normal
            q_new, dq_new = integrator.step_rk4(q, dq, t, dt)

            # === INJECTION CRÉATIVE ===
            if creativity is not None:
                # Bruit stochastique
                noise = creativity.inject_noise(q, dq)
                q_new += noise * dt

                # Force de nouveauté
                novelty = creativity.novelty_force(q_new)
                q_new += novelty * dt

                # Saut métaphorique
                if creativity.should_metaphor_jump():
                    q_new = creativity.metaphor_jump(q_new)

                creativity.record_visit(q_new)

            # But-directed steering
            if goal is not None:
                direction = goal.target_position - q_new
                dist = np.linalg.norm(direction)
                if dist > 1e-8:
                    steer = direction / dist * self.config.goal_steering_strength * 0.01
                    q_new += steer

            # Vérification de divergence (relâchée)
            if np.linalg.norm(q_new) > self.config.divergence_threshold:
                trajectory = trajectory[:step + 1]
                break

            q = q_new
            dq = dq_new
            t += dt
            trajectory[step + 1] = q.copy()

        return trajectory

    def _creative_project(self, trajectory: np.ndarray, creativity) -> str:
        """
        Projection créative : sélection de mots avec soft-max.

        Utilise le moteur de créativité pour sélectionner des mots
        diversifiés au lieu du mot le plus proche.
        """
        self.mvt.projector._build_inverse_vocab()
        self.mvt.projector._extend_vocabulary()
        self.mvt.projector._sample_rate = self.config.sample_rate

        T_len = len(trajectory)
        if T_len == 0:
            return ""

        indices = list(range(0, T_len, self.config.sample_rate))
        sampled = trajectory[indices]

        words: List[str] = []
        recent_words: List[str] = []

        for q in sampled:
            # Collecter tous les candidats avec leurs distances
            candidates = []
            for word, pos in self.mvt.encoder._word_cache.items():
                dist = self.mvt.metric.distance(q, pos)
                if dist < 8.0:  # Seuil relâché
                    candidates.append((word, dist))

            for word, pos in self.mvt.projector._extended_vocab:
                dist = self.mvt.metric.distance(q, pos)
                if dist < 8.0:
                    candidates.append((word, dist))

            if not candidates:
                continue

            # Sélection créative
            selected = creativity.creative_word_selection(q, candidates, recent_words)

            if selected:
                # Vérifier les répétitions consécutives
                consecutive = sum(
                    1 for w in reversed(recent_words)
                    if w == selected
                )
                if consecutive < self.config.max_consecutive_repeat:
                    words.append(selected)
                    recent_words.append(selected)
                # Si trop de répétitions, on saute (silence créatif)

            if len(words) >= self.mvt.projector._max_words:
                break

        return " ".join(words)

    def reflect(self, result: GenerationResult, prompt: str) -> ReflectionResult:
        """
        Phase de réflexion : l'agent évalue sa propre production.

        L'agent se pose des questions :
        - Le texte est-il satisfaisant ?
        - Est-il assez créatif ?
        - Devrais-je réessayer avec une autre stratégie ?

        Returns:
            ReflectionResult avec les scores et la décision
        """
        self.state = AgentState.REFLECTING
        self._total_reflections += 1

        text = result.text
        words = text.split()

        # 1. Score de satisfaction basé sur la longueur et la diversité
        unique_words = set(words)
        diversity = len(unique_words) / max(1, len(words))

        # Score de longueur (on veut au moins quelques mots)
        length_score = min(1.0, len(words) / 10.0)

        # Score de diversité lexicale
        diversity_score = diversity

        # Score de syntopie (si one-shot)
        syntopy_score = result.syntopy_score

        # Satisfaction combinée
        satisfaction = (
            0.3 * length_score +
            0.3 * diversity_score +
            0.2 * syntopy_score +
            0.2 * (1.0 if len(words) > 3 else 0.0)
        )

        # 2. Score de créativité
        if hasattr(self.mvt, 'creativity'):
            creativity_score = self.mvt.creativity.evaluate_creativity(
                result.trajectory
            ) if len(result.trajectory) > 0 else 0.0
        else:
            creativity_score = diversity * 0.8

        # 3. Score de cohérence (basé sur l'action - plus l'action est basse, plus c'est fluide)
        coherence = max(0, 1.0 - result.action / (abs(result.action) + 1.0) * 0.01)

        # 4. Décision : faut-il réessayer ?
        should_retry = (
            satisfaction < self.config.self_reflection_threshold
            and self._iteration < self.config.max_agent_iterations
        )

        # 5. Feedback textuel
        if satisfaction > 0.7:
            feedback = "Production satisfaisante. Bon équilibre créativité/coherence."
        elif satisfaction > 0.4:
            feedback = "Production acceptable mais pourrait etre plus riche. Exploration supplementaire recommandee."
        else:
            feedback = "Production insuffisante. Replanification necessaire."

        if diversity < 0.3 and len(words) > 5:
            feedback += " Trop de repetition lexicale — augmenter la temperature."

        # 6. Nouveaux sous-buts si replanification
        new_goals = []
        if should_retry:
            # Augmenter la créativité pour le prochain essai
            _, q0 = self.mvt.encoder.encode_prompt(prompt)
            creative_perturbation = np.random.randn(self.config.ambient_dim) * 0.5
            new_goal = Goal(
                description=f"Essai {self._iteration + 2} avec plus de diversite",
                target_position=q0 + creative_perturbation,
                success_threshold=self.config.self_reflection_threshold,
                priority=1.0,
            )
            new_goals.append(new_goal)

        reflection = ReflectionResult(
            satisfaction_score=satisfaction,
            creativity_score=creativity_score,
            coherence_score=coherence,
            should_retry=should_retry,
            feedback=feedback,
            new_goals=new_goals,
        )

        return reflection

    def replan(self, prompt: str, reflection: ReflectionResult) -> List[Goal]:
        """
        Phase de replanification : adapte la stratégie.

        Si l'agent est insatisfait, il modifie ses buts et
        ajuste ses paramètres créatifs.
        """
        self.state = AgentState.REPLANNING
        self._total_replans += 1

        # Augmenter la créativité pour la prochaine itération
        if hasattr(self.mvt, 'creativity'):
            self.mvt.creativity.temperature = min(
                2.0,
                self.mvt.creativity.temperature * 1.2
            )
            self.mvt.creativity.metaphor_jump_prob = min(
                0.5,
                self.mvt.creativity.metaphor_jump_prob * 1.3
            )

        # Mettre à jour les buts
        if reflection.new_goals:
            self.goals = reflection.new_goals
        else:
            _, q0 = self.mvt.encoder.encode_prompt(prompt)
            self.goals = [
                Goal(
                    description=f"Replanification #{self._iteration}",
                    target_position=q0 + np.random.randn(self.config.ambient_dim) * 0.3,
                    success_threshold=self.config.self_reflection_threshold,
                    priority=1.0,
                )
            ]

        return self.goals

    # ================================================================
    #  API PRINCIPALE
    # ================================================================

    def run(
        self,
        prompt: str,
        example: Optional[str] = None,
        verbose: bool = True,
    ) -> GenerationResult:
        """
        Lance l'agent de manière autonome sur un prompt.

        La boucle agentique :
        1. Observer le prompt
        2. Planifier des sous-buts
        3. Générer du texte (avec créativité)
        4. Réfléchir sur le résultat
        5. Si insatisfait → replanifier et recommencer
        6. Retourner le meilleur résultat

        Args:
            prompt: Texte du prompt
            example: Exemple pour le one-shot (optionnel)
            verbose: Afficher le raisonnement de l'agent

        Returns:
            Meilleur GenerationResult
        """
        start_time = time.time()
        self._iteration = 0

        if verbose:
            print(f"[Agent] Observation du prompt...")

        # Observation
        observation = self.observe(prompt)

        if verbose:
            print(f"[Agent] Complexite: {observation['complexity']:.2f}, "
                  f"Concepts cles: {observation['key_concepts']}")

        # Planification
        if verbose:
            print(f"[Agent] Planification de {len(self.goals) or 1} sous-but(s)...")

        # Définir l'exemple si fourni
        if example:
            self.mvt.set_example(example)

        # Boucle agentique principale
        best_result = None
        best_score = -1.0

        while self._iteration < self.config.max_agent_iterations:
            self._iteration += 1

            if verbose:
                print(f"\n[Agent] --- Iteration {self._iteration} ---")
                print(f"[Agent] Etat: {self.state.value}")

            # Planifier (ou replanifier)
            if self._iteration == 1:
                self.plan(prompt)
            else:
                pass  # Les buts ont été mis à jour par la réflexion

            if verbose:
                for i, g in enumerate(self.goals):
                    print(f"[Agent]   But {i+1}: {g.description} "
                          f"(priorite={g.priority:.1f})")

            # Générer pour chaque but
            for goal in self.goals:
                if verbose:
                    print(f"[Agent] Generation pour: {goal.description}")

                result = self.generate_step(prompt, goal)

                if verbose:
                    print(f"[Agent]   -> \"{result.text[:80]}...\""
                          if len(result.text) > 80
                          else f"[Agent]   -> \"{result.text}\"")

                # Réfléchir
                reflection = self.reflect(result, prompt)

                if verbose:
                    print(f"[Agent]   Satisfaction: {reflection.satisfaction_score:.2f} | "
                          f"Creativite: {reflection.creativity_score:.2f} | "
                          f"Coherence: {reflection.coherence_score:.2f}")
                    print(f"[Agent]   Feedback: {reflection.feedback}")

                # Mettre à jour le meilleur résultat
                combined_score = (
                    0.4 * reflection.satisfaction_score +
                    0.3 * reflection.creativity_score +
                    0.3 * reflection.coherence_score
                )

                if combined_score > best_score:
                    best_score = combined_score
                    best_result = result

                # Plasticité
                self.mvt.plasticity.update_metric(
                    result.trajectory,
                    success=reflection.satisfaction_score > 0.4,
                )

                # Décision : continuer ou arrêter ?
                if not reflection.should_retry:
                    if verbose:
                        print(f"[Agent] Satisfaction suffisante. Arret de la boucle.")
                    break

            # Vérifier si on a dépassé le max
            if self._iteration >= self.config.max_agent_iterations:
                if verbose:
                    print(f"[Agent] Maximum d'iterations atteint.")
                break

            # Replanifier pour la prochaine itération
            if self.state == AgentState.REFLECTING:
                if best_result:
                    last_reflection = self.reflect(best_result, prompt)
                    if last_reflection.should_retry:
                        self.replan(prompt, last_reflection)
                        if verbose:
                            print(f"[Agent] Replanification. Temperature={self.mvt.creativity.temperature:.2f}")

        # Finalisation
        self.state = AgentState.FINISHED
        total_time = time.time() - start_time

        if best_result:
            best_result.generation_time = total_time

        # Stocker en mémoire épisodique
        self._episodic_memory.append({
            "prompt": prompt,
            "result": best_result,
            "iterations": self._iteration,
            "final_score": best_score,
            "time": total_time,
        })

        if verbose:
            print(f"\n[Agent] Termine en {self._iteration} iteration(s), "
                  f"{total_time:.2f}s")
            print(f"[Agent] Score final: {best_score:.2f}")
            if best_result:
                print(f"[Agent] Texte final: \"{best_result.text}\"")

        return best_result if best_result else GenerationResult(
            text="", trajectory=np.zeros((1, self.config.ambient_dim))
        )

    def get_agent_stats(self) -> Dict[str, Any]:
        """Statistiques de l'agent."""
        return {
            "state": self.state.value,
            "total_iterations": self._iteration,
            "total_reflections": self._total_reflections,
            "total_replans": self._total_replans,
            "total_goals_achieved": self._total_goals_achieved,
            "episodic_memory_size": len(self._episodic_memory),
            "active_goals": len(self.goals),
            "creativity": {
                "temperature": self.mvt.creativity.temperature if hasattr(self.mvt, 'creativity') else 0,
                "novelty_bias": self.mvt.creativity.novelty_bias if hasattr(self.mvt, 'creativity') else 0,
            } if hasattr(self.mvt, 'creativity') else {},
        }

    def reset(self):
        """Réinitialise l'agent."""
        self.mvt.reset()
        self.state = AgentState.IDLE
        self._iteration = 0
        self.goals = []
        self._trace = []
        self._episodic_memory = []
        self._total_reflections = 0
        self._total_replans = 0
        self._total_goals_achieved = 0

    def __repr__(self) -> str:
        return (
            f"MVSAgent(state={self.state.value}, "
            f"iterations={self._iteration}, "
            f"goals={len(self.goals)}, "
            f"memory={len(self._episodic_memory)})"
        )
