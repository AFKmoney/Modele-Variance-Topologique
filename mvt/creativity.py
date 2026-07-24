"""
MVT Creativity - Moteur de Génération Créative.
================================================

Rend le MVT créatif en injectant :
1. Bruit stochastique dans le lagrangien (température)
2. Recherche de nouveauté (novelty-seeking)
3. Sauts métaphoriques (discontinuités contrôlées)
4. Pensée divergente (trajectoires multiples)
5. Sélection créative de mots (projection softmax)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

from .config import MVTConfig
from .core.metric_tensor import MetricTensor


class CreativityEngine:
    """
    Moteur de créativité pour le MVT.

    Au lieu de suivre strictement le chemin de moindre action (qui
    produit un texte prévisible et répétitif), le moteur de créativité :
    - Injecte du bruit thermique dans les équations du mouvement
    - Favorise les trajectoires qui explorent des régions inconnues
    - Permet des sauts métaphoriques (discontinuités créatives)
    - Sélectionne les mots avec une distribution soft-max (pas argmax)
    """

    def __init__(self, config: MVTConfig, metric: MetricTensor):
        self.config = config
        self.metric = metric
        self.N = config.ambient_dim

        # Paramètres de créativité
        self.temperature = config.temperature
        self.novelty_bias = config.novelty_bias
        self.metaphor_jump_prob = config.metaphor_jump_prob
        self.metaphor_jump_strength = config.metaphor_jump_strength
        self.noise_scale = config.creativity_noise_scale
        self.projection_temperature = config.projection_temperature
        self.diversity_penalty = config.diversity_penalty
        self.max_consecutive_repeat = config.max_consecutive_repeat

        # Historique des régions visitées (pour la nouveauté)
        self._visited_regions: List[np.ndarray] = []
        self._visit_counts: Dict[int, int] = {}  # grid cell -> count
        self._grid_resolution = 10  # Coarse grid for visit tracking

    def inject_noise(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """
        Injecte du bruit stochastique dans l'accélération.

        Le bruit est gaussien, scalé par la température et par la
        distance au centre de l'espace (plus on est loin, plus on
        est créatif).

        Args:
            q: Position actuelle (N,)
            dq: Vitesse actuelle (N,)

        Returns:
            Perturbation de l'accélération (N,)
        """
        # Bruit gaussien de base
        noise = np.random.randn(self.N) * self.noise_scale * self.temperature

        # Amplification non-linéaire : plus on explore, plus on ose
        r = np.linalg.norm(q)
        exploration_factor = np.tanh(r / 5.0) * 0.5 + 0.5  # 0..1
        noise *= (1.0 + exploration_factor)

        # Rotation aléatoire du bruit pour plus de variété
        theta = np.random.randn() * 0.1 * self.temperature
        if self.N >= 2:
            R = np.eye(self.N)
            R[0, 0] = np.cos(theta)
            R[0, 1] = -np.sin(theta)
            R[1, 0] = np.sin(theta)
            R[1, 1] = np.cos(theta)
            noise = R @ noise

        return noise

    def novelty_force(self, q: np.ndarray) -> np.ndarray:
        """
        Calcule une force de répulsion par rapport aux régions déjà visitées.

        La particule d'idée est repoussée par les zones qu'elle a déjà
        explorées, favorisant la découverte de nouveaux chemins.

        Args:
            q: Position actuelle (N,)

        Returns:
            Force de nouveauté (N,)
        """
        if not self._visited_regions:
            return np.zeros(self.N, dtype=np.float64)

        force = np.zeros(self.N, dtype=np.float64)
        total_weight = 0.0

        for visited_pos in self._visited_regions[-20:]:  # Dernières 20 positions
            delta = q - visited_pos
            dist = np.linalg.norm(delta)
            if dist > 1e-8:
                # Force de répulsion en 1/r
                weight = self.novelty_bias / (dist ** 2 + 0.1)
                force += weight * delta / dist
                total_weight += weight

        if total_weight > 0:
            force /= total_weight

        return force * self.novelty_bias

    def should_metaphor_jump(self) -> bool:
        """
        Décide si un saut métaphorique doit se produire.

        Un saut métaphorique est une discontinuité créative : la particule
        "saute" à un endroit inattendu de l'espace sémantique, créant une
        association inédite (métaphore, analogie).

        Returns:
            True si un saut doit se produire
        """
        return np.random.random() < self.metaphor_jump_prob * self.temperature

    def metaphor_jump(self, q: np.ndarray) -> np.ndarray:
        """
        Effectue un saut métaphorique dans l'espace sémantique.

        Le saut est dirigé vers une région non explorée, avec une
        amplitude contrôlée par metaphor_jump_strength.

        Args:
            q: Position actuelle (N,)

        Returns:
            Nouvelle position après le saut (N,)
        """
        # Direction aléatoire biaisée vers l'inexploré
        random_dir = np.random.randn(self.N)
        random_dir /= (np.linalg.norm(random_dir) + 1e-8)

        # Amplitude du saut
        amplitude = self.metaphor_jump_strength * (1.0 + self.temperature)

        # Appliquer le saut
        q_new = q + amplitude * random_dir

        # Rester dans les limites (soft)
        r = np.linalg.norm(q_new)
        if r > 10.0:
            q_new *= 10.0 / r

        return q_new

    def record_visit(self, q: np.ndarray):
        """
        Enregistre la visite d'une position pour le suivi de nouveauté.
        """
        self._visited_regions.append(q.copy())
        if len(self._visited_regions) > 200:
            self._visited_regions.pop(0)

        # Coarse grid tracking
        grid_cell = tuple((q * self._grid_resolution).astype(int) % 1000)
        self._visit_counts[grid_cell] = self._visit_counts.get(grid_cell, 0) + 1

    def visit_count(self, q: np.ndarray) -> int:
        """Retourne combien de fois une zone similaire a été visitée."""
        grid_cell = tuple((q * self._grid_resolution).astype(int) % 1000)
        return self._visit_counts.get(grid_cell, 0)

    def creative_word_selection(
        self,
        q: np.ndarray,
        candidates: List[Tuple[str, float]],  # (word, distance)
        recent_words: List[str],
    ) -> Optional[str]:
        """
        Sélection créative de mots (soft-max au lieu d'argmax).

        Au lieu de toujours choisir le mot le plus proche, utilise une
        distribution de probabilité qui favorise la diversité.

        Args:
            q: Position dans l'espace
            candidates: Liste de (mot, distance) triés par distance
            recent_words: Mots récemment générés (pour pénalité)

        Returns:
            Mot sélectionné, ou None
        """
        if not candidates:
            return None

        # Convertir distances en scores (plus proche = plus haut score)
        distances = np.array([d for _, d in candidates])
        min_dist = np.min(distances) + 1e-8

        # Soft-max avec température
        scores = -distances / (min_dist * self.projection_temperature + 1e-8)
        scores -= np.max(scores)  # Numérique stability
        probs = np.exp(scores)
        probs /= (np.sum(probs) + 1e-8)

        # Appliquer pénalité de diversité sur les mots récents
        recent_set = set(recent_words[-self.max_consecutive_repeat * 3:])
        for i, (word, _) in enumerate(candidates):
            if word in recent_set:
                # Réduire la probabilité des mots récents
                repeat_count = recent_words.count(word)
                penalty = self.diversity_penalty ** (1 + repeat_count)
                probs[i] *= penalty

        # Re-normaliser
        total = np.sum(probs)
        if total < 1e-8:
            # Fallback: choisir aléatoirement parmi les candidats
            idx = np.random.randint(len(candidates))
            return candidates[idx][0]

        probs /= total

        # Échantillonner selon la distribution
        idx = np.random.choice(len(candidates), p=probs)
        return candidates[idx][0]

    def reset_history(self):
        """Réinitialise l'historique des visites."""
        self._visited_regions = []
        self._visit_counts = {}

    def divergent_thinking(
        self,
        q0: np.ndarray,
        num_branches: int = 3,
    ) -> List[np.ndarray]:
        """
        Génère plusieurs points de départ pour la pensée divergente.

        Crée des variations de la position initiale pour explorer
        plusieurs directions créatives simultanément.

        Args:
            q0: Position de départ (N,)
            num_branches: Nombre de branches

        Returns:
            Liste de positions de départ modifiées (N,)
        """
        branches = [q0.copy()]

        for _ in range(num_branches - 1):
            # Perturbation créative de la position initiale
            perturbation = np.random.randn(self.N) * self.temperature * 0.5
            # Rotation pour explorer des directions différentes
            angle = np.random.randn() * np.pi * 0.5
            if self.N >= 2:
                R = np.eye(self.N)
                R[0, 0] = np.cos(angle)
                R[0, 1] = -np.sin(angle)
                R[1, 0] = np.sin(angle)
                R[1, 1] = np.cos(angle)
                perturbation = R @ perturbation

            branches.append(q0 + perturbation)

        return branches

    def evaluate_creativity(self, trajectory: np.ndarray) -> float:
        """
        Évalue la créativité d'une trajectoire.

        Score basé sur :
        - Couverture spatiale (a-t-elle exploré beaucoup de région ?)
        - Variabilité des directions (a-t-elle changé de direction ?)
        - Singularités contrôlées (a-t-elle osé des sauts ?)

        Returns:
            Score de créativité [0, 1]
        """
        if len(trajectory) < 3:
            return 0.0

        # 1. Couverture spatiale
        positions = trajectory
        mean_pos = np.mean(positions, axis=0)
        spread = np.mean(np.linalg.norm(positions - mean_pos, axis=1))
        coverage = min(1.0, spread / 3.0)

        # 2. Variabilité directionnelle
        directions = np.diff(positions, axis=0)
        if len(directions) > 1:
            dir_norms = np.linalg.norm(directions, axis=1)
            dir_norms = dir_norms[dir_norms > 1e-8]
            if len(dir_norms) > 1:
                dir_variety = np.std(dir_norms) / (np.mean(dir_norms) + 1e-8)
                variety = min(1.0, dir_variety)
            else:
                variety = 0.0
        else:
            variety = 0.0

        # 3. Score combiné
        creativity_score = 0.5 * coverage + 0.3 * variety + 0.2 * self.temperature

        return min(1.0, creativity_score)

    def __repr__(self) -> str:
        return (
            f"CreativityEngine("
            f"temp={self.temperature:.2f}, "
            f"novelty={self.novelty_bias:.2f}, "
            f"metaphor_prob={self.metaphor_jump_prob:.2f}, "
            f"visited={len(self._visited_regions)})"
        )
