"""
MVT Input Encoder - Texte brut vers champ de force sémantique.
================================================================

Transforme le texte d'entrée en un champ vectoriel continu dans R^N.
Chaque mot/concept est projeté comme un attracteur dans l'espace,
créant un "paysage énergétique" qui guidera la particule d'idée.
"""

from __future__ import annotations

import hashlib
import math
import numpy as np
from typing import Optional, Dict, List

from .config import MVTConfig
from .core.metric_tensor import MetricTensor
from .core.vector_field import VectorField


class InputEncoder:
    """
    Encodeur d'entrée : Texte → Champ de force dans R^N.

    L'encodeur transforme une séquence de texte en un ensemble de
    points de contrôle dans l'espace sémantique. Chaque token est
    mappé à une position via un hash déterministe, et l'ensemble
    forme un champ de force qui guide la génération.

    Caractéristiques:
    - Pas de tokenization statistique : chaque mot est un concept
      avec une position unique dans l'espace.
    - Pas de vocabulaire figé : tout mot peut être encodé.
    - Position déterministe : même mot → même position.
    - Pondération contextuelle : importance relative des concepts.
    """

    def __init__(self, config: MVTConfig, metric: MetricTensor):
        self.config = config
        self.metric = metric
        self.N = config.ambient_dim

        # Cache des embeddings de mots
        self._word_cache: Dict[str, np.ndarray] = {}

        # Paramètres d'encodage
        self._scale = 2.0  # Échelle spatiale des embeddings
        self._base_strength = 1.0

        # Fréquence des mots pour la pondération TF-IDF simplifiée
        self._word_freq: Dict[str, int] = {}
        self._total_words = 0

    def _hash_to_position(self, word: str) -> np.ndarray:
        """
        Convertit un mot en position dans R^N via un hash déterministe.

        Utilise SHA-256 pour générer une position stable et
        quasi-uniforme dans l'espace.

        Args:
            word: Mot à encoder

        Returns:
            Position dans R^N (N,)
        """
        if word in self._word_cache:
            return self._word_cache[word]

        # Hash déterministe
        h = hashlib.sha256(word.encode('utf-8')).hexdigest()

        # Convertir le hash en coordonnées
        N = self.N
        pos = np.zeros(N, dtype=np.float64)

        for i in range(N):
            # Prendre 4 hex chars = 16 bits par dimension
            start = (i * 4) % (len(h) - 3)
            val = int(h[start:start + 4], 16)
            # Normaliser dans [-scale, scale]
            pos[i] = (val / 65535.0) * 2.0 * self._scale - self._scale

        # Appliquer une non-linéarité pour plus de variété
        pos = self._scale * np.tanh(pos / self._scale)

        self._word_cache[word] = pos
        return pos

    def _word_importance(self, word: str) -> float:
        """
        Calcule l'importance d'un mot (simplification de TF-IDF).

        Les mots rares ont plus de poids (plus discriminants).
        Les mots fréquents (articles, prépositions) ont moins de poids.

        Returns:
            Poids entre 0.1 et 2.0
        """
        # Mots fonctionnels (faible importance)
        functional_words = {
            'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du',
            'et', 'ou', 'mais', 'donc', 'car', 'ni', 'que', 'qui',
            'est', 'a', 'a', 'en', 'dans', 'pour', 'par', 'sur',
            'avec', 'sans', 'ce', 'cette', 'ces', 'il', 'elle',
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'and',
            'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'from', 'it', 'this', 'that', 'i', 'you',
        }

        if word.lower() in functional_words:
            return 0.1

        # Inverse frequency weighting
        freq = self._word_freq.get(word, 1)
        if self._total_words > 0:
            idf = math.log(1 + self._total_words / (freq + 1))
        else:
            idf = 1.0

        return min(2.0, 0.3 + 0.5 * idf)

    def encode_text(
        self, text: str, field: Optional[VectorField] = None
    ) -> VectorField:
        """
        Encode un texte en champ de force sémantique.

        Chaque mot est projeté comme un attracteur dans l'espace.
        Les mots consécutifs sont reliés par des forces directionnelles.

        Args:
            text: Texte à encoder
            field: Champ existant à enrichir (None = nouveau champ)

        Returns:
            Champ vectoriel sémantique
        """
        if field is None:
            field = VectorField(self.config)

        # Tokenization simple (par mots)
        words = text.split()
        self._total_words += len(words)

        # Mise à jour des fréquences
        for w in words:
            self._word_freq[w] = self._word_freq.get(w, 0) + 1

        # Placer chaque mot comme attracteur
        positions = []
        for word in words:
            pos = self._hash_to_position(word)
            importance = self._word_importance(word)
            field.add_source(pos, strength=self._base_strength * importance)
            positions.append(pos)

        # Créer des liens séquentiels entre mots adjacents
        # (forces directionnelles qui guident le flux)
        if len(positions) > 1:
            for i in range(len(positions) - 1):
                midpoint = 0.5 * (positions[i] + positions[i + 1])
                link_strength = 0.3 * min(
                    self._word_importance(words[i]),
                    self._word_importance(words[i + 1]),
                )
                field.add_source(midpoint, strength=link_strength)

        return field

    def encode_prompt(
        self, prompt: str, field: Optional[VectorField] = None
    ) -> Tuple[VectorField, np.ndarray]:
        """
        Encode un prompt en champ de force + position de départ.

        La position de départ est le "centre de masse" sémantique
        du prompt, pondéré par l'importance des mots.

        Args:
            prompt: Texte du prompt

        Returns:
            (champ_de_force, position_initiale)
        """
        field = self.encode_text(prompt, field)

        # Calculer le centre de masse pondéré
        words = prompt.split()
        if not words:
            q0 = np.zeros(self.N)
            return field, q0

        center = np.zeros(self.N, dtype=np.float64)
        total_weight = 0.0

        for word in words:
            pos = self._hash_to_position(word)
            weight = self._word_importance(word)
            center += weight * pos
            total_weight += weight

        q0 = center / (total_weight + 1e-8)

        return field, q0

    def text_similarity(self, text1: str, text2: str) -> float:
        """
        Calcule la similarité sémantique entre deux textes.

        Utilise la distance géodésique entre les centres de masse
        dans l'espace métrique.

        Returns:
            Score de similarité entre 0 et 1
        """
        _, q1 = self.encode_prompt(text1)
        _, q2 = self.encode_prompt(text2)

        dist = self.metric.distance(q1, q2)
        # Normalisation empirique
        similarity = 1.0 / (1.0 + dist)
        return similarity
