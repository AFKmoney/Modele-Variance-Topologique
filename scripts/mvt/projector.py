"""
MVT Projector - Projection Continue vers Discret.
====================================================

Transforme la trajectoire continue q(t) dans R^N en texte lisible.
C'est l'étape finale qui "photographie" la courbe continue et la
projette en mots compréhensibles par l'humain.
"""

from __future__ import annotations

import hashlib
import numpy as np
from typing import Optional, List, Dict, Tuple

from .config import MVTConfig
from .core.metric_tensor import MetricTensor
from .encoder import InputEncoder


class ManifoldProjector:
    """
    Projecteur Variété Continue → Texte Discret.

    Le projecteur "inverse" l'encodeur : à partir d'une position
    dans R^N, il trouve le mot/concept le plus proche et le
    projette en texte.

    Méthode:
    1. Échantillonner la trajectoire à intervalles réguliers
    2. Pour chaque point, trouver le concept le plus proche
       dans le "vocabulaire inversé"
    3. Assembler les concepts en texte fluide
    """

    def __init__(
        self,
        config: MVTConfig,
        metric: MetricTensor,
        encoder: InputEncoder,
    ):
        self.config = config
        self.metric = metric
        self.encoder = encoder
        self.N = config.ambient_dim

        # Vocabulaire inversé : position → mot
        # Construit à partir du cache de l'encodeur
        self._inv_vocab: Dict[Tuple[float, ...], str] = {}

        # Vocabulaire étendu (pour la génération)
        self._extended_vocab: List[Tuple[str, np.ndarray]] = []

        # Paramètres de projection
        self._sample_rate = 5  # Échantillonner 1 point tous les N pas
        self._min_word_length = 2
        self._max_words = 100

    def _build_inverse_vocab(self):
        """
        Construit le vocabulaire inversé à partir du cache de l'encodeur.

        Chaque mot connu est associé à sa position dans l'espace.
        """
        self._inv_vocab = {}
        for word, pos in self.encoder._word_cache.items():
            # Arrondir pour créer une clé hashable
            key = tuple(np.round(pos, decimals=4))
            self._inv_vocab[key] = word

    def _extend_vocabulary(self, extra_words: Optional[List[str]] = None):
        """
        Étend le vocabulaire avec des mots supplémentaires.

        Si aucun mot n'est fourni, utilise un vocabulaire de base.
        """
        if extra_words is None:
            extra_words = self._default_vocabulary()

        for word in extra_words:
            if word not in self.encoder._word_cache:
                pos = self.encoder._hash_to_position(word)
                self._extended_vocab.append((word, pos))

    def _default_vocabulary(self) -> List[str]:
        """
        Vocabulaire par défaut pour la projection.

        Contient des mots de base pour permettre la génération même
        si aucun texte n'a été encodé auparavant.
        """
        # Mots de base en français et anglais
        words = [
            # Français
            "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou",
            "mais", "donc", "car", "est", "sont", "a", "ont", "dans", "pour",
            "par", "sur", "avec", "sans", "ce", "cette", "ces", "il", "elle",
            "nous", "vous", "ils", "elles", "qui", "que", "quoi", "comment",
            "pourquoi", "quand", "où", "ici", "là", "bien", "mal", "plus",
            "moins", "très", "beaucoup", "peu", "toujours", "jamais", "pas",
            "aussi", "comme", "entre", "sous", "avant", "après", "depuis",
            "pendant", "alors", "donc", "ensuite", "premier", "deuxième",
            "troisième", "dernier", "nouveau", "ancien", "grand", "petit",
            "bon", "mauvais", "vrai", "faux", "possible", "impossible",
            # Anglais
            "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "shall", "not",
            "but", "and", "or", "if", "then", "else", "when", "while",
            "because", "therefore", "however", "also", "very", "much",
            "more", "less", "most", "least", "all", "some", "any",
            "many", "few", "each", "every", "both", "neither", "either",
            # Concepts abstraits
            "idée", "concept", "pensée", "raison", "logique", "vérité",
            "beauté", "justice", "liberté", "égalité", "fraternité",
            "temps", "espace", "énergie", "matière", "vie", "mort",
            "beginning", "end", "start", "finish", "create", "destroy",
            "build", "break", "open", "close", "give", "take",
            "function", "variable", "class", "object", "method",
            "return", "import", "def", "if", "else", "for", "while",
            "print", "input", "output", "data", "value", "result",
        ]
        return words

    def project_trajectory(
        self, trajectory: np.ndarray, max_words: Optional[int] = None
    ) -> str:
        """
        Projette une trajectoire continue en texte discret.

        Processus:
        1. Échantillonner la trajectoire
        2. Pour chaque point, trouver le concept le plus proche
        3. Filtrer et assembler

        Args:
            trajectory: Trajectoire (T, N)
            max_words: Nombre max de mots, ou None pour la config

        Returns:
            Texte généré
        """
        if max_words is None:
            max_words = self._max_words

        # Assurer qu'on a un vocabulaire inversé
        self._build_inverse_vocab()
        self._extend_vocabulary()

        # Échantillonner la trajectoire
        T_len = len(trajectory)
        if T_len == 0:
            return ""

        indices = list(range(0, T_len, self._sample_rate))
        sampled = trajectory[indices]

        # Trouver le mot le plus proche pour chaque point
        words: List[str] = []
        seen_words: set = set()  # Éviter les répétitions immédiates

        for q in sampled:
            best_word = self._find_nearest_word(q)
            if best_word and best_word not in seen_words:
                words.append(best_word)
                seen_words.add(best_word)
            elif best_word:
                seen_words.clear()
                seen_words.add(best_word)
                words.append(best_word)

            if len(words) >= max_words:
                break

        return " ".join(words)

    def _find_nearest_word(self, q: np.ndarray) -> Optional[str]:
        """
        Trouve le mot le plus proche (distance géodésique) d'un point q.

        Args:
            q: Point dans l'espace (N,)

        Returns:
            Mot le plus proche, ou None
        """
        best_word = None
        best_dist = float('inf')

        # Chercher dans le cache de l'encodeur
        for word, pos in self.encoder._word_cache.items():
            dist = self.metric.distance(q, pos)
            if dist < best_dist:
                best_dist = dist
                best_word = word

        # Chercher dans le vocabulaire étendu
        for word, pos in self._extended_vocab:
            dist = self.metric.distance(q, pos)
            if dist < best_dist:
                best_dist = dist
                best_word = word

        # Seuil de distance maximale
        if best_dist > 5.0:
            return None

        return best_word

    def project_with_positions(
        self, trajectory: np.ndarray
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Projette la trajectoire et retourne les positions associées.

        Returns:
            Liste de (mot, position) pour chaque point projeté
        """
        self._build_inverse_vocab()
        self._extend_vocabulary()

        T_len = len(trajectory)
        indices = list(range(0, T_len, self._sample_rate))

        results: List[Tuple[str, np.ndarray]] = []
        for idx in indices:
            q = trajectory[idx]
            word = self._find_nearest_word(q)
            if word:
                results.append((word, q.copy()))

        return results

    def singularity_declaration(self, q: np.ndarray) -> str:
        """
        Déclare une singularité (l'IA ne sait pas).

        Au lieu d'halluciner, le MVT déclare explicitement son ignorance.

        Args:
            q: Point de singularité

        Returns:
            Message de déclaration
        """
        return "[Singularité — Je ne dispose pas d'information suffisante pour continuer.]"

    def __repr__(self) -> str:
        return (
            f"ManifoldProjector(N={self.N}, "
            f"vocab_size={len(self.encoder._word_cache) + len(self._extended_vocab)})"
        )
