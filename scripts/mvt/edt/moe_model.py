"""
Modèle MoE-MVT (Mixture-of-Experts — Modèle de Variance Topologique).

Implémentation complète en PyTorch d'un modèle de langage à mélange d'experts
où chaque expert est un « spécialiste topologique » d'une variété sémantique.
Optimisé pour l'exécution sur CPU avec des dimensions réduites.

Toutes les docstrings sont rédigées en français.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Configuration
# ===========================================================================


@dataclass
class MoEMVTConfig:
    """Configuration du modèle MoE-MVT.

    Cette classe regroupe tous les hyperparamètres nécessaires à la construction
    du modèle. Les valeurs par défaut sont choisies pour permettre une exécution
    fluide sur CPU.

    Attributs:
        vocab_size:            Taille du vocabulaire.
        d_model:               Dimension du modèle (espace de plongement).
        n_layers:              Nombre de blocs transformeurs.
        n_experts:             Nombre d'experts par couche MoE.
        top_k:                 Nombre d'experts activés par jeton.
        d_ff:                  Dimension intermédiaire des experts (MLP).
        max_seq_len:           Longueur maximale de la séquence.
        aux_loss_weight:       Poids de la perte auxiliaire d'équilibrage.
        router_temperature:    Température du routeur pour le softmax.
        embedding_dropout:     Taux d'abandon pour les plongements.
    """
    vocab_size: int = 8000
    d_model: int = 256
    n_layers: int = 4
    n_experts: int = 16
    top_k: int = 2
    d_ff: int = 512
    max_seq_len: int = 128
    aux_loss_weight: float = 0.01
    router_temperature: float = 1.0
    embedding_dropout: float = 0.1


# ===========================================================================
# Plongement topologique
# ===========================================================================


class TopoEmbedding(nn.Module):
    """Couche de plongement combinant plongement de jetons et encodage positionnel.

    Le plongement positionnel est appris (non sinusoïdal) et stocké comme
    paramètre pour s'adapter automatiquement aux régularités du corpus.

    Attributs:
        tok_embed: Plongement de jetons (vocab_size × d_model).
        pos_embed: Paramètre de position (1 × max_seq_len × d_model), appris.
        dropout:   Couche d'abandon appliquée après l'addition.
    """

    def __init__(self, config: MoEMVTConfig) -> None:
        super().__init__()
        self.tok_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.max_seq_len, config.d_model) * 0.02
        )
        self.dropout = nn.Dropout(config.embedding_dropout)
        self.max_seq_len = config.max_seq_len

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Calculer les états cachés à partir des identifiants de jetons.

        Args:
            tokens: Tenseur d'entiers de forme (batch, seq_len) contenant
                    les identifiants de jetons du vocabulaire.

        Returns:
            Tenseur de forme (batch, seq_len, d_model) représentant les
            états cachés après plongement + position + abandon.
        """
        seq_len = tokens.size(1)
        # Plongement de jetons : (batch, seq, d_model)
        x = self.tok_embed(tokens)
        # Ajout du plongement positionnel (tronqué à la longueur réelle)
        x = x + self.pos_embed[:, :seq_len, :]
        return self.dropout(x)


# ===========================================================================
# Routeur topologique
# ===========================================================================


class TopoRouter(nn.Module):
    """Routeur qui distribue chaque jeton vers les k meilleurs experts.

    Le routeur projette l'état caché dans l'espace des experts via une couche
    linéaire sans biais, applique un softmax avec température, sélectionne les
    top-k experts et calcule la perte auxiliaire d'équilibrage de charge.

    La perte auxiliaire favorise une répartition uniforme des jetons entre
    les experts pour éviter l'effondrement (collapse) où un seul expert
    traiterait tous les jetons.

    Attributs:
        gate:            Couche linéaire de projection (d_model → n_experts).
        temperature:     Température pour le softmax du routeur.
        n_experts:       Nombre total d'experts.
        top_k:           Nombre d'experts sélectionnés par jeton.
        aux_loss_weight: Poids de la perte auxiliaire.
    """

    def __init__(self, config: MoEMVTConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.n_experts, bias=False)
        self.temperature = config.router_temperature
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.aux_loss_weight = config.aux_loss_weight

    def forward(
        self, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Router les jetons vers les top-k experts.

        Args:
            hidden: États cachés de forme (batch, seq_len, d_model).

        Returns:
            Tuple de trois éléments :
            - router_logits : (batch, seq_len, n_experts) — logits bruts du routeur.
            - top_k_indices : (batch, seq_len, top_k) — indices des experts sélectionnés.
            - aux_loss      : scalaire — perte auxiliaire d'équilibrage de charge.
        """
        # Logits bruts : (batch, seq, n_experts)
        router_logits = self.gate(hidden)

        # Softmax avec température pour obtenir les probabilités de routage
        router_probs = F.softmax(router_logits / self.temperature, dim=-1)

        # Sélection des top-k experts par position
        # top_k_probs : (batch, seq, top_k)
        # top_k_indices : (batch, seq, top_k)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Perte auxiliaire d'équilibrage de charge
        # f_i = fraction des jetons routés vers l'expert i
        # aux_loss = n_experts * Σ(f_i²)
        batch_size, seq_len, _ = hidden.shape
        n_tokens = batch_size * seq_len

        # Compter combien de jetons sont routés vers chaque expert
        # On utilise les indices top_k pour construire un compteur
        flat_indices = top_k_indices.view(-1)  # (n_tokens * top_k,)
        expert_counts = torch.zeros(
            self.n_experts, device=hidden.device, dtype=hidden.dtype
        )
        expert_counts.scatter_add_(
            0, flat_indices, torch.ones_like(flat_indices, dtype=hidden.dtype)
        )
        # f_i = proportion de (jeton, slot) assignés à l'expert i
        total_slots = n_tokens * self.top_k
        f_i = expert_counts / total_slots
        # Perte auxiliaire : on veut minimiser Σ f_i² (équilibrage)
        aux_loss = self.n_experts * torch.sum(f_i ** 2)

        return router_logits, top_k_indices, aux_loss


# ===========================================================================
# Expert topologique
# ===========================================================================


class TopoExpert(nn.Module):
    """Expert individuel : réseau à deux couches avec activation GELU.

    Chaque expert est un MLP simple sans normalisation interne,
    représentant un « spécialiste topologique » d'une région de la
    variété sémantique. La transformation est :
        x → x @ W₁ᵀ → GELU → x @ W₂ᵀ

    Attributs:
        w1: Première couche linéaire (d_model → d_ff, sans biais).
        w2: Seconde couche linéaire (d_ff → d_model, sans biais).
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propager les entrées à travers l'expert.

        Args:
            x: Tenseur d'entrée de forme (..., d_model).

        Returns:
            Tenseur de sortie de forme (..., d_model) après passage
            dans le MLP à deux couches avec activation GELU.
        """
        return self.w2(F.gelu(self.w1(x)))


# ===========================================================================
# Couche MoE topologique
# ===========================================================================


class TopoMoE(nn.Module):
    """Couche MoE combinant le routeur et les experts.

    Pour chaque position de la séquence, le routeur sélectionne les top-k
    experts. La sortie est la combinaison pondérée des sorties des experts
    sélectionnés, où les poids sont les probabilités du routeur.

    Attributs:
        router:  Instance de TopoRouter.
        experts: Liste de modules contenant les n_experts instances de TopoExpert.
        top_k:   Nombre d'experts activés par jeton.
    """

    def __init__(self, config: MoEMVTConfig) -> None:
        super().__init__()
        self.router = TopoRouter(config)
        self.experts = nn.ModuleList(
            [TopoExpert(config.d_model, config.d_ff) for _ in range(config.n_experts)]
        )
        self.top_k = config.top_k
        self.n_experts = config.n_experts
        self.temperature = config.router_temperature

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propager les états cachés à travers la couche MoE.

        Args:
            x: États cachés de forme (batch, seq_len, d_model).

        Returns:
            Tuple de deux éléments :
            - output   : (batch, seq_len, d_model) — sortie combinée des experts.
            - aux_loss : scalaire — perte auxiliaire d'équilibrage du routeur.
        """
        batch, seq_len, d = x.shape

        # Routage : obtenir les logits, indices et perte auxiliaire
        router_logits, top_k_indices, aux_loss = self.router(x)

        # Probabilités du routeur (avec température)
        router_probs = F.softmax(router_logits / self.temperature, dim=-1)
        # Gather les probabilités top-k : (batch, seq, top_k)
        top_k_probs = torch.gather(router_probs, dim=-1, index=top_k_indices)

        # Aplatir pour un traitement efficace : (batch*seq, ...)
        x_flat = x.view(-1, d)                                     # (B*S, d_model)
        top_k_indices_flat = top_k_indices.view(-1, self.top_k)     # (B*S, top_k)
        top_k_probs_flat = top_k_probs.view(-1, self.top_k)         # (B*S, top_k)

        # Initialiser la sortie accumulée
        output_flat = torch.zeros_like(x_flat)

        # Pour chaque slot k dans top_k, calculer la contribution des experts
        for k in range(self.top_k):
            # Identifiants d'experts pour ce slot : (B*S,)
            expert_ids = top_k_indices_flat[:, k]
            # Poids de routage pour ce slot : (B*S,)
            weights = top_k_probs_flat[:, k]

            # Pour chaque expert, traiter les jetons qui lui sont assignés
            for e in range(self.n_experts):
                # Masque des positions assignées à l'expert e pour ce slot
                mask = expert_ids == e  # (B*S,) bool
                if not mask.any():
                    continue
                # Sortie de l'expert pour les jetons sélectionnés
                expert_out = self.experts[e](x_flat[mask])  # (n_selected, d_model)
                # Accumuler la sortie pondérée
                output_flat[mask] += weights[mask].unsqueeze(-1) * expert_out

        # Remettre en forme : (batch, seq_len, d_model)
        output = output_flat.view(batch, seq_len, d)

        return output, aux_loss


# ===========================================================================
# Attention topologique
# ===========================================================================


class TopoAttention(nn.Module):
    """Couche d'attention multi-têtes avec masque causal.

    Implémentation standard d'attention personnelle (self-attention) avec
    projection QKV fusionnée et masque causal pour la génération
    auto-régressive.

    Attributs:
        qkv:       Couche linéaire fusionnée (d_model → 3*d_model, sans biais).
        out_proj:  Couche de projection de sortie (d_model → d_model, sans biais).
        n_heads:   Nombre de têtes d'attention.
        d_head:    Dimension par tête (d_model // n_heads).
        causal_mask: Masque causal enregistré comme tampon.
    """

    def __init__(self, config: MoEMVTConfig, n_heads: int = 4) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = n_heads
        self.d_head = config.d_model // n_heads
        assert self.d_head * n_heads == config.d_model, (
            "d_model doit être divisible par n_heads"
        )

        # Projection QKV fusionnée
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        # Projection de sortie
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Masque causal pré-calculé : True aux positions interdites
        # Forme : (1, 1, max_seq_len, max_seq_len)
        causal = torch.triu(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculer l'attention multi-têtes avec masque causal.

        Args:
            x: États cachés de forme (batch, seq_len, d_model).

        Returns:
            Tenseur de forme (batch, seq_len, d_model) après attention.
        """
        batch, seq_len, _ = x.shape

        # Projection QKV et séparation : (batch, seq, 3*d_model) → 3 × (batch, seq, d_model)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape pour multi-têtes : (batch, n_heads, seq, d_head)
        q = q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Produit scalaire Q·Kᵀ mis à l'échelle
        scale = math.sqrt(self.d_head)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, S, S)

        # Application du masque causal
        mask = self.causal_mask[:, :, :seq_len, :seq_len]  # (1, 1, S, S)
        attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        # Softmax pour obtenir les poids d'attention
        attn_weights = F.softmax(attn_weights, dim=-1)
        # Remplacement des NaN (causés par des lignes entièrement masquées)
        attn_weights = attn_weights.nan_to_num(0.0)

        # Application des poids à V
        attn_output = torch.matmul(attn_weights, v)  # (B, H, S, d_head)

        # Reshape retour : (B, H, S, d_head) → (B, S, d_model)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        )

        # Projection de sortie
        return self.out_proj(attn_output)


# ===========================================================================
# Bloc transformeur topologique
# ===========================================================================


class TopoBlock(nn.Module):
    """Bloc transformeur combinant attention et couche MoE.

    Architecture pre-norm avec connexions résiduelles :
        h = x + Attn(Norm₁(x))
        out, aux = MoE(Norm₂(h))
        return (x + out, aux)

    Attributs:
        norm1: Première normalisation de couche (avant attention).
        attn:  Couche d'attention multi-têtes.
        norm2: Seconde normalisation de couche (avant MoE).
        moe:   Couche MoE (routeur + experts).
    """

    def __init__(self, config: MoEMVTConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = TopoAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.moe = TopoMoE(config)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propager à travers le bloc transformeur avec connexions résiduelles.

        Args:
            x: États cachés de forme (batch, seq_len, d_model).

        Returns:
            Tuple de deux éléments :
            - output   : (batch, seq_len, d_model) — états cachés mis à jour.
            - aux_loss : scalaire — perte auxiliaire d'équilibrage du routeur MoE.
        """
        # Connexion résiduelle + attention
        h = x + self.attn(self.norm1(x))
        # Traitement MoE avec perte auxiliaire
        moe_out, aux_loss = self.moe(self.norm2(h))
        # Connexion résiduelle + MoE
        output = x + moe_out
        return output, aux_loss


# ===========================================================================
# Modèle complet MoE-MVT
# ===========================================================================


class MoEMVT(nn.Module):
    """Modèle complet MoE-MVT (Modèle de Variance Topologique).

    Ce modèle de langage combine des plongements topologiques, des couches
    d'attention multi-têtes et des couches MoE (Mixture-of-Experts) pour
    capturer la structure topologique de la variété sémantique du langage.

    Les poids de la tête de langage (lm_head) sont liés aux plongements de
    jetons (weight tying) pour réduire le nombre de paramètres et améliorer
    la généralisation.

    Attributs:
        embed:    Couche de plongement (TopoEmbedding).
        blocks:   Liste des blocs transformeurs (TopoBlock).
        norm_out: Normalisation de couche finale.
        lm_head:  Tête de prédiction du vocabulaire (liée à embed.tok_embed).
        config:   Configuration du modèle.
    """

    def __init__(self, config: MoEMVTConfig) -> None:
        super().__init__()
        self.config = config

        # Plongement
        self.embed = TopoEmbedding(config)

        # Blocs transformeurs
        self.blocks = nn.ModuleList(
            [TopoBlock(config) for _ in range(config.n_layers)]
        )

        # Normalisation finale
        self.norm_out = nn.LayerNorm(config.d_model)

        # Tête de langage — poids liés au plongement de jetons
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.tok_embed.weight

    def forward(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagation avant complète du modèle.

        Args:
            tokens: Tenseur d'entiers de forme (batch, seq_len) contenant
                    les identifiants de jetons du vocabulaire.

        Returns:
            Tuple de deux éléments :
            - logits : (batch, seq_len, vocab_size) — logits sur le vocabulaire.
            - aux_loss : scalaire — somme des pertes auxiliaires de toutes les couches.
        """
        # Plongement initial
        x = self.embed(tokens)  # (batch, seq, d_model)

        # Accumulateur de pertes auxiliaires
        total_aux_loss = torch.tensor(0.0, device=tokens.device)

        # Propagation à travers les blocs transformeurs
        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss = total_aux_loss + aux_loss

        # Normalisation finale
        x = self.norm_out(x)

        # Projection vers le vocabulaire
        logits = self.lm_head(x)  # (batch, seq, vocab_size)

        return logits, total_aux_loss

    def count_params(self) -> Tuple[int, int]:
        """Compter les paramètres du modèle.

        Calcule le nombre total de paramètres uniques (avec partage de poids)
        et le nombre de paramètres actifs par jeton (tenant compte de
        l'activation partielle des experts dans le MoE).

        Returns:
            Tuple de deux entiers :
            - total_params        : Nombre total de paramètres uniques.
            - active_params_per_token : Paramètres activés pour un seul jeton.
        """
        # Nombre total de paramètres uniques
        total_params = sum(p.numel() for p in self.parameters())

        # Paramètres actifs par jeton : total - experts inactifs
        # Chaque jeton active top_k experts par couche au lieu de n_experts
        cfg = self.config
        expert_params = 2 * cfg.d_model * cfg.d_ff  # w1 + w2, sans biais
        inactive_per_layer = (cfg.n_experts - cfg.top_k) * expert_params
        inactive_total = cfg.n_layers * inactive_per_layer
        active_params_per_token = total_params - inactive_total

        return int(total_params), int(active_params_per_token)


# ===========================================================================
# Fonction utilitaire autonome
# ===========================================================================


def count_params(model: MoEMVT) -> Tuple[int, int]:
    """Compter les paramètres d'un modèle MoEMVT.

    Fonction autonome qui délègue le calcul à la méthode count_params
    du modèle. Si le modèle ne possède pas cette méthode, elle calcule
    le nombre total de paramètres sans la distinction actif/inactif.

    Args:
        model: Instance du modèle MoEMVT (ou tout modèle nn.Module).

    Returns:
        Tuple de deux entiers :
        - total_params            : Nombre total de paramètres.
        - active_params_per_token : Paramètres actifs par jeton (0 si non applicable).
    """
    if hasattr(model, "count_params") and callable(model.count_params):
        return model.count_params()
    total = sum(p.numel() for p in model.parameters())
    return int(total), int(total)


# ===========================================================================
# Point d'entrée pour les tests rapides
# ===========================================================================


if __name__ == "__main__":
    # Test rapide du modèle sur CPU
    print("=== Test du modèle MoE-MVT ===")

    config = MoEMVTConfig()
    model = MoEMVT(config)
    model.eval()  # Mode évaluation (désactive dropout)

    total, active = model.count_params()
    print(f"Paramètres totaux        : {total:,}")
    print(f"Paramètres actifs/jeton  : {active:,}")
    print(f"Ratio de parcimonie     : {active/total:.1%}")

    # Propagation avant avec des jetons aléatoires
    batch_size, seq_len = 2, 32
    tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits, aux_loss = model(tokens)

    print(f"\nForme d'entrée  : {tokens.shape}")
    print(f"Forme de sortie : {logits.shape}")
    print(f"Perte auxiliaire: {aux_loss.item():.4f}")

    # Vérification du partage de poids
    assert model.lm_head.weight is model.embed.tok_embed.weight, (
        "Erreur : les poids de lm_head ne sont pas liés à tok_embed !"
    )
    print("\n✓ Partage de poids (weight tying) vérifié avec succès.")
    print("=== Tous les tests passés ===")
