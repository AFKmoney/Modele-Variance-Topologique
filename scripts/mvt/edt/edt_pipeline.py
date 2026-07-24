"""
EDT — Expert Decoupled Training pour MoE-MVT.
================================================
Pipeline d'entraînement en 4 phases optimisé pour CPU.
Chaque composant est entraîné indépendamment, puis aligné.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field

from .moe_model import MoEMVT, MoEMVTConfig


# ===========================================================================
# Configuration EDT
# ===========================================================================

@dataclass
class EDTConfig:
    """Configuration complète du pipeline EDT pour CPU."""
    # Phase 1 — Experts
    phase1_steps_per_expert: int = 150
    phase1_batch_size: int = 32
    phase1_lr: float = 1e-3
    phase1_weight_decay: float = 0.01
    phase1_hidden_samples: int = 2000

    # Phase 2a — Attention
    phase2a_steps_per_layer: int = 300
    phase2a_batch_size: int = 16
    phase2a_lr: float = 1e-3

    # Phase 2b — Embedding
    phase2b_n_tokens: int = 2_000_000
    phase2b_batch_size: int = 64
    phase2b_seq_len: int = 64
    phase2b_lr: float = 1e-3

    # Phase 3 — Joint
    phase3_n_tokens: int = 500_000
    phase3_batch_size: int = 4
    phase3_seq_len: int = 32
    phase3_lr: float = 3e-4
    phase3_n_active_layers: int = 2
    phase3_aux_loss_weight: float = 0.01

    # Général
    seq_len: int = 32
    grad_clip: float = 1.0
    device: str = "cpu"
    save_dir: str = "/home/z/my-project/download/mvt_checkpoints"


# ===========================================================================
# Utilitaires
# ===========================================================================

def _save(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  💾 Checkpoint sauvegardé : {path}")

def _load(model, path):
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    print(f"  📂 Checkpoint chargé : {path}")

def _format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.2f}h"


# ===========================================================================
# Banque d'états cachés
# ===========================================================================

def generate_hidden_states(
    model: MoEMVT,
    corpus_tokens: torch.Tensor,
    config: EDTConfig,
    seq_len: Optional[int] = None,
) -> torch.Tensor:
    """
    Générer une banque d'états cachés réels à partir du corpus.

    Utilise l'embedding du modèle pour produire des états cachés
    sans passer par les couches transformeurs (séparable).
    """
    if seq_len is None:
        seq_len = config.seq_len
    n_samples = config.phase1_hidden_samples
    batch_size = config.phase1_batch_size
    n_batches = n_samples // batch_size
    d_model = model.config.d_model

    print(f"\n  Génération de {n_samples} états cachés (seq_len={seq_len})...", flush=True)
    hidden_states = []
    corpus_len = len(corpus_tokens)

    model.eval()
    with torch.no_grad():
        for i in range(n_batches):
            # Échantillonner des séquences aléatoires
            max_start = max(1, corpus_len - seq_len - 1)
            idx = torch.randint(0, max_start, (batch_size,))
            tokens = torch.stack([corpus_tokens[j:j+seq_len] for j in idx]).to(config.device)

            h = model.embed(tokens)  # (batch, seq, d_model)
            hidden_states.append(h.cpu())

    hidden_bank = torch.cat(hidden_states, dim=0)  # (n_samples, seq, d_model)
    print(f"  ✓ Banque créée : {hidden_bank.shape}", flush=True)
    return hidden_bank


# ===========================================================================
# Phase 1 — Experts indépendants
# ===========================================================================

def phase1_experts(
    model: MoEMVT,
    hidden_bank: torch.Tensor,
    config: EDTConfig,
) -> Dict[str, float]:
    """
    Phase 1 — Pré-entraînement indépendant de chaque expert.

    Chaque expert apprend à transformer des états cachés réels.
    Loss = MSE(expert(h_in), h_target) où h_target = h[next_position].
    """
    print("\n" + "=" * 64)
    print("  PHASE 1 — Pré-entraînement indépendant des experts")
    print("=" * 64)

    t0 = time.time()
    d_model = model.config.d_model
    n_experts_trained = 0
    total_loss = 0.0

    for layer_idx in range(len(model.blocks)):
        moe = model.blocks[layer_idx].moe
        layer_loss_sum = 0.0

        for expert_idx in range(moe.n_experts):
            expert = moe.experts[expert_idx]
            params = list(expert.parameters())
            opt = torch.optim.AdamW(params, lr=config.phase1_lr,
                                     weight_decay=config.phase1_weight_decay)

            for step in range(config.phase1_steps_per_expert):
                # Échantillonner des états cachés
                batch_idx = torch.randint(0, len(hidden_bank) - 1,
                                          (config.phase1_batch_size,))
                h_in = hidden_bank[batch_idx].to(config.device)
                target_idx = (batch_idx + 1) % len(hidden_bank)
                h_target = hidden_bank[target_idx].to(config.device)

                opt.zero_grad()
                h_out = expert(h_in)
                loss = F.mse_loss(h_out, h_target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
                opt.step()

                layer_loss_sum += loss.item()

            n_experts_trained += 1

        avg_loss = layer_loss_sum / (moe.n_experts * config.phase1_steps_per_expert)
        elapsed = time.time() - t0
        print(f"  Couche {layer_idx+1}/{len(model.blocks)} : "
              f"{moe.n_experts} experts entraînés — "
              f"loss={avg_loss:.4f} — {_format_time(elapsed)}", flush=True)

    total_time = time.time() - t0
    avg_total_loss = total_loss / max(1, n_experts_trained)

    stats = {"time": total_time, "avg_loss": avg_total_loss, "n_experts": n_experts_trained}
    print(f"\n  ✓ Phase 1 terminée en {_format_time(total_time)}")
    print(f"    {n_experts_trained} experts entraînés, loss moyen = {avg_total_loss:.4f}")

    return stats


# ===========================================================================
# Phase 2a — Attention indépendante
# ===========================================================================

def phase2a_attention(
    model: MoEMVT,
    hidden_bank: torch.Tensor,
    config: EDTConfig,
) -> Dict[str, float]:
    """
    Phase 2a — Pré-entraînement indépendant de chaque couche d'attention.

    L'attention apprend à structurer et débruiter les états cachés
    provenant de l'embedding.
    """
    print("\n" + "=" * 64)
    print("  PHASE 2a — Pré-entraînement indépendant de l'attention")
    print("=" * 64)

    t0 = time.time()
    total_loss = 0.0
    total_steps = 0

    for layer_idx in range(len(model.blocks)):
        block = model.blocks[layer_idx]
        attn = block.attn
        norm = block.norm1
        params = list(attn.parameters()) + list(norm.parameters())
        opt = torch.optim.AdamW(params, lr=config.phase2a_lr,
                                 weight_decay=config.phase1_weight_decay)

        layer_loss_sum = 0.0

        for step in range(config.phase2a_steps_per_layer):
            batch_idx = torch.randint(0, len(hidden_bank) - 1,
                                      (config.phase2a_batch_size,))
            h_in = hidden_bank[batch_idx].to(config.device)
            target_idx = (batch_idx + 1) % len(hidden_bank)
            h_target = hidden_bank[target_idx].to(config.device)

            opt.zero_grad()
            h_normed = norm(h_in)
            attn_out = attn(h_normed)
            h_out = h_in + attn_out  # connexion résiduelle
            loss = F.mse_loss(h_out, h_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
            opt.step()

            layer_loss_sum += loss.item()

        avg_loss = layer_loss_sum / config.phase2a_steps_per_layer
        total_loss += avg_loss
        total_steps += 1
        elapsed = time.time() - t0
        print(f"  Couche {layer_idx+1}/{len(model.blocks)} : "
              f"loss={avg_loss:.4f} — {_format_time(elapsed)}", flush=True)

    total_time = time.time() - t0
    avg_total_loss = total_loss / total_steps

    stats = {"time": total_time, "avg_loss": avg_total_loss}
    print(f"\n  ✓ Phase 2a terminée en {_format_time(total_time)}")
    print(f"    loss moyen = {avg_total_loss:.4f}")

    return stats


# ===========================================================================
# Phase 2b — Embedding
# ===========================================================================

def phase2b_embedding(
    model: MoEMVT,
    corpus_tokens: torch.Tensor,
    config: EDTConfig,
) -> Dict[str, float]:
    """
    Phase 2b — Pré-entraînement de l'embedding par next-token prediction.

    Seul l'embedding est entraîné (tout le reste est gelé).
    Très rapide car pas de couches transformeurs dans le graphe.
    """
    print("\n" + "=" * 64)
    print("  PHASE 2b — Pré-entraînement de l'embedding")
    print("=" * 64)

    t0 = time.time()

    # Geler tout sauf l'embedding
    for p in model.parameters():
        p.requires_grad = False
    for p in model.embed.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(model.embed.parameters(),
                             lr=config.phase2b_lr,
                             weight_decay=config.phase1_weight_decay)

    vocab_size = model.config.vocab_size
    seq_len = config.phase2b_seq_len
    batch_size = config.phase2b_batch_size
    corpus_len = len(corpus_tokens)
    n_tokens = config.phase2b_n_tokens
    n_steps = n_tokens // (batch_size * seq_len)

    tokens = corpus_tokens.to(config.device)
    loss_history = []

    print(f"  {n_tokens/1e6:.1f}M tokens, {n_steps:,} steps, "
          f"batch={batch_size}, seq={seq_len}", flush=True)

    for step in range(n_steps):
        max_start = max(1, corpus_len - batch_size * seq_len - 1)
        idx = torch.randint(0, max_start, (1,)).item()
        inp = torch.stack([tokens[idx + b*seq_len : idx + (b+1)*seq_len]
                           for b in range(batch_size)])
        tgt = torch.stack([tokens[idx + b*seq_len + 1 : idx + (b+1)*seq_len + 1]
                           for b in range(batch_size)])

        opt.zero_grad()
        h = model.embed(inp)
        logits = model.lm_head(h)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.embed.parameters(), config.grad_clip)
        opt.step()

        loss_history.append(loss.item())

        if step % 500 == 0:
            elapsed = time.time() - t0
            tok_per_sec = (step + 1) * batch_size * seq_len / elapsed
            print(f"  S{step:>6}/{n_steps}  loss={loss.item():.4f}  "
                  f"tok/s={tok_per_sec:.0f}", flush=True)

    # Dégeler tout
    for p in model.parameters():
        p.requires_grad = True

    total_time = time.time() - t0
    avg_loss = np.mean(loss_history)
    tok_per_sec = n_tokens / total_time

    stats = {"time": total_time, "avg_loss": avg_loss, "tok_per_sec": tok_per_sec}
    print(f"\n  ✓ Phase 2b terminée en {_format_time(total_time)}")
    print(f"    loss moyen = {avg_loss:.4f}, {tok_per_sec:.0f} tok/s")

    return stats


# ===========================================================================
# PGSG — Partial Gradient Sequential Update
# ===========================================================================

class PGSG:
    """
    Partial Gradient Sequential Update pour CPU.
    Sélectionne n_active_layers couches par step — réduit le backprop de ~60%.
    """
    def __init__(self, model: MoEMVT, n_active_layers: int = 2):
        self.model = model
        self.n_active_layers = min(n_active_layers, len(model.blocks))
        self.n_layers = len(model.blocks)
        self.step_counter = 0

    def step_begin(self) -> set:
        """Active un sous-ensemble de couches pour ce step."""
        active = set()
        for i in range(self.n_active_layers):
            idx = (self.step_counter + i) % self.n_layers
            active.add(idx)

        # Désactiver tous les gradients
        for p in self.model.parameters():
            p.requires_grad = False

        # Activer les couches sélectionnées + embedding + lm_head
        for idx in active:
            for p in self.model.blocks[idx].parameters():
                p.requires_grad = True

        for p in self.model.embed.parameters():
            p.requires_grad = True
        for p in self.model.norm_out.parameters():
            p.requires_grad = True

        self.active_layers = active
        return active

    def step_end(self):
        """Remet tous les gradients activés et avance le compteur."""
        for p in self.model.parameters():
            p.requires_grad = True
        self.step_counter += 1


# ===========================================================================
# Phase 3 — Joint fine-tune
# ===========================================================================

def phase3_joint(
    model: MoEMVT,
    corpus_tokens: torch.Tensor,
    config: EDTConfig,
) -> Dict[str, float]:
    """
    Phase 3 — Joint fine-tune avec PGSG.

    Tous les composants sont dégeler et alignés ensemble.
    PGSG sélectionne un sous-ensemble de couches par step pour économiser CPU.
    """
    print("\n" + "=" * 64)
    print("  PHASE 3 — Joint fine-tune (alignement global)")
    print("=" * 64)

    t0 = time.time()

    # Dégeler tout
    for p in model.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(model.parameters(),
                             lr=config.phase3_lr,
                             weight_decay=config.phase1_weight_decay)

    pgsg = PGSG(model, n_active_layers=config.phase3_n_active_layers)

    vocab_size = model.config.vocab_size
    seq_len = config.phase3_seq_len
    batch_size = config.phase3_batch_size
    corpus_len = len(corpus_tokens)
    n_tokens = config.phase3_n_tokens
    n_steps = n_tokens // (batch_size * seq_len)
    aux_w = config.phase3_aux_loss_weight

    tokens = corpus_tokens.to(config.device)
    loss_history = []

    print(f"  {n_tokens/1e6:.1f}M tokens, {n_steps:,} steps, "
          f"batch={batch_size}, PGSG active={config.phase3_n_active_layers}/"
          f"{len(model.blocks)}", flush=True)

    for step in range(n_steps):
        max_start = max(1, corpus_len - batch_size * seq_len - 1)
        idx = torch.randint(0, max_start, (1,)).item()
        inp = torch.stack([tokens[idx + b*seq_len : idx + (b+1)*seq_len]
                           for b in range(batch_size)])
        tgt = torch.stack([tokens[idx + b*seq_len + 1 : idx + (b+1)*seq_len + 1]
                           for b in range(batch_size)])

        # PGSG : sélectionner les couches actives
        active = pgsg.step_begin()

        opt.zero_grad()
        logits, aux_loss = model(inp)
        ce = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1))
        loss = ce + aux_w * torch.clamp(aux_loss, max=1.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        opt.step()
        pgsg.step_end()

        loss_history.append(ce.item())

        if step % 200 == 0:
            elapsed = time.time() - t0
            tok_per_sec = (step + 1) * batch_size * seq_len / elapsed
            print(f"  S{step:>5}/{n_steps}  ce={ce.item():.4f}  aux={aux_loss.item():.4f}  "
                  f"tok/s={tok_per_sec:.0f}  layers={sorted(active)}", flush=True)

    total_time = time.time() - t0
    avg_loss = np.mean(loss_history)
    tok_per_sec = n_tokens / total_time

    stats = {"time": total_time, "avg_loss": avg_loss, "tok_per_sec": tok_per_sec}
    print(f"\n  ✓ Phase 3 terminée en {_format_time(total_time)}")
    print(f"    loss moyen = {avg_loss:.4f}, {tok_per_sec:.0f} tok/s")

    return stats


# ===========================================================================
# Pipeline principal
# ===========================================================================

def run_edt(
    model: MoEMVT,
    corpus_tokens: torch.Tensor,
    config: Optional[EDTConfig] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Pipeline EDT complet : 4 phases d'entraînement.

    Args:
        model: Modèle MoE-MVT pré-initialisé.
        corpus_tokens: Corpus tokenisé (1D tensor d'entiers).
        config: Configuration EDT (None = défaut CPU).
        verbose: Afficher les détails.

    Returns:
        Dictionnaire avec les statistiques de chaque phase.
    """
    if config is None:
        config = EDTConfig()

    model = model.to(config.device)
    model.train()

    # Stats du modèle
    total_p, active_p = model.count_params()
    model_mb = sum(p.numel() * 4 for p in model.parameters()) / 1e6

    if verbose:
        print("\n" + "=" * 64)
        print("  EXPERT DECOUPLED TRAINING (EDT) — MoE-MVT sur CPU")
        print("=" * 64)
        print(f"  Paramètres totaux    : {total_p:,} ({total_p/1e6:.1f}M)")
        print(f"  Paramètres actifs   : {active_p:,} ({active_p/1e6:.1f}M)")
        print(f"  Sparsité             : {1 - active_p/total_p:.1%}")
        print(f"  Taille modèle (float32) : {model_mb:.1f} MB")
        print(f"  Corpus               : {len(corpus_tokens):,} tokens")
        print(f"  Couches              : {model.config.n_layers}")
        print(f"  Experts/couche       : {model.config.n_experts}")
        print(f"  Top-k                : {model.config.top_k}")
        print(f"  Device               : {config.device}")

    all_stats = {}
    t_total = time.time()

    # Phase 1 : Experts
    hidden_bank = generate_hidden_states(model, corpus_tokens, config)
    stats1 = phase1_experts(model, hidden_bank, config)
    all_stats["phase1"] = stats1
    ckpt1 = os.path.join(config.save_dir, "after_phase1.pt")
    _save(model, ckpt1)

    # Phase 2a : Attention
    stats2a = phase2a_attention(model, hidden_bank, config)
    all_stats["phase2a"] = stats2a
    ckpt2a = os.path.join(config.save_dir, "after_phase2a.pt")
    _save(model, ckpt2a)

    # Libérer la banque d'états cachés
    del hidden_bank

    # Phase 2b : Embedding
    stats2b = phase2b_embedding(model, corpus_tokens, config)
    all_stats["phase2b"] = stats2b
    ckpt2b = os.path.join(config.save_dir, "after_phase2b.pt")
    _save(model, ckpt2b)

    # Phase 3 : Joint
    stats3 = phase3_joint(model, corpus_tokens, config)
    all_stats["phase3"] = stats3
    ckpt3 = os.path.join(config.save_dir, "mvt_edt_final.pt")
    _save(model, ckpt3)

    total_time = time.time() - t_total
    all_stats["total_time"] = total_time

    # Résumé
    if verbose:
        print("\n" + "=" * 64)
        print("  RÉSUMÉ EDT — TERMINÉ")
        print("=" * 64)
        print(f"  Phase 1 (Experts)     : {_format_time(stats1['time'])}")
        print(f"  Phase 2a (Attention)   : {_format_time(stats2a['time'])}")
        print(f"  Phase 2b (Embedding)   : {_format_time(stats2b['time'])}")
        print(f"  Phase 3 (Joint)        : {_format_time(stats3['time'])}")
        print(f"  {'—' * 40}")
        print(f"  TEMPS TOTAL             : {_format_time(total_time)}")
        print(f"  Modèle sauvegardé       : {ckpt3}")

    return all_stats


# ===========================================================================
# Corpus synthétique (pour tests)
# ===========================================================================

def generate_synthetic_corpus(
    vocab_size: int = 8000,
    length: int = 100_000,
    seed: int = 42,
) -> torch.Tensor:
    """
    Générer un corpus synthétique avec des régularités bigram.

    Pour tester EDT sans corpus réel.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Créer des bigrammes fréquents pour que l'embedding apprenne quelque chose
    tokens = torch.zeros(length, dtype=torch.long)
    tokens[0] = torch.randint(0, vocab_size, (1,)).item()

    # Matrice de transition bigram (some structure)
    trans = torch.rand(vocab_size, vocab_size)
    # Renforcer 50 paires bigram fortes
    for _ in range(50):
        i, j = torch.randint(0, vocab_size, (2,)).tolist()
        trans[i, :] *= 0.1
        trans[i, j] = 10.0

    trans = trans / trans.sum(dim=1, keepdim=True)

    for i in range(1, length):
        prev = tokens[i-1].item()
        probs = trans[prev]
        tokens[i] = torch.multinomial(probs, 1).item()

    return tokens


# ===========================================================================
# Point d'entrée
# ===========================================================================

if __name__ == "__main__":
    print("=== EDT — Test rapide sur CPU ===\n")

    # Config petite pour test rapide
    model_cfg = MoEMVTConfig(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        n_experts=4,
        top_k=1,
        d_ff=128,
        max_seq_len=32,
    )

    edt_cfg = EDTConfig(
        phase1_steps_per_expert=10,
        phase1_hidden_samples=200,
        phase1_batch_size=8,
        phase2a_steps_per_layer=10,
        phase2a_batch_size=4,
        phase2b_n_tokens=5000,
        phase2b_batch_size=4,
        phase2b_seq_len=8,
        phase3_n_tokens=2000,
        phase3_batch_size=2,
        phase3_seq_len=8,
        phase3_n_active_layers=1,
        save_dir="/home/z/my-project/download/mvt_checkpoints",
    )

    model = MoEMVT(model_cfg)
    corpus = generate_synthetic_corpus(vocab_size=1000, length=5000)

    stats = run_edt(model, corpus, edt_cfg, verbose=True)
    print(f"\n✓ EDT test terminé avec succès en {_format_time(stats['total_time'])}")
