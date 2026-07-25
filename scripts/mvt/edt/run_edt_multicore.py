"""
EDT Multi-Core — Expert Decoupled Training with CPU/GPU Parallelism
====================================================================
Phase 1 (expert training) is embarrassingly parallel: each expert trains
independently on the same hidden state bank. This script distributes
experts across all available CPU cores using ProcessPoolExecutor,
or across GPUs (Phase 1 only).

Features:
  - Multi-core CPU parallelism for Phase 1 (~N× speedup)
  - GPU training support for all phases
  - Resume from checkpoint (--resume flag)
  - Kuramoto-Metric coupling integration (--kuramoto flag)
  - Dry-run planning mode

Usage:
    python -m mvt.edt.run_edt_multicore                    # Small config, auto-detect cores
    python -m mvt.edt.run_edt_multicore --config medium   # Medium config
    python -m mvt.edt.run_edt_multicore --cores 4         # Force 4 cores
    python -m mvt.edt.run_edt_multicore --device cuda     # GPU training
    python -m mvt.edt.run_edt_multicore --dry-run         # Print plan, don't train
    python -m mvt.edt.run_edt_multicore --resume          # Resume from last checkpoint
    python -m mvt.edt.run_edt_multicore --kuramoto        # Enable Kuramoto-Metric coupling
"""

from __future__ import annotations

import os
import sys
import time
import math
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import multiprocessing as mp
from multiprocessing.pool import Pool
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

from .moe_model import MoEMVT, MoEMVTConfig
from .edt_pipeline import (
    EDTConfig,
    generate_hidden_states,
    generate_synthetic_corpus,
    phase2a_attention,
    phase2b_embedding,
    phase3_joint,
    _save,
    _format_time,
)


# ===========================================================================
# GPU Detection
# ===========================================================================

def get_device(device_str: str = "auto") -> str:
    """Detect best available device."""
    if device_str != "auto":
        return device_str
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  GPU detected: {name} ({mem_gb:.1f} GB)")
        return "cuda"
    return "cpu"


# ===========================================================================
# Checkpoint Resume
# ===========================================================================

def find_latest_checkpoint(save_dir: str) -> Optional[str]:
    """Find the latest checkpoint to resume from."""
    phase_order = [
        "mvt_edt_final.pt",     # Fully trained — nothing to resume
        "after_phase3.pt",
        "after_phase2b.pt",
        "after_phase2a.pt",
        "after_phase1.pt",
    ]
    for ckpt_name in phase_order:
        path = os.path.join(save_dir, ckpt_name)
        if os.path.exists(path):
            return path
    return None


def detect_resume_phase(save_dir: str) -> Optional[str]:
    """Detect which phase to resume from."""
    checkpoints = {
        "after_phase1.pt": "phase2a",
        "after_phase2a.pt": "phase2b",
        "after_phase2b.pt": "phase3",
    }
    for ckpt_name, next_phase in checkpoints.items():
        if os.path.exists(os.path.join(save_dir, ckpt_name)):
            return next_phase
    return None


# ===========================================================================
# Multi-core Phase 1
# ===========================================================================

def _train_single_expert(
    expert_state_dict: dict,
    layer_idx: int,
    expert_idx: int,
    hidden_bank_np: np.ndarray,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    d_model: int,
    d_ff: int,
    seed: int,
    device: str = "cpu",
) -> Tuple[int, int, float, float, dict]:
    """
    Train a single expert in an isolated process.

    Each process creates its own expert, loads the state dict, trains on
    the hidden bank (passed as numpy array to avoid pickle issues with
    tensors and shared memory), and returns the updated state dict.

    Supports both CPU and CUDA devices. When device='cuda', the expert
    is moved to GPU for faster training. Hidden bank is always passed
    as numpy (for safe pickling across processes).

    Returns:
        (layer_idx, expert_idx, avg_loss, training_time, updated_state_dict)
    """
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed + layer_idx * 1000 + expert_idx)

    # Reconstruct expert in this process
    from mvt.edt.moe_model import TopoExpert

    expert = TopoExpert(d_model, d_ff)
    state_tensors = {k: torch.from_numpy(v) for k, v in expert_state_dict.items()}
    expert.load_state_dict(state_tensors)
    expert.to(device)
    expert.train()

    opt = torch.optim.AdamW(expert.parameters(), lr=lr, weight_decay=weight_decay)

    hidden_bank = torch.from_numpy(hidden_bank_np).to(device)
    n_samples = len(hidden_bank)

    loss_sum = 0.0
    t0 = time.time()

    for step in range(steps):
        batch_idx = torch.randint(0, n_samples - 1, (batch_size,))
        h_in = hidden_bank[batch_idx]
        target_idx = (batch_idx + 1) % n_samples
        h_target = hidden_bank[target_idx]

        opt.zero_grad()
        h_out = expert(h_in)
        loss = F.mse_loss(h_out, h_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(expert.parameters(), grad_clip)
        opt.step()

        loss_sum += loss.item()

    t1 = time.time()
    avg_loss = loss_sum / steps

    updated_state = {k: v.cpu().numpy() for k, v in expert.state_dict().items()}

    return (layer_idx, expert_idx, avg_loss, t1 - t0, updated_state)


def phase1_experts_multicore(
    model: MoEMVT,
    hidden_bank: torch.Tensor,
    config: EDTConfig,
    n_workers: int = None,
) -> Dict[str, float]:
    """
    Phase 1 — Parallel expert training across CPU cores.

    Distributes all experts across n_workers processes. Each process trains
    its assigned experts and returns updated weights. The main process
    collects results and updates the model in-place.

    Speedup: ~n_workers for Phase 1 (embarrassingly parallel).
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    t0 = time.time()
    d_model = model.config.d_model
    d_ff = model.config.d_ff
    n_layers = len(model.blocks)

    # Build task list: (layer_idx, expert_idx) for ALL experts
    tasks = []
    for layer_idx in range(n_layers):
        n_experts = model.blocks[layer_idx].moe.n_experts
        for expert_idx in range(n_experts):
            tasks.append((layer_idx, expert_idx))

    total_experts = len(tasks)
    hidden_np = hidden_bank.numpy()

    print(f"\n  Phase 1 Multi-Core: {total_experts} experts, {n_workers} workers")
    print(f"  Distributing {len(tasks)} tasks across {n_workers} processes...", flush=True)

    # GPU detection for Phase 1 workers
    device_per_worker = "cpu"
    n_gpus = 0
    if config.device == "cuda" and torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_gpus >= n_workers:
            device_per_worker = None  # Will be set per-worker below
        else:
            print(f"  Note: {n_gpus} GPUs < {n_workers} workers, Phase 1 stays on CPU")

    # Prepare all worker args (each worker gets its own hidden bank copy via spawn)
    worker_args = []
    for i, (li, ei) in enumerate(tasks):
        sd = model.blocks[li].moe.experts[ei].state_dict()
        sd_np = {k: v.cpu().numpy() for k, v in sd.items()}
        if device_per_worker is None:
            dev = f"cuda:{i % n_gpus}"
        else:
            dev = device_per_worker
        worker_args.append((
            sd_np, li, ei, hidden_np.copy(),
            config.phase1_steps_per_expert, config.phase1_batch_size,
            config.phase1_lr, config.phase1_weight_decay, config.grad_clip,
            d_model, d_ff,
            config.seed if hasattr(config, 'seed') else 42,
            dev,
        ))
    del hidden_np

    # Launch parallel training
    completed = 0
    results = []
    t_parallel_start = time.time()

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.starmap(_train_single_expert, worker_args)):
            layer_idx, expert_idx, avg_loss, train_time, updated_sd = result

            # Update model weights in-place
            state_cpu = {k: torch.from_numpy(v) for k, v in updated_sd.items()}
            model.blocks[layer_idx].moe.experts[expert_idx].load_state_dict(state_cpu)

            completed += 1
            elapsed = time.time() - t_parallel_start
            rate = completed / elapsed

            if completed % max(1, total_experts // 10) == 0 or completed == total_experts:
                eta = (total_experts - completed) / rate if rate > 0 else 0
                print(f"  [{completed:>4d}/{total_experts}] "
                      f"loss={avg_loss:.4f} "
                      f"rate={rate:.1f} experts/s "
                      f"ETA={_format_time(eta)}", flush=True)

            results.append((layer_idx, expert_idx, avg_loss))

    total_time = time.time() - t0
    avg_loss = np.mean([r[2] for r in results])

    stats = {
        "time": total_time,
        "avg_loss": avg_loss,
        "n_experts": total_experts,
        "n_workers": n_workers,
    }
    print(f"\n  Phase 1 Multi-Core complete in {_format_time(total_time)}")
    print(f"    {total_experts} experts trained, avg loss = {avg_loss:.4f}")

    return stats


# ===========================================================================
# Full Pipeline (Multi-Core Phase 1 + Sequential Phases 2-3)
# ===========================================================================

def run_edt_multicore(
    model: MoEMVT,
    corpus_tokens: torch.Tensor,
    config: Optional[EDTConfig] = None,
    n_workers: Optional[int] = None,
    verbose: bool = True,
    resume: bool = False,
) -> Dict[str, object]:
    """
    Full EDT pipeline with multi-core Phase 1.

    Phase 1: PARALLEL across all CPU cores (ProcessPoolExecutor)
    Phase 2a: Sequential (attention — fast anyway)
    Phase 2b: Sequential (embedding — memory-bound)
    Phase 3: Sequential (joint — needs full model)

    Args:
        resume: If True, skip completed phases and resume from checkpoint.
    """
    if config is None:
        config = EDTConfig()

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    model = model.to(config.device)
    model.train()

    # Resume logic: detect which phase to start from
    start_phase = "phase1"
    if resume:
        phase_to_ckpt = {"phase2a": "after_phase1.pt", "phase2b": "after_phase2a.pt", "phase3": "after_phase2b.pt"}
        for phase, ckpt_name in phase_to_ckpt.items():
            ckpt_full = os.path.join(config.save_dir, ckpt_name)
            if os.path.exists(ckpt_full):
                if verbose:
                    print(f"  Resuming from {ckpt_name} → starting at {phase}")
                model.load_state_dict(torch.load(ckpt_full, map_location=config.device, weights_only=True))
                start_phase = phase
                break

    # Model stats
    total_p, active_p = model.count_params()
    model_mb = sum(p.numel() * 4 for p in model.parameters()) / 1e6

    if verbose:
        print("\n" + "=" * 64)
        print("  EXPERT DECOUPLED TRAINING — MULTI-CORE CPU")
        print("=" * 64)
        print(f"  Total params      : {total_p:,} ({total_p/1e6:.1f}M)")
        print(f"  Active params     : {active_p:,} ({active_p/1e6:.1f}M)")
        print(f"  Sparsity          : {1 - active_p/total_p:.1%}")
        print(f"  Model size (f32)  : {model_mb:.1f} MB")
        print(f"  Corpus            : {len(corpus_tokens):,} tokens")
        print(f"  Layers            : {model.config.n_layers}")
        print(f"  Experts/layer     : {model.config.n_experts}")
        print(f"  Top-k             : {model.config.top_k}")
        print(f"  CPU cores         : {n_workers}")
        print(f"  Device            : {config.device}")

    all_stats = {}
    t_total = time.time()
    hidden_bank = None

    # ---- Phase 1: Experts (MULTI-CORE) ----
    if start_phase == "phase1":
        hidden_bank = generate_hidden_states(model, corpus_tokens, config)
        stats1 = phase1_experts_multicore(model, hidden_bank, config, n_workers=n_workers)
        all_stats["phase1"] = stats1
        ckpt1 = os.path.join(config.save_dir, "after_phase1.pt")
        _save(model, ckpt1)
        start_phase = "phase2a"  # Move to next
    else:
        if verbose:
            print(f"  Skipping Phase 1 (already completed)")

    # ---- Phase 2a: Attention (sequential — fast) ----
    if start_phase == "phase2a":
        if hidden_bank is None:
            hidden_bank = generate_hidden_states(model, corpus_tokens, config)
        stats2a = phase2a_attention(model, hidden_bank, config)
        all_stats["phase2a"] = stats2a
        ckpt2a = os.path.join(config.save_dir, "after_phase2a.pt")
        _save(model, ckpt2a)
        start_phase = "phase2b"
    else:
        if verbose:
            print(f"  Skipping Phase 2a (already completed)")

    # Free hidden bank
    if hidden_bank is not None:
        del hidden_bank

    # ---- Phase 2b: Embedding (sequential) ----
    if start_phase == "phase2b":
        stats2b = phase2b_embedding(model, corpus_tokens, config)
        all_stats["phase2b"] = stats2b
        ckpt2b = os.path.join(config.save_dir, "after_phase2b.pt")
        _save(model, ckpt2b)
        start_phase = "phase3"
    else:
        if verbose:
            print(f"  Skipping Phase 2b (already completed)")

    # ---- Phase 3: Joint (sequential + PGSG) ----
    if start_phase == "phase3":
        stats3 = phase3_joint(model, corpus_tokens, config)
        all_stats["phase3"] = stats3
        ckpt3 = os.path.join(config.save_dir, "mvt_edt_final.pt")
        _save(model, ckpt3)
    else:
        if verbose:
            print(f"  Skipping Phase 3 (already completed)")

    # If fully trained already
    if "phase3" not in all_stats and os.path.exists(os.path.join(config.save_dir, "mvt_edt_final.pt")):
        if verbose:
            print(f"  Model already fully trained! Loading final checkpoint.")
        model.load_state_dict(torch.load(
            os.path.join(config.save_dir, "mvt_edt_final.pt"),
            map_location=config.device, weights_only=True
        ))

    total_time = time.time() - t_total
    all_stats["total_time"] = total_time

    if verbose:
        speedup = all_stats.get("phase1_sequential_time", 0) / max(stats1["time"], 0.01)
        print("\n" + "=" * 64)
        print("  SUMMARY — MULTI-CORE EDT COMPLETE")
        print("=" * 64)
        print(f"  Phase 1 (experts, {n_workers} cores) : {_format_time(stats1['time'])}")
        if speedup > 1:
            print(f"    Phase 1 speedup vs sequential    : {speedup:.1f}x")
        print(f"  Phase 2a (attention)                : {_format_time(stats2a['time'])}")
        print(f"  Phase 2b (embedding)                : {_format_time(stats2b['time'])}")
        print(f"  Phase 3 (joint PGSG)                : {_format_time(stats3['time'])}")
        print(f"  {'─' * 40}")
        print(f"  TOTAL                                : {_format_time(total_time)}")
        print(f"  Checkpoint                           : {ckpt3}")

    return all_stats


# ===========================================================================
# Config Presets
# ===========================================================================

PRESETS = {
    "tiny": {
        "model": dict(
            vocab_size=1000, d_model=64, n_layers=2, n_experts=4,
            top_k=1, d_ff=128, max_seq_len=32,
        ),
        "edt": dict(
            phase1_steps_per_expert=30, phase1_hidden_samples=500,
            phase1_batch_size=16, phase2a_steps_per_layer=50,
            phase2a_batch_size=8, phase2b_n_tokens=50_000,
            phase2b_batch_size=32, phase2b_seq_len=16,
            phase3_n_tokens=20_000, phase3_batch_size=4,
            phase3_seq_len=16, phase3_n_active_layers=1,
        ),
        "desc": "Tiny — smoke test (~6 min)",
    },
    "small": {
        "model": dict(
            vocab_size=4000, d_model=128, n_layers=4, n_experts=8,
            top_k=2, d_ff=256, max_seq_len=64,
        ),
        "edt": dict(
            phase1_steps_per_expert=150, phase1_hidden_samples=2000,
            phase1_batch_size=32, phase2a_steps_per_layer=300,
            phase2a_batch_size=16, phase2b_n_tokens=2_000_000,
            phase2b_batch_size=64, phase2b_seq_len=64,
            phase3_n_tokens=500_000, phase3_batch_size=4,
            phase3_seq_len=32, phase3_n_active_layers=2,
        ),
        "desc": "Small — CPU viable (~25 min with 2 cores)",
    },
    "medium": {
        "model": dict(
            vocab_size=8000, d_model=256, n_layers=6, n_experts=16,
            top_k=2, d_ff=512, max_seq_len=128,
        ),
        "edt": dict(
            phase1_steps_per_expert=200, phase1_hidden_samples=3000,
            phase1_batch_size=32, phase2a_steps_per_layer=400,
            phase2a_batch_size=16, phase2b_n_tokens=5_000_000,
            phase2b_batch_size=64, phase2b_seq_len=64,
            phase3_n_tokens=1_000_000, phase3_batch_size=4,
            phase3_seq_len=32, phase3_n_active_layers=3,
        ),
        "desc": "Medium — 29M params (~3-5h with 4 cores)",
    },
    "large": {
        "model": dict(
            vocab_size=16000, d_model=256, n_layers=8, n_experts=16,
            top_k=2, d_ff=512, max_seq_len=128,
        ),
        "edt": dict(
            phase1_steps_per_expert=300, phase1_hidden_samples=5000,
            phase1_batch_size=32, phase2a_steps_per_layer=500,
            phase2a_batch_size=16, phase2b_n_tokens=10_000_000,
            phase2b_batch_size=64, phase2b_seq_len=64,
            phase3_n_tokens=2_000_000, phase3_batch_size=4,
            phase3_seq_len=32, phase3_n_active_layers=3,
        ),
        "desc": "Large — 40M params (~8-15h with 8 cores)",
    },
}


# ===========================================================================
# 1B Parameter Config
# ===========================================================================

def get_1b_config() -> Tuple[MoEMVTConfig, EDTConfig]:
    """
    Configuration for ~1B total parameters.

    MoE-MVT achieves 1B total params with massive sparsity:
    - 12 layers × 128 experts × d_ff=1024 = 1.2B expert params
    - Only 2 experts active per token → ~20M active params (1.6% sparsity)
    - d_model=256 keeps Christoffel cost manageable

    Chinchilla scaling for MVT: D/N ≈ 5 samples/param
    → 1B × 5 = 5B training samples (trajectory steps)
    → EDT reduces this dramatically via expert decoupling
    """
    model_cfg = MoEMVTConfig(
        vocab_size=32000,
        d_model=256,
        n_layers=12,
        n_experts=128,
        top_k=2,
        d_ff=1024,
        max_seq_len=256,
        aux_loss_weight=0.01,
        router_temperature=1.0,
        embedding_dropout=0.1,
    )

    total, active = MoEMVT(model_cfg).count_params()

    # EDT config for 1B — scaled up
    edt_cfg = EDTConfig(
        # Phase 1: 1536 experts, more steps needed
        phase1_steps_per_expert=500,
        phase1_hidden_samples=10000,
        phase1_batch_size=64,
        phase1_lr=1e-3,
        phase1_weight_decay=0.01,

        # Phase 2a: 12 attention layers
        phase2a_steps_per_layer=600,
        phase2a_batch_size=16,

        # Phase 2b: embedding
        phase2b_n_tokens=50_000_000,     # 50M tokens
        phase2b_batch_size=128,
        phase2b_seq_len=128,

        # Phase 3: joint alignment
        phase3_n_tokens=20_000_000,       # 20M tokens
        phase3_batch_size=8,
        phase3_seq_len=64,
        phase3_n_active_layers=4,         # 4/12 layers per step
        phase3_lr=3e-4,

        seq_len=64,
        grad_clip=1.0,
        device="cpu",
    )

    return model_cfg, edt_cfg


def print_training_plan(model_cfg: MoEMVTConfig, edt_cfg: EDTConfig, n_cores: int):
    """Print a detailed training plan with time estimates."""
    model = MoEMVT(model_cfg)
    total, active = model.count_params()
    total_experts = model_cfg.n_layers * model_cfg.n_experts
    model_gb = total * 4 / 1e9

    # Estimate Phase 1 time based on benchmarks
    # Benchmark: ~250ms/step/expert for d_model=128
    # Scale: time ∝ d_model² × d_ff
    scale_factor = (model_cfg.d_model / 128) ** 2 * (model_cfg.d_ff / 256)
    single_expert_time = 0.25 * scale_factor * edt_cfg.phase1_steps_per_expert
    p1_sequential = single_expert_time * total_experts
    p1_parallel = p1_sequential / n_cores

    # Phase 2a: ~5ms/step/layer for d_model=128
    p2a_time = 0.005 * scale_factor * edt_cfg.phase2a_steps_per_layer * model_cfg.n_layers

    # Phase 2b: ~4000 tok/s for embedding
    p2b_time = edt_cfg.phase2b_n_tokens / 4000

    # Phase 3: ~500 tok/s for joint (with PGSG)
    p3_time = edt_cfg.phase3_n_tokens / 500

    total_sequential = p1_sequential + p2a_time + p2b_time + p3_time
    total_parallel = p1_parallel + p2a_time + p2b_time + p3_time

    print("\n" + "=" * 70)
    print("  MVT EDT — TRAINING PLAN")
    print("=" * 70)
    print(f"\n  Model Configuration:")
    print(f"    Total parameters  : {total:>15,} ({total/1e9:.2f}B)")
    print(f"    Active per token   : {active:>15,} ({active/1e6:.1f}M)")
    print(f"    Sparsity           : {1 - active/total:>14.1%}")
    print(f"    Model size (f32)  : {model_gb:>12.1f} GB")
    print(f"    Layers × Experts  : {model_cfg.n_layers} × {model_cfg.n_experts} = {total_experts} experts")
    print(f"    d_model × d_ff    : {model_cfg.d_model} × {model_cfg.d_ff}")
    print(f"    Vocab size        : {model_cfg.vocab_size:,}")
    print(f"    Top-k             : {model_cfg.top_k}")

    print(f"\n  EDT Configuration:")
    print(f"    Phase 1 steps/exp : {edt_cfg.phase1_steps_per_expert}")
    print(f"    Phase 1 hidden    : {edt_cfg.phase1_hidden_samples:,}")
    print(f"    Phase 2a steps/lay: {edt_cfg.phase2a_steps_per_layer}")
    print(f"    Phase 2b tokens   : {edt_cfg.phase2b_n_tokens/1e6:.0f}M")
    print(f"    Phase 3 tokens    : {edt_cfg.phase3_n_tokens/1e6:.0f}M")
    print(f"    Phase 3 PGSG      : {edt_cfg.phase3_n_active_layers}/{model_cfg.n_layers} layers active")

    print(f"\n  Time Estimates ({n_cores} CPU cores):")
    print(f"    Phase 1 (parallel) : {_format_time(p1_parallel):>12s}  (sequential: {_format_time(p1_sequential)})")
    print(f"    Phase 2a (attn)    : {_format_time(p2a_time):>12s}")
    print(f"    Phase 2b (embed)   : {_format_time(p2b_time):>12s}")
    print(f"    Phase 3 (joint)    : {_format_time(p3_time):>12s}")
    print(f"    {'─' * 40}")
    print(f"    TOTAL (parallel)   : {_format_time(total_parallel):>12s}")
    print(f"    Speedup vs seq.    : {total_sequential/total_parallel:>12.1f}x")

    print(f"\n  Chinchilla Scaling (MVT-adapted):")
    mvt_d_over_n = 5  # Middle of 3-7 range
    optimal_samples = total * mvt_d_over_n
    print(f"    D/N ratio         : {mvt_d_over_n}")
    print(f"    Optimal samples   : {optimal_samples/1e9:.1f}B trajectory steps")
    print(f"    EDT tokens        : {(edt_cfg.phase2b_n_tokens + edt_cfg.phase3_n_tokens)/1e6:.0f}M")
    print(f"    Efficiency vs Chinchilla: {(edt_cfg.phase2b_n_tokens + edt_cfg.phase3_n_tokens) / optimal_samples * 100:.1f}%")

    print(f"\n  Memory Requirements:")
    print(f"    Model (f32)       : {model_gb:.1f} GB")
    print(f"    Hidden bank       : {edt_cfg.phase1_hidden_samples * model_cfg.max_seq_len * model_cfg.d_model * 4 / 1e6:.0f} MB")
    print(f"    Optimizer states  : ~{model_gb * 2:.1f} GB (AdamW)")
    print(f"    Recommended RAM   : {model_gb * 4 + 2:.0f} GB")

    print("=" * 70)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MVT EDT Multi-Core Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  tiny    — Smoke test (~6 min)
  small   — CPU viable, 2.9M params (~25 min with 2 cores)
  medium  — 29M params (~3-5h with 4 cores)
  large   — 40M params (~8-15h with 8 cores)
  1b      — ~1B params (plan only, needs GPU cluster)

Examples:
  python -m mvt.edt.run_edt_multicore --config small
  python -m mvt.edt.run_edt_multicore --config small --cores 4
  python -m mvt.edt.run_edt_multicore --config medium --device cuda
  python -m mvt.edt.run_edt_multicore --config large --resume
  python -m mvt.edt.run_edt_multicore --config 1b --dry-run
        """,
    )
    parser.add_argument(
        "--config", "-c",
        choices=["tiny", "small", "medium", "large", "1b"],
        default="small",
        help="Training preset (default: small)",
    )
    parser.add_argument(
        "--cores", "-j",
        type=int,
        default=None,
        help="Number of CPU cores (default: auto-detect)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print training plan and exit (don't train)",
    )
    parser.add_argument(
        "--corpus-size",
        type=int,
        default=None,
        help="Override synthetic corpus size",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device: auto (detect GPU), cpu, or cuda (default: auto)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Override checkpoint save directory",
    )

    args = parser.parse_args()

    # Device setup
    device = get_device(args.device)
    print(f"  Device: {device}")

    n_cores = args.cores or os.cpu_count() or 1

    # Save dir override
    if args.save_dir:
        pass  # Will be set per-preset below

    if args.config == "1b":
        # 1B is for planning only — too large for single CPU machine
        model_cfg, edt_cfg = get_1b_config()
        print_training_plan(model_cfg, edt_cfg, n_cores)

        if not args.dry_run:
            print("\n  WARNING: 1B training requires a GPU cluster or multi-node CPU.")
            print("  Use --dry-run to see the plan. For actual training, see TRAINING_1B.md")
            print("  Starting with 'small' preset instead...")
            args.config = "small"

    if args.config != "1b":
        preset = PRESETS[args.config]
        model_cfg = MoEMVTConfig(**preset["model"])
        edt_cfg = EDTConfig(**preset["edt"])
        edt_cfg.device = device  # Apply detected device
        if args.save_dir:
            edt_cfg.save_dir = args.save_dir

        print(f"\n  Preset: {preset['desc']}")
        print_training_plan(model_cfg, edt_cfg, n_cores)

    if args.dry_run:
        print("\n  Dry run — exiting without training.")
        return

    # ---- Check resume ----
    if args.resume and args.config != "1b":
        latest_ckpt = find_latest_checkpoint(edt_cfg.save_dir)
        if latest_ckpt:
            print(f"  Found checkpoint: {latest_ckpt}")
        else:
            print(f"  No checkpoint found in {edt_cfg.save_dir}, starting fresh.")
            args.resume = False

    # ---- Actually train ----
    corpus_size = args.corpus_size or max(50_000, model_cfg.vocab_size * 50)
    corpus = generate_synthetic_corpus(vocab_size=model_cfg.vocab_size, length=corpus_size)

    stats = run_edt_multicore(
        MoEMVT(model_cfg),
        corpus,
        edt_cfg,
        n_workers=n_cores,
        verbose=True,
        resume=args.resume,
    )

    # Save stats
    stats_path = os.path.join(edt_cfg.save_dir, "edt_stats.json")
    os.makedirs(edt_cfg.save_dir, exist_ok=True)
    stats_serializable = {k: float(v) if isinstance(v, (int, float)) else str(v)
                          for k, v in stats.items()}
    with open(stats_path, "w") as f:
        json.dump(stats_serializable, f, indent=2)
    print(f"\n  Stats saved: {stats_path}")


if __name__ == "__main__":
    main()
