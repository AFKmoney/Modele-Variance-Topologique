"""
EDT — Entraînement MoE-MVT sur CPU (version complète).
"""
import sys
sys.path.insert(0, '/home/z/my-project/scripts')

import torch
import time
from mvt.edt import MoEMVT, MoEMVTConfig, EDTConfig, run_edt, generate_synthetic_corpus

def main():
    torch.set_num_threads(2)

    print("=" * 64)
    print("  EDT — Entraînement MoE-MVT sur CPU")
    print("  Expert Decoupled Training Pipeline")
    print("=" * 64)

    # === Modèle ===
    model_cfg = MoEMVTConfig(
        vocab_size=4000,
        d_model=128,
        n_layers=4,
        n_experts=8,
        top_k=2,
        d_ff=256,
        max_seq_len=64,
    )

    # === EDT (optimisé CPU) ===
    edt_cfg = EDTConfig(
        phase1_steps_per_expert=50,
        phase1_batch_size=16,
        phase1_lr=1e-3,
        phase1_hidden_samples=500,
        phase2a_steps_per_layer=100,
        phase2a_batch_size=8,
        phase2a_lr=1e-3,
        phase2b_n_tokens=100_000,
        phase2b_batch_size=32,
        phase2b_seq_len=16,
        phase2b_lr=1e-3,
        phase3_n_tokens=50_000,
        phase3_batch_size=2,
        phase3_seq_len=8,
        phase3_lr=3e-4,
        phase3_n_active_layers=2,
        save_dir="/home/z/my-project/download/mvt_checkpoints",
    )

    model = MoEMVT(model_cfg)
    total_p, active_p = model.count_params()
    print(f"\n  Modèle : {total_p:,} params total, {active_p:,} actifs/token")
    print(f"  Sparsité : {1 - active_p/total_p:.1%}")

    # === Corpus ===
    print("\n  Génération du corpus (50K tokens, bigram structure)...")
    corpus = generate_synthetic_corpus(vocab_size=model_cfg.vocab_size, length=50000, seed=42)
    print(f"  Corpus : {len(corpus):,} tokens")

    # === EDT ===
    t_start = time.time()
    stats = run_edt(model, corpus, edt_cfg, verbose=True)
    total_time = time.time() - t_start

    # === Génération ===
    print("\n" + "=" * 64)
    print("  TEST DE GÉNÉRATION")
    print("=" * 64)

    model.eval()
    with torch.no_grad():
        prompt_tokens = corpus[:5]
        print(f"  Prompt token ids : {prompt_tokens.tolist()}")

        generated = prompt_tokens.tolist()
        input_seq = prompt_tokens.unsqueeze(0)
        for _ in range(20):
            logits, _ = model(input_seq)
            next_tok = logits[0, -1].argmax().item()
            generated.append(next_tok)
            input_seq = torch.cat([input_seq, torch.tensor([[next_tok]])], dim=1)

        print(f"  Séquence générée ({len(generated)} tokens) : {generated}")

    # === Résumé ===
    print("\n" + "=" * 64)
    print("  RÉSUMÉ FINAL EDT")
    print("=" * 64)
    print(f"  Temps total        : {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Phase 1 (Experts)  : {stats['phase1']['time']:.1f}s")
    print(f"  Phase 2a (Attn)    : {stats['phase2a']['time']:.1f}s")
    print(f"  Phase 2b (Embed)   : {stats['phase2b']['time']:.1f}s  loss={stats['phase2b']['avg_loss']:.4f}")
    print(f"  Phase 3 (Joint)    : {stats['phase3']['time']:.1f}s  loss={stats['phase3']['avg_loss']:.4f}")
    print(f"  Embed tok/s        : {stats['phase2b'].get('tok_per_sec', 0):.0f}")
    print(f"  Joint tok/s        : {stats['phase3'].get('tok_per_sec', 0):.0f}")

    # Vérif checkpoint
    ckpt = "/home/z/my-project/download/mvt_checkpoints/mvt_edt_final.pt"
    model2 = MoEMVT(model_cfg)
    model2.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    print(f"\n  Checkpoint chargé : {ckpt} ✓")

if __name__ == "__main__":
    main()
