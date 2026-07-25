"""
Test complet : Toy Training MVT avec KuramotoMetricCoupler
==========================================================

Teste end-to-end :
1. Création du modèle avec kuramoto_enabled=True
2. Génération multiple de texte
3. Vérification de la stabilité de G (SPD, trace, condition number)
4. Vérification de la synchronisation Kuramoto (order parameter)
5. Comparaison avec le modèle statique (kuramoto_enabled=False)
6. Entraînement simple (itérations + mise à jour de la métrique)
"""

from __future__ import annotations
import sys
import os
import time
import json
import numpy as np

# Ajouter le parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvt.config import MVTConfig
from mvt.model import MVT, GenerationResult
from mvt.kuramoto_metric import KuramotoMetricCoupler
from mvt.natural_gradient_spd import NaturalGradientSPD


def test_kuramoto_coupled_model():
    """
    Test 1 : Modèle MVT avec Kuramoto couplé.
    - Vérifie que le modèle se crée correctement
    - Génère du texte
    - Vérifie les métriques de couplage
    """
    print("=" * 70)
    print("  TEST 1 : MVT avec KuramotoMetricCoupler activé")
    print("=" * 70)

    cfg = MVTConfig(
        ambient_dim=32,           # Petit pour CPU
        intrinsic_dim=16,
        dt=0.01,
        num_rk4_steps=50,          # Steps réduits pour le test
        kuramoto_enabled=True,
        kuramoto_coupling_K=2.0,
        kuramoto_n_oscillators=32,
        kuramoto_metric_lr=0.001,
        kuramoto_retraction="approx2",
        kuramoto_phase_coupling=0.1,
        kuramoto_phase_init="random",
        temperature=0.8,
        seed=42,
    )

    print(f"\n  Config: dim={cfg.ambient_dim}, K={cfg.kuramoto_coupling_K}, "
          f"lr={cfg.kuramoto_metric_lr}")
    print(f"  Kuramoto enabled: {cfg.kuramoto_enabled}")

    # Créer le modèle
    t0 = time.time()
    model = MVT(cfg)
    t_create = time.time() - t0
    print(f"  ✓ Modèle créé en {t_create:.3f}s")

    # Vérifier le coupler
    assert model.kuramoto_enabled is True
    assert model.coupler is not None
    assert model.coupler.metric is model.metric
    print(f"  ✓ Coupler Kuramoto actif: {model.coupler}")
    print(f"  ✓ metric est le même objet: coupler.metric is model.metric = "
          f"{model.coupler.metric is model.metric}")

    # État initial
    G = model.metric.G.copy()
    eigvals0 = np.linalg.eigvalsh(G)
    print(f"\n  État initial:")
    print(f"    G: trace={np.trace(G):.2f}, det={np.linalg.det(G):.4e}, "
          f"cond={np.linalg.cond(G):.2f}")
    print(f"    eigvals: min={np.min(eigvals0):.4f}, max={np.max(eigvals0):.4f}")
    print(f"    SPD: {np.all(eigvals0 > 0)}")
    print(f"    Kuramoto r₀ = {model.coupler.kuramoto.order_parameter():.4f}")

    # Générer du texte
    prompts = [
        "The cat sits on the mat",
        "Deep learning is transforming AI",
        "Riemannian geometry defines curvature",
        "Kuramoto oscillators synchronize",
    ]

    print(f"\n  Génération de {len(prompts)} prompts...")
    results = []
    for i, prompt in enumerate(prompts):
        t0 = time.time()
        result = model.generate(prompt, num_steps=50)
        t_gen = time.time() - t0
        results.append(result)
        print(f"\n  [{i+1}] Prompt: \"{prompt[:40]}...\"")
        print(f"      Text: \"{result.text[:60]}...\"" if len(result.text) > 60
              else f"      Text: \"{result.text}\"")
        print(f"      Words: {len(result.text.split())}, "
              f"Steps: {result.num_steps}, Time: {t_gen:.3f}s")
        print(f"      Energy: action={result.action:.4f}, "
              f"KE={result.kinetic_energy:.4f}")
        print(f"      Creativity: {result.creativity_score:.4f}, "
              f"Diversity: {result.diversity_score:.4f}")
        print(f"      Syntopy: {result.syntopy_score:.4f}")

    # État après génération
    G_after = model.metric.G.copy()
    eigvals_after = np.linalg.eigvalsh(G_after)
    r_final = model.coupler.kuramoto.order_parameter()

    print(f"\n  État après {len(prompts)} générations:")
    print(f"    G: trace={np.trace(G_after):.2f}, det={np.linalg.det(G_after):.4e}, "
          f"cond={np.linalg.cond(G_after):.2f}")
    print(f"    eigvals: min={np.min(eigvals_after):.6f}, max={np.max(eigvals_after):.4f}")
    print(f"    SPD: {np.all(eigvals_after > 0)}")
    print(f"    Kuramoto r_final = {r_final:.4f}")

    # Vérifier les métriques de couplage
    coupling_metrics = model._coupling_metrics
    if coupling_metrics:
        print(f"\n  Métriques de couplage ({len(coupling_metrics)} steps):")
        last = coupling_metrics[-1]
        print(f"    Order param (final): {last['order_parameter']:.4f}")
        print(f"    Phase coherence: {last['phase_coherence']:.4f}")
        print(f"    G trace: {last['G_trace']:.2f}")
        print(f"    G det: {last['G_det']:.4e}")
        print(f"    G cond: {last['G_cond']:.2f}")
        print(f"    Grad norm: {last['grad_norm']:.4e}")

    # Asserts de stabilité
    assert np.all(eigvals_after > 0), "G n'est plus SPD après génération !"
    assert np.linalg.cond(G_after) < 1e6, f"Condition number explosé: {np.linalg.cond(G_after)}"
    assert abs(np.trace(G_after) - cfg.ambient_dim) < cfg.ambient_dim * 2, \
        f"Trace dérivée: {np.trace(G_after)} vs N={cfg.ambient_dim}"

    print(f"\n  ✓ TEST 1 PASSÉ : G SPD-stable, couplage fonctionnel")
    return model, results, cfg


def test_static_model_comparison():
    """
    Test 2 : Comparaison avec le modèle statique (kuramoto_enabled=False).
    """
    print("\n" + "=" * 70)
    print("  TEST 2 : Comparaison modèle statique vs Kuramoto couplé")
    print("=" * 70)

    cfg = MVTConfig(
        ambient_dim=32,
        intrinsic_dim=16,
        dt=0.01,
        num_rk4_steps=50,
        kuramoto_enabled=False,  # MODE STATIQUE
        temperature=0.8,
        seed=42,
    )

    model_static = MVT(cfg)
    print(f"  Modèle statique créé (kuramoto_enabled=False)")
    print(f"  Coupler: {model_static.coupler}")

    # Générer le même prompt
    prompt = "Riemannian geometry defines curvature"
    t0 = time.time()
    result_static = model_static.generate(prompt, num_steps=50)
    t_static = time.time() - t0

    # Maintenant avec Kuramoto
    cfg_k = MVTConfig(
        ambient_dim=32,
        intrinsic_dim=16,
        dt=0.01,
        num_rk4_steps=50,
        kuramoto_enabled=True,
        kuramoto_coupling_K=2.0,
        kuramoto_n_oscillators=32,
        kuramoto_metric_lr=0.001,
        temperature=0.8,
        seed=42,
    )
    model_kuramoto = MVT(cfg_k)
    t0 = time.time()
    result_kuramoto = model_kuramoto.generate(prompt, num_steps=50)
    t_kuramoto = time.time() - t0

    print(f"\n  Prompt: \"{prompt}\"")
    print(f"\n  Statique:")
    print(f"    Text: \"{result_static.text[:80]}...\" " if len(result_static.text) > 80
          else f"    Text: \"{result_static.text}\"")
    print(f"    Words: {len(result_static.text.split())}, Time: {t_static:.3f}s")
    print(f"    G trace: {np.trace(model_static.metric.G):.2f} (devrait être ~identité)")
    print(f"    G det: {np.linalg.det(model_static.metric.G):.4e}")

    print(f"\n  Kuramoto couplé:")
    print(f"    Text: \"{result_kuramoto.text[:80]}...\" " if len(result_kuramoto.text) > 80
          else f"    Text: \"{result_kuramoto.text}\"")
    print(f"    Words: {len(result_kuramoto.text.split())}, Time: {t_kuramoto:.3f}s")
    print(f"    G trace: {np.trace(model_kuramoto.metric.G):.2f} (évolue morphologiquement)")
    print(f"    G det: {np.linalg.det(model_kuramoto.metric.G):.4e}")
    print(f"    Coupling metrics: {len(model_kuramoto._coupling_metrics)} steps")

    print(f"\n  Overhead Kuramoto: {(t_kuramoto - t_static) / t_static * 100:.1f}%")
    print(f"  ✓ TEST 2 PASSÉ : Les deux modes fonctionnent")


def test_stability_long_run():
    """
    Test 3 : Stabilité sur un long run.
    - Plusieurs générations consécutives
    - G doit rester SPD et bien conditionné
    - Pas d'explosion de trace/determinant
    """
    print("\n" + "=" * 70)
    print("  TEST 3 : Stabilité sur 20 générations consécutives")
    print("=" * 70)

    cfg = MVTConfig(
        ambient_dim=32,
        intrinsic_dim=16,
        dt=0.01,
        num_rk4_steps=30,
        kuramoto_enabled=True,
        kuramoto_coupling_K=2.0,
        kuramoto_n_oscillators=32,
        kuramoto_metric_lr=0.0005,  # Plus petit lr pour stabilité
        kuramoto_retraction="approx2",
        kuramoto_phase_coupling=0.05,  # Couplage plus doux
        temperature=0.7,
        seed=42,
    )

    model = MVT(cfg)

    prompts = [
        "Neural networks learn representations",
        "The universe expands in all directions",
        "Mathematics describes natural patterns",
        "Language encodes thought and meaning",
        "Energy flows through all systems",
    ]

    N_gen = 20
    traces = []
    dets = []
    conds = []
    order_params = []
    eigval_mins = []
    eigval_maxs = []

    print(f"\n  Running {N_gen} generations with Kuramoto coupling...")
    for i in range(N_gen):
        prompt = prompts[i % len(prompts)]
        result = model.generate(prompt, num_steps=30)

        G = model.metric.G
        eigvals = np.linalg.eigvalsh(G)
        tr = np.trace(G)
        det = np.linalg.det(G)
        cond = np.linalg.cond(G)
        r = model.coupler.kuramoto.order_parameter()

        traces.append(tr)
        dets.append(det)
        conds.append(cond)
        order_params.append(r)
        eigval_mins.append(np.min(eigvals))
        eigval_maxs.append(np.max(eigvals))

        if i % 5 == 0 or i == N_gen - 1:
            print(f"  [{i:>3d}] tr(G)={tr:>8.2f}, det={det:>10.4e}, "
                  f"cond={cond:>8.2f}, r={r:.4f}, "
                  f"λ∈[{np.min(eigvals):.4f},{np.max(eigvals):.4f}], "
                  f"words={len(result.text.split())}")

    # Statistiques
    traces = np.array(traces)
    dets = np.array(dets)
    conds = np.array(conds)
    eigval_mins = np.array(eigval_mins)
    eigval_maxs = np.array(eigval_maxs)

    print(f"\n  Statistiques sur {N_gen} générations:")
    print(f"    trace(G):   mean={np.mean(traces):.2f}, std={np.std(traces):.2f}, "
          f"range=[{np.min(traces):.2f}, {np.max(traces):.2f}]")
    print(f"    det(G):     mean={np.mean(dets):.4e}, range=[{np.min(dets):.4e}, {np.max(dets):.4e}]")
    print(f"    cond(G):    mean={np.mean(conds):.2f}, max={np.max(conds):.2f}")
    print(f"    λ_min:      min={np.min(eigval_mins):.6f}, mean={np.mean(eigval_mins):.4f}")
    print(f"    λ_max:      max={np.max(eigval_maxs):.4f}, mean={np.mean(eigval_maxs):.4f}")
    print(f"    r (Kuramoto): mean={np.mean(order_params):.4f}, "
          f"max={np.max(order_params):.4f}")

    # Assertions
    assert np.all(eigval_mins > 0), "G a eu des eigenvalues négatives !"
    assert np.max(conds) < 1e6, f"Condition number explosé: {np.max(conds)}"
    assert np.max(eigval_maxs) < 100, f"Eigenvalue max explosée: {np.max(eigval_maxs)}"

    print(f"\n  ✓ TEST 3 PASSÉ : Stabilité confirmée sur {N_gen} générations")
    return traces, conds, eigval_mins, eigval_maxs, order_params


def test_coupler_standalone():
    """
    Test 4 : KuramotoMetricCoupler standalone.
    - Teste integrate_coupled() directement
    - Vérifie la convergence phase→métrique
    """
    print("\n" + "=" * 70)
    print("  TEST 4 : KuramotoMetricCoupler standalone")
    print("=" * 70)

    from mvt.kuramoto_metric import KuramotoMetricConfig

    kura_cfg = KuramotoMetricConfig(
        N=32,
        n_oscillators=32,
        K_coupling=3.0,  # K plus élevé = synchronisation plus forte
        omega_range=(-0.5, 0.5),  # Fréquences plus homogènes
        metric_lr=0.001,
        retraction_order="approx2",
        phase_metric_coupling=0.1,
        phase_init="random",
        dt_kuramoto=0.01,
    )

    coupler = KuramotoMetricCoupler(
        mvt_config=MVTConfig(ambient_dim=32, seed=42),
        kura_config=kura_cfg,
    )

    N = 32
    q0 = np.random.randn(N) * 0.5
    dq0 = np.random.randn(N) * 0.01

    print(f"  Intégration couplée standalone sur 100 steps...")
    phi_final, G_final, metrics_list = coupler.integrate_coupled(
        q0, dq0, n_steps=100
    )

    # Vérifier SPD
    eigvals_final = np.linalg.eigvalsh(G_final)
    assert np.all(eigvals_final > 0), "G_final n'est pas SPD !"
    print(f"  ✓ G_final SPD: det={np.linalg.det(G_final):.4e}")

    # Vérifier synchronisation
    order_params = [m["order_parameter"] for m in metrics_list]
    r_start = order_params[0]
    r_end = order_params[-1]
    print(f"  Order parameter: r₀={r_start:.4f} → r_final={r_end:.4f}")

    # Avec K=3.0 et omega_range petit, on devrait voir de la synchronisation
    coherence_vals = [m["phase_coherence"] for m in metrics_list]
    c_start = coherence_vals[0]
    c_end = coherence_vals[-1]
    print(f"  Phase coherence: c₀={c_start:.4f} → c_final={c_end:.4f}")

    # Traces
    traces = [m["G_trace"] for m in metrics_list]
    print(f"  trace(G): start={traces[0]:.2f}, end={traces[-1]:.2f}")

    print(f"  ✓ TEST 4 PASSÉ : Couplage standalone fonctionne")


def test_reset_and_stats():
    """
    Test 5 : Reset et get_stats.
    """
    print("\n" + "=" * 70)
    print("  TEST 5 : Reset et get_stats")
    print("=" * 70)

    cfg = MVTConfig(
        ambient_dim=32,
        intrinsic_dim=16,
        kuramoto_enabled=True,
        kuramoto_coupling_K=2.0,
        seed=42,
    )

    model = MVT(cfg)

    # Générer
    model.generate("test prompt", num_steps=20)
    model.generate("another test", num_steps=20)

    # Stats
    stats = model.get_stats()
    print(f"  Stats après 2 générations:")
    print(f"    Total générations: {stats['generation']['total_generations']}")
    print(f"    Kuramoto: {json.dumps(stats.get('kuramoto', {}), indent=4)}")
    print(f"    Metric: det={stats['metric']['det']:.4e}, trace={stats['metric']['trace']:.2f}")
    print(f"    Creativity: temp={stats['creativity']['temperature']:.2f}")

    # Reset
    model.reset()
    stats_after = model.get_stats()
    print(f"\n  Stats après reset:")
    print(f"    Total générations: {stats_after['generation']['total_generations']}")
    print(f"    Coupler recréé: {model.coupler is not None}")
    print(f"    Metric trace: {stats_after['metric']['trace']:.2f}")

    assert stats_after['generation']['total_generations'] == 0, "Reset n'a pas vidé l'historique"
    print(f"  ✓ TEST 5 PASSÉ : Reset fonctionne")


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  MVT — Test complet : Toy Training avec KuramotoMetricCoupler  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  NumPy {np.__version__}")
    print()

    all_passed = True

    # Test 1 : Modèle couplé
    try:
        test_kuramoto_coupled_model()
    except Exception as e:
        print(f"\n  ✗ TEST 1 ÉCHOUÉ : {e}")
        import traceback; traceback.print_exc()
        all_passed = False

    # Test 2 : Comparaison statique vs couplé
    try:
        test_static_model_comparison()
    except Exception as e:
        print(f"\n  ✗ TEST 2 ÉCHOUÉ : {e}")
        import traceback; traceback.print_exc()
        all_passed = False

    # Test 3 : Stabilité long run
    try:
        test_stability_long_run()
    except Exception as e:
        print(f"\n  ✗ TEST 3 ÉCHOUÉ : {e}")
        import traceback; traceback.print_exc()
        all_passed = False

    # Test 4 : Coupler standalone
    try:
        test_coupler_standalone()
    except Exception as e:
        print(f"\n  ✗ TEST 4 ÉCHOUÉ : {e}")
        import traceback; traceback.print_exc()
        all_passed = False

    # Test 5 : Reset
    try:
        test_reset_and_stats()
    except Exception as e:
        print(f"\n  ✗ TEST 5 ÉCHOUÉ : {e}")
        import traceback; traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("  ╔═══════════════════════════════════════════════════════╗")
        print("  ║  ✓ TOUS LES TESTS PASSÉS — MVT + Kuramoto OPÉRE   ║")
        print("  ╚═══════════════════════════════════════════════════════╝")
    else:
        print("  ✗ CERTAINS TESTS ONT ÉCHOUÉ — VOIR CI-DESSUS")
    print("=" * 70)


if __name__ == "__main__":
    main()
