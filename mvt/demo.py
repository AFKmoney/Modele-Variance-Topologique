"""
MVT v2 - Démo Agentique & Créative.
====================================

Démonstration complète du MVT v2 avec :
1. Génération créative (température, diversité, sauts métaphoriques)
2. Agent autonome (boucle observe → plan → generate → reflect)
3. One-shot avec syntopie
4. Auto-évolution topologique
5. Anti-hallucination SOUPLE (singularités contrôlées)
"""

import sys
import json
import numpy as np

sys.path.insert(0, '/home/z/my-project/scripts')

from mvt.config import MVTConfig
from mvt.model import MVT, GenerationResult
from mvt.agent import MVSAgent, AgentState
from mvt.creativity import CreativityEngine


def sep(title: str = ""):
    width = 72
    if title:
        padding = (width - len(title) - 4) // 2
        print(f"\n{'=' * width}")
        print(f"{' ' * padding}{title}")
        print(f"{'=' * width}\n")
    else:
        print(f"\n{'-' * width}\n")


def demo_creative_generation():
    """Démonstration de la génération créative avec température."""
    sep("1. GÉNÉRATION CRÉATIVE (Température)")

    config = MVTConfig(
        ambient_dim=16,
        intrinsic_dim=8,
        dt=0.02,
        num_rk4_steps=60,
        seed=42,
        temperature=0.8,
        novelty_bias=0.4,
        metaphor_jump_prob=0.1,
        creativity_noise_scale=0.15,
        projection_temperature=0.7,
        diversity_penalty=0.3,
        max_consecutive_repeat=2,
        sample_rate=3,
        curvature_threshold=100.0,
        divergence_threshold=1e8,
    )

    model = MVT(config)
    print(f"Config: temp={config.temperature}, "
          f"novelty={config.novelty_bias}, "
          f"metaphor_prob={config.metaphor_jump_prob}")
    print(f"Modèle: {model}")

    prompts = [
        "L'univers est",
        "La conscience humaine",
        "Le temps qui passe",
        "L'art de la pensee",
        "Les etoiles brillent",
    ]

    for prompt in prompts:
        result = model.generate(prompt)
        words = result.text.split()
        unique = set(words)
        print(f"\n  Prompt: \"{prompt}\"")
        print(f"  Sortie: \"{result.text}\"")
        print(f"  Mots: {len(words)} | Uniques: {len(unique)} | "
              f"Diversité: {result.diversity_score:.2f}")
        print(f"  Créativité: {result.creativity_score:.2f} | "
              f"Temps: {result.generation_time:.4f}s")

    return model


def demo_agentic_loop():
    """Démonstration de la boucle agentique autonome."""
    sep("2. AGENT AUTONOME (Observe → Plan → Génère → Réfléchit)")

    config = MVTConfig(
        ambient_dim=16,
        intrinsic_dim=8,
        dt=0.02,
        num_rk4_steps=40,
        seed=42,
        temperature=0.7,
        novelty_bias=0.4,
        metaphor_jump_prob=0.08,
        creativity_noise_scale=0.12,
        projection_temperature=0.7,
        diversity_penalty=0.3,
        max_consecutive_repeat=2,
        sample_rate=3,
        curvature_threshold=100.0,
        divergence_threshold=1e8,
        max_agent_iterations=3,
        self_reflection_threshold=0.4,
        goal_steering_strength=0.3,
        branching_factor=2,
    )

    agent = MVSAgent(config)
    print(f"Agent: {agent}")
    print(f"Max itérations: {config.max_agent_iterations}")
    print(f"Seuil satisfaction: {config.self_reflection_threshold}")

    prompt = "Imagine un monde ou la gravité n'existe pas"
    print(f"\nPrompt: \"{prompt}\"")
    print()

    result = agent.run(prompt, verbose=True)

    stats = agent.get_agent_stats()
    print(f"\nStats agent: {json.dumps(stats, indent=2, default=str)}")

    return agent, result


def demo_one_shot_creative():
    """One-shot créatif avec syntopie."""
    sep("3. ONE-SHOT CRÉATIF (Syntopie ★)")

    config = MVTConfig(
        ambient_dim=16,
        intrinsic_dim=8,
        dt=0.02,
        num_rk4_steps=50,
        seed=42,
        temperature=0.9,
        novelty_bias=0.5,
        metaphor_jump_prob=0.12,
        syntopy_strength=0.8,
        projection_temperature=0.8,
        diversity_penalty=0.3,
        max_consecutive_repeat=2,
        sample_rate=3,
        curvature_threshold=100.0,
        divergence_threshold=1e8,
    )

    model = MVT(config)

    example = "Le chat dort sur le canape. Le chien court dans le jardin."
    model.set_example(example)
    print(f"Exemple: \"{example}\"")

    prompts = [
        "Le poisson nage dans",
        "L'oiseau vole au-dessus de",
        "Le cheval galope dans la",
    ]

    for prompt in prompts:
        result = model.generate(prompt)
        print(f"\n  Requête: \"{prompt}\"")
        print(f"  Sortie: \"{result.text}\"")
        print(f"  Syntopie: {result.syntopy_score:.2f} | "
              f"Créativité: {result.creativity_score:.2f} | "
              f"Diversité: {result.diversity_score:.2f}")

    return model


def demo_plasticity_creative():
    """Plasticité topologique avec créativité — auto-évolution douce."""
    sep("4. AUTO-ÉVOLUTION DOUCE (Érosion/Sédimentation Créative)")

    config = MVTConfig(
        ambient_dim=16,
        intrinsic_dim=8,
        dt=0.02,
        num_rk4_steps=40,
        seed=42,
        temperature=0.8,
        novelty_bias=0.4,
        metaphor_jump_prob=0.1,
        projection_temperature=0.7,
        diversity_penalty=0.3,
        max_consecutive_repeat=2,
        sample_rate=3,
        curvature_threshold=100.0,
        divergence_threshold=1e8,
        alpha_erosion=0.02,  # Érosion douce
        beta_sedimentation=0.04,  # Sédimentation active
    )

    model = MVT(config)
    print(f"Érosion: {config.alpha_erosion} (douce) | "
          f"Sédimentation: {config.beta_sedimentation} (active)")

    prompts = [
        "La musique est un langage",
        "Les couleurs dansent sous la lumiere",
        "Le vent murmure des secrets",
        "La mer reflecte le ciel",
        "Le silence parle plus fort",
    ]

    for i, prompt in enumerate(prompts):
        result = model.generate(prompt)
        print(f"\n  [{i+1}] \"{prompt}\"")
        print(f"      → \"{result.text}\"")
        stats = result.plasticity_stats
        print(f"      Barrières: {stats['total_barriers']} | "
              f"Canaux: {stats['total_channels']} | "
              f"det(G): {stats['metric_det']:.4f}")

    print(f"\nPlasticité finale: {json.dumps(model.plasticity.get_plasticity_stats(), indent=4, default=str)}")

    return model


def demo_agent_creative_task():
    """Agent qui accomplit une tâche créative complexe."""
    sep("5. AGENT — TÂCHE CRÉATIVE COMPLEXE")

    config = MVTConfig(
        ambient_dim=16,
        intrinsic_dim=8,
        dt=0.02,
        num_rk4_steps=50,
        seed=42,
        temperature=0.9,
        novelty_bias=0.5,
        metaphor_jump_prob=0.15,
        creativity_noise_scale=0.2,
        projection_temperature=0.9,
        diversity_penalty=0.3,
        max_consecutive_repeat=2,
        sample_rate=3,
        curvature_threshold=100.0,
        divergence_threshold=1e8,
        max_agent_iterations=3,
        self_reflection_threshold=0.45,
        goal_steering_strength=0.4,
    )

    agent = MVSAgent(config)

    prompt = "Decris une cite flottante dans les nuages"
    print(f"Prompt: \"{prompt}\"")
    print()

    result = agent.run(prompt, verbose=True)

    return agent, result


def main():
    sep("MVT v2 — AGENTIQUE & CRÉATIF")
    print("Topological Variance Model v2.0")
    print("Mécanique Géométrique + Agent Autonome + Créativité")
    print()
    print("Nouveautés v2 :")
    print("  1. Moteur de créativité (bruit, nouveauté, sauts métaphoriques)")
    print("  2. Agent autonome (Observe → Plan → Génère → Réfléchit)")
    print("  3. Anti-hallucination SOUPLE (singularités contrôlées)")
    print("  4. Projection créative (soft-max, diversité lexicale)")
    print("  5. Auto-évolution douce (érosion minimale, sédimentation active)")
    print()

    demo_creative_generation()
    demo_agentic_loop()
    demo_one_shot_creative()
    demo_plasticity_creative()
    demo_agent_creative_task()

    sep("FIN DE LA DÉMONSTRATION")
    print("Le MVT v2 est agentique et créatif.")
    print("L'anti-hallucination est souple — l'IA explore librement.")
    print()


if __name__ == "__main__":
    main()
