"""
MVT - Démo Interactive.
======================

Démonstration complète du Modèle à Variance Topologique.
Montre les 5 composants en action :
1. Encodage du prompt en champ de force
2. Syntopie (one-shot)
3. Intégration Euler-Lagrange (RK4)
4. Plasticité topologique (auto-évolution)
5. Projection en texte
"""

import sys
import json
import numpy as np

# Assurer l'import depuis le bon chemin
sys.path.insert(0, '/home/z/my-project/scripts')

from mvt.config import MVTConfig
from mvt.model import MVT, GenerationResult


def separator(title: str = ""):
    """Affiche un séparateur visuel."""
    width = 72
    if title:
        padding = (width - len(title) - 4) // 2
        print("\n" + "=" * width)
        print(f"{' ' * padding}{title}")
        print("=" * width + "\n")
    else:
        print("\n" + "-" * width + "\n")


def demo_basic_generation():
    """Démonstration basique : générer du texte à partir d'un prompt."""
    separator("1. GÉNÉRATION BASIQUE")

    # Configuration
    config = MVTConfig(
        ambient_dim=16,       # Espace R^16
        intrinsic_dim=8,
        dt=0.02,
        num_rk4_steps=30,
        seed=42,
    )

    model = MVT(config)
    print(f"Modèle initialisé : {model}")

    # Générer du texte
    prompt = "L'intelligence artificielle est"
    print(f"\nPrompt : \"{prompt}\"")
    print("Intégration des équations d'Euler-Lagrange en cours...")

    result = model.generate(prompt, return_trajectory=True)

    print(f"\nTexte généré : \"{result.text}\"")
    print(f"Action S = {result.action:.6f}")
    print(f"Énergie cinétique T = {result.kinetic_energy:.6f}")
    print(f"Énergie potentielle V = {result.potential_energy:.6f}")
    print(f"Nombre de pas : {result.num_steps}")
    print(f"Temps de génération : {result.generation_time:.4f}s")
    print(f"Singularité : {'Oui (étape ' + str(result.singularity_step) + ')' if result.has_singularity else 'Non'}")

    return model, result


def demo_syntopy_one_shot():
    """Démonstration de la syntopie : one-shot sans entraînement."""
    separator("2. SYNTOPIE — ONE-SHOT ABSOLU (★)")

    config = MVTConfig(
        ambient_dim=32,
        intrinsic_dim=16,
        dt=0.02,
        num_rk4_steps=100,
        syntopy_strength=0.8,
        seed=42,
    )

    model = MVT(config)

    # Exemple de format
    example = "La recette de la tarte aux pommes : éplucher les pommes, les couper en morceaux, les mettre dans un moule, ajouter la pâte, cuire à 180°C pendant 40 minutes."
    print(f"Exemple fourni :")
    print(f"  \"{example[:80]}...\"")

    model.set_example(example)
    print(f"\nSyntopie activée : {model.syntopy}")

    # Requête
    prompt = "La recette des crêpes"
    print(f"\nRequête : \"{prompt}\"")
    print("L'opérateur ★ fusionne les topologies...")
    print("Pas de fine-tuning, pas de gradient. Déformation géométrique instantanée.")

    result = model.generate(prompt, return_trajectory=True)

    print(f"\nTexte généré : \"{result.text}\"")
    print(f"Score de syntopie : {result.syntopy_score:.4f}")

    return model, result


def demo_plasticity():
    """Démonstration de la plasticité topologique (auto-évolution)."""
    separator("3. AUTO-ÉVOLUTION TOPOLOGIQUE (Érosion/Sédimentation)")

    config = MVTConfig(
        ambient_dim=16,
        intrinsic_dim=8,
        dt=0.02,
        num_rk4_steps=20,
        alpha_erosion=0.05,
        beta_sedimentation=0.03,
        seed=42,
    )

    model = MVT(config)

    print("Génération multiple avec plasticité activée...")
    print("Le tenseur G(t) évolue à chaque génération :\n")
    print(f"  dG_ij/dt = -α · Courbure(G_ij) + β · Flux_ij\n")

    prompts = [
        "La physique quantique décrit",
        "Le calcul différentiel permet de",
        "La géométrie riemannienne étudie",
        "Les équations d'Euler-Lagrange",
        "La topologie algébrique relie",
    ]

    results = []
    for i, prompt in enumerate(prompts):
        result = model.generate(prompt)
        results.append(result)
        stats = result.plasticity_stats

        print(f"  [{i+1}] \"{prompt}\"")
        print(f"      → \"{result.text[:60]}...\"" if len(result.text) > 60 else f"      → \"{result.text}\"")
        print(f"      Érosions: {stats['total_erosions']} | Sédimentations: {stats['total_sedimentations']}")
        print(f"      Barrières: {stats['total_barriers']} | Canaux: {stats['total_channels']}")
        print(f"      det(G) = {stats['metric_det']:.6e}")
        print()

    # Statistiques finales
    print("Statistiques de plasticité finales :")
    final_stats = model.plasticity.get_plasticity_stats()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")

    return model, results


def demo_singularity_detection():
    """Démonstration de la détection de singularité (anti-hallucination)."""
    separator("4. DÉTECTION DE SINGULARITÉ (Anti-Hallucination)")

    config = MVTConfig(
        ambient_dim=16,
        intrinsic_dim=8,
        dt=0.05,
        num_rk4_steps=30,
        curvature_threshold=5.0,
        seed=42,
    )

    model = MVT(config)

    prompt = "Une question dont l'IA ne connaît pas la réponse précise"
    print(f"Prompt : \"{prompt}\"")
    print("\nSi la trajectoire atteint une singularité (courbure infinie) :")
    print("  → L'IA déclare son ignorance au lieu d'halluciner.")
    print()

    result = model.generate(prompt)
    print(f"Texte généré : \"{result.text}\"")
    print(f"Singularité détectée : {result.has_singularity}")
    print(f"Étape de singularité : {result.singularity_step}")

    return model, result


def demo_full_pipeline():
    """Démonstration du pipeline complet avec toutes les couches."""
    separator("5. PIPELINE COMPLET — TOUS LES COMPOSANTS")

    config = MVTConfig(
        ambient_dim=16,
        intrinsic_dim=8,
        dt=0.02,
        num_rk4_steps=20,
        syntopy_strength=0.7,
        alpha_erosion=0.04,
        beta_sedimentation=0.02,
        seed=42,
    )

    model = MVT(config)
    print(f"Configuration : dim={config.ambient_dim}, dt={config.dt}, steps={config.num_rk4_steps}")
    print(f"Modèle : {model}")

    # Pipeline avec exemple + plasticité
    example = "Question: Quelle est la capitale de la France ? Réponse: Paris."
    model.set_example(example)
    print(f"\nExemple one-shot défini : \"{example}\"")

    # Plusieurs générations
    prompts = [
        "Question: Quelle est la capitale de l'Allemagne ?",
        "Question: Quelle est la capitale du Japon ?",
        "Question: Quelle est la capitale du Brésil ?",
    ]

    print(f"\nGénération de {len(prompts)} réponses avec syntopie active...\n")

    for i, prompt in enumerate(prompts):
        result = model.generate(prompt)
        print(f"  [{i+1}] {prompt}")
        print(f"      → {result.text}")
        print(f"      Action={result.action:.4f} | Syntopie={result.syntopy_score:.4f} | Temps={result.generation_time:.4f}s")
        print()

    # Statistiques globales
    separator("STATISTIQUES GLOBALES DU MODÈLE")
    stats = model.get_stats()
    print(json.dumps(stats, indent=2, default=str))

    return model


def main():
    """Lance toutes les démonstrations."""
    separator("MODÈLE À VARIANCE TOPOLOGIQUE (MVT)")
    print("Topological Variance Model")
    print("Mécanique Géométrique Appliquée à la Pensée")
    print()
    print("Architecture :")
    print("  1. Input Encoder       → Texte → Champ de force F_0 dans R^N")
    print("  2. Syntopic Layer       → Opérateur ★ (fusion topologique)")
    print("  3. Lagrangian Int.     → Euler-Lagrange + Runge-Kutta 4")
    print("  4. Plasticity Engine    → Érosion/Sédimentation de G(t)")
    print("  5. Manifold Projection → Courbe continue → Texte")

    # Démonstrations
    demo_basic_generation()
    demo_syntopy_one_shot()
    demo_plasticity()
    demo_singularity_detection()
    demo_full_pipeline()

    separator("FIN DE LA DÉMONSTRATION")
    print("Le système MVT est entièrement fonctionnel.")
    print("Tous les composants mathématiques sont implémentés :")
    print("  ✓ Tenseur métrique G(t) avec connexion de Levi-Civita")
    print("  ✓ Lagrangien sémantique L = T - V")
    print("  ✓ Intégrateur Runge-Kutta 4 sur Euler-Lagrange")
    print("  ✓ Opérateur de syntopie ★ pour le one-shot")
    print("  ✓ Plasticité topologique (érosion/sédimentation)")
    print("  ✓ Détection de singularités (anti-hallucination)")
    print("  ✓ Projection variété → texte")
    print()


if __name__ == "__main__":
    main()
