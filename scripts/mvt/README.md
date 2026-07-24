# MVT — Modèle à Variance Topologique

> Une architecture d'IA fondamentalement nouvelle qui remplace les transformers par la géométrie différentielle, la mécanique lagrangienne et les champs topologiques pour la génération de langage.

## Architecture

Le MVT est construit sur trois piliers mathématiques :

| Composant | Théorie | Rôle |
|---|---|---|
| **Tenseur Métrique G(t)** | Géométrie riemannienne | Espace sémantique courbé avec symboles de Christoffel |
| **Lagrangien Sémantique** | Mécanique analytique | L = T - V ; équations d'Euler-Lagrange via RK4 |
| **Syntopie (★)** | Topologie algébrique | Fusion topologique pour l'apprentissage one-shot |

### Pipeline

1. **Encodage** : Mots → positions dans R^N via hachage SHA-256 + TF-IDF
2. **Champ vectoriel** : Sources/sinks de Coulomb pour la sémantique
3. **Syntopie** : Fusion topologique (si exemple fourni) — one-shot learning sans gradients
4. **Intégration RK4** : Trajectoire sur la variété via Euler-Lagrange
5. **Projection** : Trajectoire → texte via plus proche voisin dans le vocabulaire

## Créativité

- **Température stochastique** : Bruit injecté dans le lagrangien
- **Force de nouveauté** : Anti-stagnation, repulsion des régions déjà visitées
- **Sauts métaphoriques** : Discontinuités créatives probabilistes (10% par défaut)
- **Soft-max word selection** : Élection probabiliste des mots avec pénalité de diversité
- **Anti-hallucination douce** : Singularités signalées mais jamais bloquantes

## Agent Autonome (MVSAgent)

Boucle agentique intégrée :

```
Observe → Plan → Generate → Reflect → Replan → Loop
```

- **Buts dirigés** : L'agent vise des cibles sémantiques dans R^N
- **Auto-réflexion** : Score de satisfaction, créativité, cohérence
- **Replanification** : Adaptation de la température et des probabilités de saut
- **Mémoire épisodique** : Trajectoires passées stockées

## MoE-MVT (Mixture-of-Experts)

Extension en architecture sparse MoE avec routage topologique :

- **Routeur** : Softmax avec température, sélection top-k experts par token
- **Experts topologiques** : MLP 2 couches avec GELU (spécialistes de régions sémantiques)
- **Attention multi-têtes** : Self-attention causale avec projection QKV fusionnée
- **Weight tying** : Embedding ↔ LM head pour réduire les paramètres

## EDT — Expert Decoupled Training

Pipeline d'entraînement en 4 phases, chaque composant est entraîné indépendamment puis aligné :

| Phase | Composant | Méthode |
|---|---|---|
| **1** | Experts (indépendant) | MSE sur états cachés réels |
| **2a** | Attention (indépendant) | MSE + connexion résiduelle |
| **2b** | Embedding | Next-token prediction (séparable) |
| **3** | Joint (PGSG) | Cross-entropy + loss auxiliaire + rotation de couches |

**PGSG** (Partial Gradient Sequential Update) : Seules N couches sur M reçoivent des gradients par step (~60% réduction du backprop).

## Installation

```bash
pip install numpy torch
```

## Usage rapide

### Génération créative (NumPy)

```python
from mvt import MVTConfig, MVT

config = MVTConfig(ambient_dim=16, temperature=0.8)
model = MVT(config)
result = model.generate("consciousness emerges from", temperature=0.9)

print(result.text)
print(f"Creativity: {result.creativity_score:.2f}")
print(f"Diversity: {result.diversity_score:.2f}")
```

### Agent autonome

```python
from mvt import MVTConfig, MVSAgent

config = MVTConfig(ambient_dim=16, max_agent_iterations=3)
agent = MVSAgent(config)
result = agent.run("The universe is", verbose=True)
```

### MoE-MVT avec EDT (PyTorch)

```python
from mvt.edt import MoEMVT, MoEMVTConfig, EDTConfig, run_edt, generate_synthetic_corpus

# Configuration
model_cfg = MoEMVTConfig(
    vocab_size=4000, d_model=128, n_layers=4,
    n_experts=8, top_k=2, d_ff=256, max_seq_len=64,
)
edt_cfg = EDTConfig(
    phase1_steps_per_expert=50,
    phase2a_steps_per_layer=100,
    phase2b_n_tokens=100_000,
    phase3_n_tokens=50_000,
    phase3_n_active_layers=2,
)

# Création + entraînement
model = MoEMVT(model_cfg)
corpus = generate_synthetic_corpus(vocab_size=4000, length=50_000)
stats = run_edt(model, corpus, edt_cfg, verbose=True)
```

### One-shot learning (Syntopie)

```python
from mvt import MVTConfig, MVT

model = MVT(MVTConfig(ambient_dim=16))

# Apprendre un style en un seul exemple
model.set_example("The quantum field oscillates with infinite possibilities")
result = model.generate("consciousness and", temperature=0.8)
```

## Structure du projet

```
mvt/
├── __init__.py              # Exports publics (MVTConfig, MVT, MVSAgent, CreativityEngine)
├── config.py                # Hyperparamètres globaux
├── model.py                 # Orchestrateur principal MVT v2
├── encoder.py               # Encodage SHA-256 + TF-IDF
├── syntopy.py               # Couche de syntopie (fusion topologique)
├── plasticity.py            # Plasticité topologique (érosion/sédimentation)
├── projector.py             # Projection trajectoire → texte
├── creativity.py           # Moteur de créativité (bruit, nouveauté, métaphores)
├── agent.py                 # Agent autonome (Observe→Plan→Generate→Reflect→Replan)
├── demo.py                  # Démos interactives
├── tests.py                 # 39 tests unitaires
├── core/
│   ├── __init__.py
│   ├── metric_tensor.py     # Tenseur métrique riemannien (einsum vectorisé)
│   └── vector_field.py      # Champ vectoriel (sources/sinks de Coulomb)
├── lagrangian/
│   ├── __init__.py
│   ├── semantic_lagrangian.py  # L = T - V avec force de nouveauté
│   └── integrator.py            # Intégrateur RK4
└── edt/
    ├── __init__.py
    ├── moe_model.py          # Architecture MoE-MVT complète
    ├── edt_pipeline.py       # Pipeline EDT 4 phases + PGSG
    └── run_edt.py            # Script d'entraînement complet
```

## Tests

```bash
cd scripts && python -m pytest mvt/tests.py -v
```

Tous les 39 tests passent en ~0.6s.

## Dépendances

- `numpy` — Calcul tensoriel, géométrie riemannienne, intégration RK4
- `torch` — MoE-MVT, entraînement EDT, autograd

## Auteur

**AFKmoney**

## Licence

MIT
