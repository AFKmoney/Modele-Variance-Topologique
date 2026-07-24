<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/NumPy-Computational_Geometry-orange?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
<img src="https://img.shields.io/badge/PyTorch-MoE_&_EDT-red?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
<img src="https://img.shields.io/badge/CPU-Optimized-success?style=for-the-badge" alt="CPU"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
<img src="https://img.shields.io/badge/Tests-39%20Passed-cyan?style=for-the-badge" alt="Tests"/>

<br/><br/>

# MVT — Modele a Variance Topologique

### L'IA ne lit plus des tokens. Elle surf sur des varietes geometriques.

<br/>

<code>pip install numpy torch</code>

<br/>

[Architecture](#-architecture) ·
[Mathematiques](#-fondements-mathematiques) ·
[Creativite](#-moteur-de-creativite) ·
[Agent Autonome](#-agent-autonome-mvsagent) ·
[MoE + EDT](#-moe-mvt--expert-decoupled-training) ·
[Usage](#-usage) ·
[Structure](#-structure)

<br/>
<br/>

```
  Les Transformers :              Le MVT :

  Token → Embed → Attention     Mot → Position dans R^N →
    → FFN → Token → ...           Champ de force Coulomb →
                                   Lagrangien (RK4) →
                                   Trajectoire sur variete →
                                   Projection continue → Texte
```

</div>

---

## Pourquoi

Les transformers sont limits par trois hypotheses fondamentales :
1. **Discretisation** : le langage est decoupe en tokens, un par un. Pas de continuite semantique.
2. **Attention globale** : chaque token "regarde" tous les autres. Quadratique en sequence length.
3. **Pas de geometrie** : l'espace latent est plat (euclidien). Pas de structure, pas de courbure.

Le MVT rejette ces trois hypotheses.

**Postulat fondamental** : le sens d'un texte est une *variete riemannienne* vivante dans R^N. La generation de langage est le mouvement d'une particule d'idee sur cette variete, soumise au lagrangien semantique.

Il n'y a pas de tokens. Il n'y a pas d'attention. Il y a un champ de force continu, une metrique courbee, et une trajectoire geodesique.

---

## Architecture

```
                              ┌─────────────────────┐
                              │   INPUT ENCODEUR     │
                              │  SHA-256 + TF-IDF    │
                              │  Mot → R^N (pos)     │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │   CHAMP VECTORIEL   │
                              │  Sources/Sinks      │
                              │  Coulomb en N-dim   │
                              └──────────┬──────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
   ┌──────────▼──────────┐    ┌──────────▼──────────┐    ┌──────────▼──────────┐
   │     SYNTOPIE ★      │    │   TENSEUR METRIQUE  │    │  LAGRANGIEN SEM.    │
   │  Fusion topologique │    │     G_ij(t)         │    │  L = T - V           │
   │  One-shot absolu    │    │  Christoffel Gamma  │    │  Euler-Lagrange      │
   │  Sans gradient      │    │  Courbure scalaire R │    │  Integrateur RK4     │
   └──────────┬──────────┘    └──────────┬──────────┘    └──────────┬──────────┘
              │                          │                          │
              └──────────────────────────┼──────────────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │   CREATIVITE ENGINE  │
                              │  Bruit + Nouveaute   │
                              │  Sauts metaphoriques │
                              │  Soft-max selection  │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │    PLASTICITE        │
                              │  dG/dt = -alpha*C   │
                              │       + beta*F       │
                              │  Erosion/Sediment    │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │    PROJECTEUR        │
                              │  R^N → Texte        │
                              │  Plus proche voisin  │
                              └─────────────────────┘
```

---

## Fondements Mathematiques

### Tenseur Metrique Dynamique G_ij(t)

L'espace semantique est une variete riemannienne dont la metrique evolue dans le temps.

```python
# Initialisation : metrique quasi-euclidienne
G = I_N + epsilon * perturbation_symetrique

# Met a jour avec garantie de definie positivite
G += dG
# Clamp des valeurs propres : lambda_i >= 1e-6
```

**Symboles de Christoffel** (vectorise via `np.einsum`) :

$$\Gamma^k_{ij} = \frac{1}{2} G^{kl} \left( \frac{\partial G_{li}}{\partial q^j} + \frac{\partial G_{lj}}{\partial q^i} - \frac{\partial G_{ij}}{\partial q^l} \right)$$

```python
Gamma = 0.5 * np.einsum('kl,lij->kij', G_inv, dG_combined)  # (N, N, N)
```

**Acceleration geodesique** :

$$\ddot{q}^k = -\Gamma^k_{ij} \, \dot{q}^i \, \dot{q}^j$$

```python
accel = -np.einsum('kij,i,j->k', Gamma, dq, dq)
```

**Courbure scalaire** :

$$R \approx \frac{1}{N} \text{tr}(G^{-1} \cdot \Delta \cdot \Delta)$$

### Lagrangien Semantique

$$\mathcal{L}(q, \dot{q}, t) = T(q, \dot{q}) - V(q, t)$$

Ou l'energie cinetique est definie par la metrique riemannienne :

$$T = \frac{1}{2} \dot{q}^T \, G \, \dot{q}$$

Et le potentiel combine confinement doux + force externe semantique :

$$V(q) = V_{\text{confinement}}(q) + V_{\text{semantique}}(q)$$

Les equations d'Euler-Lagrange sont resolues numeriquement par **Runge-Kutta 4** :

```python
# RK4 step
k1_q, k1_dq = derivees(q, dq, t)
k2_q, k2_dq = derivees(q + 0.5*dt*k1_q, dq + 0.5*dt*k1_dq, t + 0.5*dt)
k3_q, k3_dq = derivees(q + 0.5*dt*k2_q, dq + 0.5*dt*k2_dq, t + 0.5*dt)
k4_q, k4_dq = derivees(q + dt*k3_q, dq + dt*k3_dq, t + dt)

q_new = q + (dt/6) * (k1_q + 2*k2_q + 2*k3_q + k4_q)
dq_new = dq + (dt/6) * (k1_dq + 2*k2_dq + 2*k3_dq + k4_dq)
```

### Syntopie — Operateur Topologique

L'operateur **★** (syntopie) fusionne deux varietes semantiques sans gradient :

$$\tau = \nabla \star (M_{\text{requete}} \oplus M_{\text{exemple}})$$

- La topologie de l'exemple "s'imprime" instantanement sur la requete
- **Zero gradient, zero fine-tuning** : la deformation est purement geometrique
- Permet le one-shot absolu : un seul exemple suffit pour contraindre la generation

### Encodage — SHA-256 Deterministe

Chaque mot est projete en position unique dans R^N via hachage :

```python
h = sha256(word.encode('utf-8')).hexdigest()
pos[i] = scale * tanh(normalize(h[start:start+4]))
```

Proprietes :
- **Deterministe** : meme mot = meme position, toujours
- **Quasi-uniforme** : SHA-256 garantit une distribution homogene
- **Illimite** : aucun vocabulaire fige, tout mot peut etre encode
- **TF-IDF** : les mots rares ont un poids plus fort

---

## Moteur de Creativite

Le MVT n'est pas un LLM deterministe. Il possede un moteur de creativite a 5 couches :

| Couche | Mecanisme | Effet |
|--------|-----------|-------|
| **Temperature** | Bruit gaussien dans l'acceleration | Exploration stochastique |
| **Novelty-seeking** | Repulsion 1/r des regions visitees | Anti-stagnation, anti-repetition |
| **Sauts metaphoriques** | Discontinuite probabiliste (10%) | Associations inedites |
| **Soft-max selection** | Election probabiliste des mots | Diversite lexicale |
| **Pensée divergente** | Trajectoires paralleles | Exploration multi-chemins |

```python
# Bruit thermique
noise = randn(N) * noise_scale * temperature * (1 + tanh(r/5))

# Force de nouveaute
force = sum(novelty_bias / (dist^2 + 0.1) * direction)
```

**Anti-hallucination douce** : les singularites sont *detectees* (courbure > seuil) mais *jamais bloquantes*. La generation continue avec un marqueur. Seuil par defaut : 100.0 (relaxe 10-100x par rapport a un modele conservateur).

```python
if abs(curvature) > threshold:
    return True, step  # Signale, mais ne bloque PAS
```

**Selection de mots** : soft-max avec penalite de diversite :

```python
probs = softmax(-distances / temperature)
probs[word] *= diversity_penalty ** (1 + repeat_count)
selected = sample(candidates, probs)
```

---

## Agent Autonome (MVSAgent)

```
    ┌──────────────────────────────────────────────┐
    │                BOUCLE AGENTIQUE               │
    │                                               │
    │   OBSERVE ──► PLAN ──► GENERATE ──► REFLECT  │
    │       ▲                                   │   │
    │       │         ◄── REPLAN ◄─── ◄────────┘   │
    │       │                                       │
    │       └──── (insatisfaction) ──────────────── │
    └──────────────────────────────────────────────┘
```

L'agent est **agentic de nature** — il ne fait pas que generer du texte, il :

1. **Observe** : analyse le prompt, identifie les concepts cles, mesure la complexite
2. **Planifie** : decompose en sous-buts semantiques (Goals) dans R^N
3. **Genere** : integre avec steering vers la cible (goal-directed)
4. **Reflechit** : evalue satisfaction, creativite, coherence (scores 0-1)
5. **Replanifie** : ajuste temperature, metaphor_prob, et cible si insuffisant

```python
agent = MVSAgent(MVTConfig(max_agent_iterations=5))
result = agent.run("Imagine un monde ou la gravite n'existe pas", verbose=True)

# L'agent :
# - Observe la complexite du prompt
# - Cree 1-2 sous-buts (exploration + creativite)
# - Genere, evalue, et eventuellement replanifie
# - Retourne le meilleur resultat apres 1-5 iterations
```

**Memoire episodique** : chaque trace (prompt, resultat, score, temps) est stockee pour adaptation future.

---

## MoE-MVT & Expert Decoupled Training

### Architecture Sparse MoE

```
                    ┌───────────────────────┐
  Token ID ────────►│  TOPO EMBEDDING       │
                    │  tok_embed + pos_embed │
                    └──────────┬────────────┘
                               │
                    ┌──────────▼────────────┐
                    │   TOPO BLOCK × N      │
                    │                       │
                    │  Norm → Attn → +res    │
                    │  Norm → MoE  → +res    │
                    │         │              │
                    │  ┌──────▼──────┐       │
                    │  │  ROUTER     │       │
                    │  │  top-k sel  │       │
                    │  └──┬──┬──┬───┘       │
                    │     │  │  │            │
                    │  ┌──▼┐┌▼──┐┌▼──┐     │
                    │  │E1││E2││Ek│ ...    │
                    │  └───┘└───┘└───┘     │
                    └──────────┬────────────┘
                               │
                    ┌──────────▼────────────┐
                    │  LM_HEAD (weight tie) │
                    └───────────────────────┘
```

Chaque expert est un "specialiste topologique" d'une region semantique. Le routeur distribue chaque token vers les `top_k` experts les plus pertinents. Seule une fraction des parametres est active par token.

**Weight tying** : `lm_head.weight = embed.tok_embed.weight` → reduction de parametres.

### EDT — Expert Decoupled Training

Le pipeline EDT entraine **chaque composant independamment**, puis les aligne brievement. Resultat : **189x acceleration** revendiquee vs training standard.

```
  PHASE 1                    PHASE 2a                    PHASE 2b
  ┌─────────────┐           ┌─────────────┐           ┌─────────────┐
  │  EXPERTS     │           │  ATTENTION   │           │  EMBEDDING  │
  │  Independant │           │  Independant │           │  Separable   │
  │             │           │             │           │             │
  │  MSE(h_in,  │           │  MSE(h_in +  │           │  Next-token  │
  │   h_target) │           │   attn,      │           │  prediction │
  │             │           │   h_target)  │           │             │
  │  150 steps  │           │  300 steps   │           │  2M tokens  │
  │  / expert   │           │  / layer     │           │             │
  └──────┬──────┘           └──────┬──────┘           └──────┬──────┘
         │                         │                         │
         └─────────────┬───────────┘                         │
                       │                                     │
              ┌────────▼────────┐                            │
              │  PHASE 3        │◄───────────────────────────┘
              │  JOINT FINE-TUNE│
              │                 │
              │  Tous degelés   │
              │  PGSG rotation  │
              │  CE + aux_loss │
              │  500K tokens   │
              └─────────────────┘
```

### PGSG — Partial Gradient Sequential Update

Seules `n_active_layers` couches sur `n_layers` total recoivent des gradients par step. Rotation circulaire : les couches ignorees a ce step le seront aux prochains.

```python
# Step 0 : layers 0,1 actives (sur 4)
# Step 1 : layers 1,2 actives
# Step 2 : layers 2,3 actives
# Step 3 : layers 3,0 actives
# Step 4 : layers 0,1 actives (cycle)
```

**Resultat** : ~60% reduction du backprop, ideal pour CPU.

---

## Usage

### Installation

```bash
pip install numpy torch
```

### Generation Creative

```python
from mvt import MVTConfig, MVT

config = MVTConfig(
    ambient_dim=16,       # Dimension de l'espace semantique
    temperature=0.8,      # 0 = deterministe, 1 = tres creatif
    novelty_bias=0.4,     # Force de recherche de nouveaute
    metaphor_jump_prob=0.1,  # 10% de chance de saut metaphorique par step
)

model = MVT(config)
result = model.generate("consciousness emerges from", temperature=0.9)

print(result.text)                          # Texte genere
print(f"Creativity : {result.creativity_score:.2f}")
print(f"Diversity  : {result.diversity_score:.2f}")
print(f"Time       : {result.generation_time:.4f}s")
print(f"Singularity: {result.has_singularity}")

# Ajuster la temperature a la volee
model.set_temperature(1.2)  # Plus chaud
model.set_temperature(0.3)  # Plus froid
```

### Agent Autonome

```python
from mvt import MVTConfig, MVSAgent

config = MVTConfig(
    ambient_dim=16,
    max_agent_iterations=5,          # Max boucles agentiques
    self_reflection_threshold=0.5,   # Seuil de satisfaction
    goal_steering_strength=0.3,     # Force d'attraction vers le but
)

agent = MVSAgent(config)
result = agent.run("Imagine un monde ou la gravite n'existe pas", verbose=True)

# L'agent observe, planifie, genere, reflechit, replanifie si besoin
# Retourne le meilleur resultat
```

### One-Shot Learning (Syntopie)

```python
from mvt import MVTConfig, MVT

model = MVT(MVTConfig(ambient_dim=16))

# Apprendre un style en UN seul exemple — zero gradient
model.set_example("Le chat dort sur le canape. Le chien court dans le jardin.")

# La topologie de l'exemple contraint la generation
result = model.generate("Le poisson nage dans", temperature=0.8)
print(result.text)
print(f"Syntopie score: {result.syntopy_score:.2f}")

model.clear_example()
```

### MoE-MVT + EDT (PyTorch, CPU)

```python
from mvt.edt import MoEMVT, MoEMVTConfig, EDTConfig, run_edt, generate_synthetic_corpus

# Configuration du modele
model_cfg = MoEMVTConfig(
    vocab_size=4000,     # Taille vocabulaire
    d_model=128,         # Dimension du modele
    n_layers=4,          # Nombre de blocs transformer
    n_experts=8,         # Experts par couche MoE
    top_k=2,             # Experts actives par token
    d_ff=256,            # Dimension intermediaire
    max_seq_len=64,      # Longueur max sequence
)

# Configuration EDT (optimisee CPU)
edt_cfg = EDTConfig(
    phase1_steps_per_expert=50,     # Steps par expert (Phase 1)
    phase1_hidden_samples=500,      # Etats caches pour pre-entrainement
    phase2a_steps_per_layer=100,    # Steps par couche attention (Phase 2a)
    phase2b_n_tokens=100_000,       # Tokens pour embedding (Phase 2b)
    phase3_n_tokens=50_000,         # Tokens pour joint (Phase 3)
    phase3_n_active_layers=2,       # PGSG : 2 couches actives / 4
    save_dir="./checkpoints",
)

# Entrainement
model = MoEMVT(model_cfg)
corpus = generate_synthetic_corpus(vocab_size=4000, length=50_000)
stats = run_edt(model, corpus, edt_cfg, verbose=True)

total, active = model.count_params()
print(f"Params: {total:,} total, {active:,} actifs/token")
print(f"Sparsite: {1 - active/total:.1%}")
```

### Plasticite Topologique

```python
from mvt import MVTConfig, MVT

model = MVT(MVTConfig(ambient_dim=16))

# Generer plusieurs fois — la metrique evolue
for prompt in ["La musique est un langage", "Les couleurs dansent"]:
    result = model.generate(prompt)
    stats = result.plasticity_stats
    print(f"Barrieres: {stats['total_barriers']}, Canaux: {stats['total_channels']}")
    print(f"det(G): {stats['metric_det']:.4f}")

# La metrique s'est auto-modifiee :
# - Erosion la ou les trajectoires ont echoue
# - Sedimentation la ou elles ont reussi
```

---

## Configuration Complete

```python
@dataclass
class MVTConfig:
    # === Espace ===
    ambient_dim: int = 128          # Dimension de R^N
    intrinsic_dim: int = 64         # Dimension intrinseque

    # === Integration ===
    dt: float = 0.01                 # Pas de temps RK4
    num_rk4_steps: int = 200         # Nombre de pas

    # === Plasticite ===
    alpha_erosion: float = 0.05       # Taux d'erosion
    beta_sedimentation: float = 0.03  # Taux de sedimentation

    # === Lagrangien ===
    damping: float = 0.05            # Amortissement (bas = creatif)
    potential_stiffness: float = 2.0  # Raideur du confinement

    # === Creativite ===
    temperature: float = 0.8          # Bruit stochastique
    novelty_bias: float = 0.4         # Biais vers le neuf
    metaphor_jump_prob: float = 0.1   # Sauts metaphoriques

    # === Agent ===
    max_agent_iterations: int = 5     # Max boucles
    self_reflection_threshold: float = 0.5
    goal_steering_strength: float = 0.3

    # === Projection ===
    projection_temperature: float = 0.7
    diversity_penalty: float = 0.3
    max_consecutive_repeat: int = 2

    # === Stabilite (relache pour creativite) ===
    curvature_threshold: float = 100.0
    divergence_threshold: float = 1e8
```

---

## Tests

```bash
cd scripts && python -m pytest mvt/tests.py -v
```

39 tests couvrant tous les composants — passent en **0.58s**.

```
core/test_metric_tensor .......... OK  (Christoffel, geodesique, courbure)
core/test_vector_field ........... OK  (Coulomb, potentiel, divergence)
lagrangian/test_semantic_lag ..... OK  (T, V, Euler-Lagrange)
lagrangian/test_integrator ........ OK  (RK4, action, divergence)
test_encoder .................... OK  (SHA-256, TF-IDF, similarity)
test_syntopy .................... OK  (fusion, syntopie score)
test_plasticity ................. OK  (erosion, sedimentation)
test_creativity ................. OK  (bruit, nouveaute, sauts, soft-max)
test_model ...................... OK  (generation, singularites)
test_agent ...................... OK  (boucle, reflexion, replan)
```

---

## Demos

```bash
cd scripts && python -m mvt.demo
```

5 demos interactives :
1. **Generation creative** — Temperature, diversite, sauts metaphoriques
2. **Agent autonome** — Boucle observe/plan/generate/reflect
3. **One-shot syntopie** — Apprendre un style en un exemple
4. **Plasticite** — Auto-evolution de la metrique
5. **Agent creatif** — Tache complexe avec replanification

---

## EDT Training

```bash
cd scripts && python -m mvt.edt.run_edt
```

Pipeline complet Phase 1 → 2a → 2b → 3 avec checkpoints intermediaires.

---

## Structure

```
mvt/
├── __init__.py                  # MVTConfig, MVT, MVSAgent, CreativityEngine
├── config.py                    # 25+ hyperparametres
├── model.py                     # Orchestrateur principal v2
├── encoder.py                   # SHA-256 + TF-IDF → R^N
├── syntopy.py                   # Operateur ★ de fusion topologique
├── plasticity.py                # dG/dt = -alpha*C + beta*F
├── projector.py                 # R^N → Texte (plus proche voisin)
├── creativity.py                # Bruit, nouveaute, sauts, soft-max
├── agent.py                     # Observe→Plan→Generate→Reflect→Replan
├── demo.py                      # 5 demos interactives
├── tests.py                     # 39 tests unitaires
│
├── core/
│   ├── metric_tensor.py         # G_ij(t), Christoffel, courbure (einsum)
│   └── vector_field.py          # Coulomb N-dim, potentiel, divergence
│
├── lagrangian/
│   ├── semantic_lagrangian.py   # L = T - V, Euler-Lagrange
│   └── integrator.py            # RK4, action, trajectoire optimale
│
└── edt/
    ├── moe_model.py             # MoE-MVT : Router + Experts + Attention
    ├── edt_pipeline.py           # 4 phases + PGSG
    └── run_edt.py                # Script d'entrainement CPU
```

---

## Performance

| Operation | Complexite | Implementation |
|-----------|-----------|----------------|
| Christoffel Symbols | O(N^3) | `np.einsum('kl,lij->kij')` — vectorise |
| RK4 Step | O(N^2) | 4 evaluations + combinaison |
| Distance geodesique | O(N^2) | `sqrt(delta^T G delta)` |
| MoE Forward | O(k * d^2 * f) | k experts actifs sur n total |
| EDT Phase 1 | O(E * S * d^2) | E experts, S steps, parallele |
| PGSG | -60% backprop | Rotation circulaire des couches |

**Optimisation CPU** : Christoffel vectorises via einsum (73s → 1.1s pour N=128), PGSG reduit le gradient computation, embedding separable pour Phase 2b.

---

## Auteur

**[AFKmoney](https://github.com/AFKmoney)**

## Licence

MIT
