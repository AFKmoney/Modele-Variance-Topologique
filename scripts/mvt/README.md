<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/NumPy-Computational_Geometry-orange?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
<img src="https://img.shields.io/badge/PyTorch-MoE_&_EDT-red?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
<img src="https://img.shields.io/badge/CPU-Optimized-success?style=for-the-badge" alt="CPU"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
<img src="https://img.shields.io/badge/Tests-44%20Passed-cyan?style=for-the-badge" alt="Tests"/>
<img src="https://img.shields.io/badge/Version-3.0-9cf?style=for-the-badge" alt="v3.0"/>

<br/><br/>

# MVT — Mod&egrave;le &agrave; Variance Topologique

### AI doesn't read tokens anymore. It surfs on geometric manifolds.

<br/>

`pip install numpy torch`

<br/>

[Architecture](#architecture) &middot;
[Mathematics](#mathematical-foundations) &middot;
[Kuramoto &harr; Metric Coupling](#-kuramoto--metric-coupling--morphological-memory) &middot;
[Creativity Engine](#creativity-engine) &middot;
[Autonomous Agent](#autonomous-agent-mvsagent) &middot;
[MoE + EDT](#moe-mvt--expert-decoupled-training) &middot;
[Chinchilla Scaling](#chinchilla-scaling-for-mvt) &middot;
[Usage](#usage) &middot;
[Structure](#project-structure)

<br/><br/>

```
  Transformers:                  MVT:

  Token &rarr; Embed &rarr; Attn   Word &rarr; Position in R^N &rarr;
    &rarr; FFN &rarr; Token          Coulomb Force Field &rarr;
                                    Lagrangian (RK4) &rarr;
                                    Trajectory on Manifold &rarr;
                                    Continuous Projection &rarr; Text
```

</div>

---

## Why

Transformers are constrained by three fundamental hypotheses:

1. **Discretization**: language is chopped into tokens, one by one. No semantic continuity.
2. **Global Attention**: every token "looks at" all others. Quadratic in sequence length.
3. **Flat Geometry**: the latent space is Euclidean. No curvature, no structure.

MVT rejects all three.

**Core Postulate**: the meaning of text is a *living Riemannian manifold* in R^N. Language generation is the movement of an idea-particle on that manifold, governed by a semantic Lagrangian.

There are no tokens. There is no attention. There is a continuous force field, a curved metric, and a geodesic trajectory.

---

## Architecture

```
                              ┌─────────────────────┐
                              │    INPUT ENCODER     │
                              │  SHA-256 + TF-IDF    │
                              │  Word &rarr; R^N (pos)│
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │   VECTOR FIELD       │
                              │  Sources / Sinks     │
                              │  Coulomb in N-dim    │
                              └──────────┬──────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
   ┌──────────▼──────────┐    ┌──────────▼──────────┐    ┌──────────▼──────────┐
   │     SYNTOPIE ★      │    │   METRIC TENSOR     │    │  SEMANTIC LAGRANG.  │
   │  Topological Fusion │    │     G_ij(t)         │    │  L = T - V           │
   │  One-shot absolute   │    │  Christoffel &Gamma; │    │  Euler-Lagrange      │
   │  Zero gradient       │    │  Scalar curvature R  │    │  RK4 Integrator      │
   └──────────┬──────────┘    └──────────┬──────────┘    └──────────┬──────────┘
              │                          │                          │
              └──────────────────────────┼──────────────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  ★ KURAMOTO COUPLER │
                              │  Kuramoto &harr; G(t) │
                              │  Morphological Memory │
                              │  NatGrad on SPD       │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │   CREATIVITY ENGINE  │
                              │  Noise + Novelty     │
                              │  Metaphor Jumps       │
                              │  Soft-max selection   │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │    PLASTICITY        │
                              │  dG/dt = &minus;&alpha;C   │
                              │       + &beta;F        │
                              │  Erosion / Sediment   │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │    PROJECTOR         │
                              │  R^N &rarr; Text      │
                              │  Nearest neighbor     │
                              └─────────────────────┘
```

---

## Mathematical Foundations

### Dynamic Metric Tensor G_ij(t)

The semantic space is a Riemannian manifold whose metric evolves over time. When Kuramoto coupling is enabled, G(t) is not a static parameter but **morphological memory** — a third scaling dimension beyond params and data.

```python
# Initialization: quasi-Euclidean metric
G = I_N + epsilon * symmetric_perturbation

# Update with guaranteed positive-definiteness
G += dG
# Eigenvalue clamping: lambda_i >= 1e-6
```

**Christoffel Symbols** (vectorized via `np.einsum`):

&Gamma;<sup>k</sup><sub>ij</sub> = (1/2) G<sup>kl</sup> (&part;G<sub>li</sub>/&part;q<sup>j</sup> + &part;G<sub>lj</sub>/&part;q<sup>i</sup> &minus; &part;G<sub>ij</sub>/&part;q<sup>l</sup>)

```python
Gamma = 0.5 * np.einsum('kl,lij->kij', G_inv, dG_combined)  # (N, N, N)
```

**Geodesic Acceleration**:

q&#772;<sup>k</sup> = &minus;&Gamma;<sup>k</sup><sub>ij</sub> q&#775;<sup>i</sup> q&#775;<sup>j</sup>

```python
accel = -np.einsum('kij,i,j->k', Gamma, dq, dq)
```

**Scalar Curvature**:

R &asymp; (1/N) tr(G<sup>&minus;1</sup> &middot; &Delta; &middot; &Delta;)

### Semantic Lagrangian

L(q, q&#775;, t) = T(q, q&#775;) &minus; V(q, t)

Where kinetic energy is defined by the Riemannian metric:

T = (1/2) q&#775;<sup>T</sup> G q&#775;

And the potential combines soft confinement + external semantic force:

V(q) = V<sub>confinement</sub>(q) + V<sub>semantic</sub>(q)

Euler-Lagrange equations are solved numerically via **Runge-Kutta 4**:

```python
# RK4 step
k1_q, k1_dq = derivatives(q, dq, t)
k2_q, k2_dq = derivatives(q + 0.5*dt*k1_q, dq + 0.5*dt*k1_dq, t + 0.5*dt)
k3_q, k3_dq = derivatives(q + 0.5*dt*k2_q, dq + 0.5*dt*k2_dq, t + 0.5*dt)
k4_q, k4_dq = derivatives(q + dt*k3_q, dq + dt*k3_dq, t + dt)

q_new = q + (dt/6) * (k1_q + 2*k2_q + 2*k3_q + k4_q)
dq_new = dq + (dt/6) * (k1_dq + 2*k2_dq + 2*k3_dq + k4_dq)
```

### Syntopy &mdash; Topological Operator

The **&starf;** (syntopy) operator fuses two semantic manifolds without gradients:

&tau; = &nabla; &starf; (M<sub>query</sub> &oplus; M<sub>example</sub>)

- The example's topology "imprints" instantly onto the query
- **Zero gradient, zero fine-tuning**: the deformation is purely geometric
- Enables absolute one-shot: a single example constrains generation

### Encoding &mdash; Deterministic SHA-256

Each word is projected to a unique position in R^N via hashing:

```python
h = sha256(word.encode('utf-8')).hexdigest()
pos[i] = scale * tanh(normalize(h[start:start+4]))
```

Properties:
- **Deterministic**: same word = same position, always
- **Quasi-uniform**: SHA-256 guarantees homogeneous distribution
- **Unlimited**: no frozen vocabulary, any word can be encoded
- **TF-IDF**: rare words carry stronger weight

---

## ★ Kuramoto &harr; Metric Coupling (Morphological Memory)

This is the core innovation of MVT v3. The metric tensor G(t) is no longer a static learnable parameter &mdash; it is a **living dynamical system** coupled to a bank of Kuramoto oscillators.

### The Coupled System

```
    d&phi;<sub>i</sub>/dt = &omega;<sub>i</sub> + (K/N) &Sigma;<sub>j</sub> sin(&phi;<sub>j</sub> &minus; &phi;<sub>i</sub>) &middot; G<sub>ij</sub>(t)

    dG/dt = NatGrad<sub>SPD</sub>( L(q, q&#775;, G, &phi;) )
```

A **closed feedback loop**:
1. Kuramoto oscillators synchronize their phases &phi;(t), modulated by the metric G
2. Phase coherence structurally modulates G: synchronized pairs get stronger coupling
3. G is updated via Natural Gradient on the SPD manifold

G(t) becomes **morphological memory**: each integration step reshapes the geometry of semantic space. This introduces a third scaling dimension:

> **params &times; data &times; morphological_steps**

### Phase &rarr; Metric Modulation

G<sub>ij</sub> &larr; G<sub>ij</sub> &times; (1 + &epsilon; &middot; cos(&phi;<sub>i</sub> &minus; &phi;<sub>j</sub>))

Synchronized pairs (&phi;<sub>i</sub> &asymp; &phi;<sub>j</sub>) see their coupling reinforced. Desynchronized pairs see it weakened. The metric **pulses** with the internal rhythm of the semantic space.

### Natural Gradient on SPD Manifold

The metric lives on S<sup>+</sup><sub>n</sub> (symmetric positive definite matrices) with the affine-invariant Riemannian metric:

&langle;X, Y&rangle;<sub>G</sub> = tr(G<sup>&minus;1</sup> X G<sup>&minus;1</sup> Y)

The Riemannian gradient transforms the Euclidean gradient:

grad<sub>R</sub> f = G &middot; (&part;f/&part;G) &middot; G

Three retraction methods (keeping G on the manifold):

| Method | Formula | Cost | Stability |
|--------|---------|------|-----------|
| `exp` (exact) | G<sup>1/2</sup> expm(G<sup>&minus;1/2</sup> &eta; G<sup>&minus;1/2</sup>) G<sup>1/2</sup> | O(N&sup3;) | Exact |
| `approx2` | G + &eta; + (1/2) &eta; G<sup>&minus;1</sup> &eta; | O(N&sup3;) | **Recommended for CPU** |
| `cholesky` | L &middot; expm(L<sup>&minus;1</sup> &eta; L<sup>&minus;T</sup>) &middot; L<sup>T</sup> | O(N&sup3;) | Most stable |

### Stability: Trace Normalization

The critical problem with NatGrad is that `grad_R = G &middot; &nabla; &middot; G` causes quadratic amplification. MVT solves this with explicit boundedness constraints:

**Loss function with regularization:**

L = (T &minus; V) + &alpha; &middot; (tr(G) &minus; N)&sup2; + &gamma; &middot; &Sigma;(log &lambda;<sub>i</sub>)&sup2;

- **Trace penalty**: anchors tr(G) &asymp; N, prevents explosion
- **Log-eigenvalue penalty**: prevents collapse/explosion of individual eigenvalues
- **Post-retraction rescaling**: if |tr(G) &minus; N| > 0.5N, rescale G *= N/tr(G)

**Verified stability over 20 consecutive generations**: condition number &le; 1.07, eigenvalues in [0.97, 1.37].

```python
# Enable Kuramoto coupling — replaces static MetricTensor
config = MVTConfig(
    kuramoto_enabled=True,
    kuramoto_coupling_K=2.0,       # Kuramoto coupling strength
    kuramoto_n_oscillators=128,    # Oscillator count (= ambient_dim)
    kuramoto_metric_lr=0.001,      # NatGrad learning rate
    kuramoto_retraction="approx2",  # Retraction method
    kuramoto_phase_coupling=0.1,   # Phase → metric coupling (epsilon)
)

model = MVT(config)
# model.coupler wraps model.metric — backward compatible
# G evolves at every RK4 step via coupled_step(q, dq)
```

### Learning Verification

G successfully encodes Kuramoto phase structure:

```
  Correlation(G, cos(Δφ)): 0.044 → 0.959  (100 steps)
  Phase coherence:          0.870 → 0.982  (100 steps)
  G remains SPD:            ✓ (all eigenvalues > 0)
```

---

## Creativity Engine

MVT is not a deterministic LLM. It has a 5-layer creativity engine:

| Layer | Mechanism | Effect |
|-------|-----------|--------|
| **Temperature** | Gaussian noise in acceleration | Stochastic exploration |
| **Novelty-seeking** | 1/r repulsion from visited regions | Anti-stagnation, anti-repetition |
| **Metaphor Jumps** | Probabilistic discontinuity (10%) | Novel associations |
| **Soft-max selection** | Probabilistic word election | Lexical diversity |
| **Divergent thinking** | Parallel trajectories | Multi-path exploration |

```python
# Thermal noise
noise = randn(N) * noise_scale * temperature * (1 + tanh(r/5))

# Novelty force
force = sum(novelty_bias / (dist^2 + 0.1) * direction)
```

**Soft anti-hallucination**: singularities are *detected* (curvature > threshold) but *never blocking*. Generation continues with a marker. Default threshold: 100.0 (relaxed 10-100x vs conservative models).

**Word selection**: soft-max with diversity penalty:

```python
probs = softmax(-distances / temperature)
probs[word] *= diversity_penalty ** (1 + repeat_count)
selected = sample(candidates, probs)
```

---

## Autonomous Agent (MVSAgent)

```
    ┌──────────────────────────────────────────────┐
    │              AGENTIC LOOP                      │
    │                                               │
    │   OBSERVE ──► PLAN ──► GENERATE ──► REFLECT  │
    │       ▲                                   │   │
    │       │         ◄── REPLAN ◄─── ◄────────┘   │
    │       │                                       │
    │       └──── (dissatisfaction) ─────────────── │
    └──────────────────────────────────────────────┘
```

The agent is **agentic by nature** &mdash; it doesn't just generate text, it:

1. **Observes**: analyzes the prompt, identifies key concepts, measures complexity
2. **Plans**: decomposes into semantic sub-goals in R^N
3. **Generates**: integrates with goal-directed steering
4. **Reflects**: evaluates satisfaction, creativity, coherence (scores 0-1)
5. **Replans**: adjusts temperature, metaphor_prob, and target if insufficient

```python
agent = MVSAgent(MVTConfig(max_agent_iterations=5))
result = agent.run("Imagine a world where gravity doesn't exist", verbose=True)
```

**Episodic memory**: every trace (prompt, result, score, time) is stored for future adaptation.

---

## MoE-MVT & Expert Decoupled Training

### Sparse MoE Architecture

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

Each expert is a "topological specialist" for a semantic region. The router dispatches each token to the `top_k` most relevant experts. Only a fraction of parameters is active per token.

**Weight tying**: `lm_head.weight = embed.tok_embed.weight` &rarr; parameter reduction.

### EDT &mdash; Expert Decoupled Training

The EDT pipeline trains **each component independently**, then briefly aligns them. Claimed result: **189x acceleration** vs standard training.

```
  PHASE 1                    PHASE 2a                    PHASE 2b
  ┌─────────────┐           ┌─────────────┐           ┌─────────────┐
  │  EXPERTS     │           │  ATTENTION   │           │  EMBEDDING  │
  │  Independent │           │  Independent │           │  Separable   │
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
              │  All unfrozen   │
              │  PGSG rotation  │
              │  CE + aux_loss │
              │  500K tokens   │
              └─────────────────┘
```

### PGSG &mdash; Partial Gradient Sequential Update

Only `n_active_layers` out of `n_layers` total receive gradients per step. Circular rotation: layers skipped this step will be active next.

```python
# Step 0: layers 0,1 active (out of 4)
# Step 1: layers 1,2 active
# Step 2: layers 2,3 active
# Step 3: layers 3,0 active
# Step 4: layers 0,1 active (cycle)
```

**Result**: ~60% backprop reduction, ideal for CPU.

---

## Chinchilla Scaling for MVT

MVT is not a transformer &mdash; standard Chinchilla scaling (20 tokens/param) does not apply. The MVT Chinchilla benchmark derives the true scaling law for differential geometry architectures.

### Key Differences

| Property | Transformer | MVT |
|----------|-------------|-----|
| Compute per step | O(N) attention | O(N&sup4;) Christoffel symbols |
| Data unit | Discrete tokens | Continuous trajectory steps |
| Metric | Static (learned) | Dynamic (morphological memory) |
| Scaling dims | params &times; data | params &times; data &times; morphological_steps |

### Empirical Results (CPU benchmark)

| N (ambient_dim) | Step time | Christoffel time | Practical max N |
|----------------|-----------|-----------------|-----------------|
| 32 | ~3ms | ~0.4ms | 256 |
| 64 | ~35ms | ~5ms | 128 |
| 128 | ~597ms | ~98ms | 64 |
| 256 | ~10,000ms+ | ~1,600ms | 32 |

**Scaling law**: optimal D/N &asymp; 3&ndash;7 samples per parameter (vs 20 tokens/param for transformers). The O(N&sup4;) Christoffel bottleneck is the dominant cost.

```bash
cd scripts && python -m mvt.chinchilla_benchmark
```

### Practical CPU Guidelines

- **Recommended `d_model`**: &le; 128 for real-time interaction on CPU
- **RK4 steps**: 30&ndash;200 (reduce for larger N)
- **Chinchilla ratio**: D/N &asymp; 3&ndash;7 (continuous trajectory samples per parameter)

---

## Usage

### Installation

```bash
pip install numpy torch
```

### Creative Generation (Kuramoto Coupled)

```python
from mvt import MVTConfig, MVT

config = MVTConfig(
    ambient_dim=64,
    temperature=0.8,
    novelty_bias=0.4,
    metaphor_jump_prob=0.1,
    kuramoto_enabled=True,           # Enable morphological memory
    kuramoto_coupling_K=2.0,        # Kuramoto coupling strength
    kuramoto_metric_lr=0.001,      # NatGrad learning rate
    kuramoto_phase_coupling=0.1,    # Phase → metric coupling
)

model = MVT(config)
result = model.generate("consciousness emerges from", temperature=0.9)

print(result.text)
print(f"Creativity : {result.creativity_score:.2f}")
print(f"Diversity  : {result.diversity_score:.2f}")
print(f"Metric det : {result.metrics['metric_det']:.4f}")
print(f"Metric cond: {result.metrics['metric_condition']:.2f}")

# Live Kuramoto sync state
stats = model.get_stats()
print(f"Kuramoto r : {stats['kuramoto']['order_parameter']:.4f}")
print(f"Coherence  : {stats['kuramoto']['phase_coherence']:.4f}")
```

### Static Mode (backward compatible)

```python
# kuramoto_enabled=False (default) — uses static MetricTensor
config = MVTConfig(
    ambient_dim=64,
    temperature=0.8,
    kuramoto_enabled=False,  # Static G, no Kuramoto
)
model = MVT(config)
result = model.generate("The universe expands", temperature=0.8)
```

### Autonomous Agent

```python
from mvt import MVTConfig, MVSAgent

config = MVTConfig(
    ambient_dim=64,
    max_agent_iterations=5,
    self_reflection_threshold=0.5,
    goal_steering_strength=0.3,
    kuramoto_enabled=True,
)

agent = MVSAgent(config)
result = agent.run("Imagine a world where gravity doesn't exist", verbose=True)
```

### One-Shot Learning (Syntopy)

```python
from mvt import MVTConfig, MVT

model = MVT(MVTConfig(ambient_dim=64, kuramoto_enabled=True))

# Learn a style from ONE example — zero gradient
model.set_example("The cat sleeps on the couch. The dog runs in the garden.")

# The example's topology constrains generation
result = model.generate("The fish swims in", temperature=0.8)
print(result.text)
print(f"Syntopy score: {result.syntopy_score:.2f}")

model.clear_example()
```

### Natural Gradient on SPD (standalone)

```python
from mvt import NaturalGradientSPD
import numpy as np

N = 64
G = np.eye(N) + 0.1 * np.random.randn(N, N)
G = 0.5 * (G + G.T)

ng = NaturalGradientSPD(lr=0.01, retraction_order="approx2")

for step in range(100):
    grad_E = 0.01 * np.random.randn(N, N)  # Some loss gradient
    grad_E = 0.5 * (grad_E + grad_E.T)
    G = ng.retract(G, grad_E)
    # G stays on S^+_n by construction — no projection needed

# Geodesic distance
d = ng.geodesic_distance(G, np.eye(N))
```

### MoE-MVT + EDT (PyTorch, CPU)

```python
from mvt.edt import MoEMVT, MoEMVTConfig, EDTConfig, run_edt, generate_synthetic_corpus

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
    save_dir="./checkpoints",
)

model = MoEMVT(model_cfg)
corpus = generate_synthetic_corpus(vocab_size=4000, length=50_000)
stats = run_edt(model, corpus, edt_cfg, verbose=True)
```

### Chinchilla Benchmark

```bash
cd scripts && python -m mvt.chinchilla_benchmark
```

---

## Full Configuration

```python
@dataclass
class MVTConfig:
    # === Space ===
    ambient_dim: int = 128           # Dimension of R^N
    intrinsic_dim: int = 64         # Intrinsic dimension

    # === Integration ===
    dt: float = 0.01                # RK4 time step
    num_rk4_steps: int = 200        # Number of integration steps

    # === Plasticity ===
    alpha_erosion: float = 0.05     # Erosion rate
    beta_sedimentation: float = 0.03 # Sedimentation rate

    # === Lagrangian ===
    damping: float = 0.05           # Damping (low = creative)
    potential_stiffness: float = 2.0 # Confinement stiffness

    # === Creativity ===
    temperature: float = 0.8        # Stochastic noise
    novelty_bias: float = 0.4       # Novelty-seeking strength
    metaphor_jump_prob: float = 0.1 # Metaphor jump probability

    # === Agent ===
    max_agent_iterations: int = 5
    self_reflection_threshold: float = 0.5
    goal_steering_strength: float = 0.3

    # === Projection ===
    projection_temperature: float = 0.7
    diversity_penalty: float = 0.3
    max_consecutive_repeat: int = 2

    # === Kuramoto <-> Metric Coupling ===
    kuramoto_enabled: bool = False          # Enable morphological memory
    kuramoto_coupling_K: float = 1.0         # Kuramoto coupling strength
    kuramoto_n_oscillators: int = None      # Oscillator count (None = ambient_dim)
    kuramoto_metric_lr: float = 0.001       # NatGrad learning rate
    kuramoto_retraction: str = "approx2"    # 'exp', 'approx2', 'cholesky'
    kuramoto_phase_coupling: float = 0.1    # Phase → metric coupling (epsilon)
    kuramoto_phase_init: str = "random"     # 'random', 'sync', 'cluster'

    # === Stability (relaxed for creativity) ===
    curvature_threshold: float = 100.0
    divergence_threshold: float = 1e8
```

---

## Tests

```bash
cd scripts && python -m mvt.tests.py -v          # 39 unit tests
cd scripts && python -m mvt.test_toy_training     # 5 integration tests (Kuramoto)
```

All tests pass. Integration tests verify:
- MVT + KuramotoMetricCoupler end-to-end generation
- Static vs Kuramoto mode comparison
- Stability over 20 consecutive generations (G stays SPD, cond &le; 1.07)
- Standalone coupler integration
- Reset and stats

```
core/test_metric_tensor .......... OK  (Christoffel, geodesic, curvature)
core/test_vector_field ........... OK  (Coulomb, potential, divergence)
lagrangian/test_semantic_lag ..... OK  (T, V, Euler-Lagrange)
lagrangian/test_integrator ........ OK  (RK4, action, divergence)
test_encoder .................... OK  (SHA-256, TF-IDF, similarity)
test_syntopy .................... OK  (fusion, syntopy score)
test_plasticity ................. OK  (erosion, sedimentation)
test_creativity ................. OK  (noise, novelty, jumps, soft-max)
test_model ...................... OK  (generation, singularities)
test_agent ...................... OK  (loop, reflection, replan)
test_toy_training ............... OK  (5 integration tests w/ Kuramoto)
```

---

## Demos

```bash
cd scripts && python -m mvt.demo
```

5 interactive demos:
1. **Creative generation** &mdash; Temperature, diversity, metaphor jumps
2. **Autonomous agent** &mdash; Observe/plan/generate/reflect loop
3. **One-shot syntopy** &mdash; Learn a style from one example
4. **Plasticity** &mdash; Self-evolution of the metric
5. **Creative agent** &mdash; Complex task with replanning

---

## EDT Training

```bash
cd scripts && python -m mvt.edt.run_edt
```

Full Phase 1 &rarr; 2a &rarr; 2b &rarr; 3 pipeline with intermediate checkpoints.

---

## Project Structure

```
mvt/
├── __init__.py                       # MVTConfig, MVT, MVSAgent, CreativityEngine
│                                     # KuramotoMetricCoupler, NaturalGradientSPD
├── config.py                         # 30+ hyperparameters
├── model.py                          # Main orchestrator v3 (Kuramoto-coupled)
├── encoder.py                        # SHA-256 + TF-IDF → R^N
├── syntopy.py                        # ★ Topological fusion operator
├── plasticity.py                     # dG/dt = -alpha*C + beta*F
├── projector.py                      # R^N → Text (nearest neighbor)
├── creativity.py                     # Noise, novelty, jumps, soft-max
├── agent.py                          # Observe→Plan→Generate→Reflect→Replan
├── kuramoto_metric.py               # ★ Kuramoto ↔ G(t) coupled system
├── natural_gradient_spd.py          # ★ NatGrad on SPD manifold (3 retractions)
├── chinchilla_benchmark.py           # ★ MVT scaling law benchmark
├── test_toy_training.py             # ★ 5 integration tests
├── demo.py                           # 5 interactive demos
├── tests.py                          # 39 unit tests
│
├── core/
│   ├── metric_tensor.py              # G_ij(t), Christoffel, curvature (einsum)
│   └── vector_field.py               # Coulomb N-dim, potential, divergence
│
├── lagrangian/
│   ├── semantic_lagrangian.py        # L = T - V, Euler-Lagrange
│   └── integrator.py                 # RK4, action, optimal trajectory
│
└── edt/
    ├── moe_model.py                  # MoE-MVT: Router + Experts + Attention
    ├── edt_pipeline.py                # 4 phases + PGSG
    └── run_edt.py                     # CPU training script
```

---

## Performance

| Operation | Complexity | Implementation |
|-----------|-----------|----------------|
| Christoffel Symbols | O(N&sup3;) | `np.einsum('kl,lij->kij')` &mdash; vectorized |
| RK4 Step | O(N&sup2;) | 4 evaluations + combination |
| Geodesic Distance | O(N&sup2;) | `sqrt(delta^T G delta)` |
| Kuramoto Step | O(N&sup2;) | Coupling weighted by G |
| NatGrad Retraction | O(N&sup3;) | `approx2`: G + &eta; + &frac12;&eta;G<sup>&minus;1</sup>&eta; |
| MoE Forward | O(k d&sup2; f) | k active experts out of n total |
| EDT Phase 1 | O(E S d&sup2;) | E experts, S steps, parallelizable |
| PGSG | &minus;60% backprop | Circular layer rotation |

**CPU Optimization**: Christoffel vectorized via einsum (73s &rarr; 1.1s for N=128), PGSG reduces gradient computation, separable embedding for Phase 2b, `approx2` retraction avoids expensive matrix exponential.

**Kuramoto overhead**: ~24% per generation step (measured on N=32, 50 RK4 steps).

---

## Author

**[AFKmoney](https://github.com/AFKmoney)**

## License

MIT
