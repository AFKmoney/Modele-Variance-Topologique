<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/NumPy-Differential_Geometry-orange?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
<img src="https://img.shields.io/badge/PyTorch-MoE_%26_EDT-red?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
<img src="https://img.shields.io/badge/CPU-Optimized-success?style=for-the-badge" alt="CPU"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
<img src="https://img.shields.io/badge/Tests-39%20Passed-cyan?style=for-the-badge" alt="Tests"/>

<br/><br/>

# MVT — Topological Variance Model

### AI doesn't read tokens anymore. It surfs on geometric manifolds.

<br/>

<code>pip install numpy torch</code>

<br/>

[Architecture](#-architecture) ·
[Mathematics](#-mathematical-foundations) ·
[Creativity](#-creativity-engine) ·
[Autonomous Agent](#-autonomous-agent-mvsagent) ·
[MoE + EDT](#-moe-mvt--expert-decoupled-training) ·
[Usage](#-usage) ·
[Config](#-configuration)

<br/>
<br/>

```
  Transformers:                  MVT:

  Token → Embed → Attention    Word → Position in R^N →
    → FFN → Token → ...           Coulomb Force Field  →
                                   Lagrangian (RK4) →
                                   Geodesic Trajectory  →
                                   Continuous Projection → Text
```

</div>

---

## Why

Transformers are constrained by three fundamental assumptions:

1. **Discretization**: Language is chopped into tokens, one by one. No semantic continuity.
2. **Global Attention**: Every token "looks at" every other. Quadratic in sequence length.
3. **No Geometry**: The latent space is flat (Euclidean). No structure, no curvature.

MVT rejects all three.

**Fundamental Postulate**: The meaning of text is a *living Riemannian manifold* in R^N. Language generation is the motion of an idea particle on this manifold, governed by the semantic Lagrangian.

There are no tokens. There is no attention. There is a continuous force field, a curved metric, and a geodesic trajectory.

---

## Architecture

```
                              ┌──────────────────────┐
                              │   INPUT ENCODER      │
                              │  SHA-256 + TF-IDF    │
                              │  Word → R^N (pos)    │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼───────────┐
                              │   VECTOR FIELD       │
                              │  Sources / Sinks     │
                              │  Coulomb in N-dim    │
                              └──────────┬───────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
   ┌──────────▼───────────┐  ┌──────────▼───────────┐  ┌──────────▼───────────┐
   │    SYNTOPY ★        │  │   METRIC TENSOR      │  │  SEMANTIC LAGRANGIAN │
   │  Topological fusion │  │   G_ij(t)            │  │  L = T - V            │
   │  Absolute one-shot   │  │  Christoffel symbols  │  │  Euler-Lagrange       │
   │  Zero gradients      │  │  Scalar curvature R   │  │  RK4 Integrator       │
   └──────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘
              │                         │                         │
              └─────────────────────────┼─────────────────────────┘
                                         │
                              ┌──────────▼───────────┐
                              │  CREATIVITY ENGINE   │
                              │  Noise + Novelty      │
                              │  Metaphor Jumps       │
                              │  Soft-max Selection   │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼───────────┐
                              │    PLASTICITY         │
                              │  dG/dt = -α·C + β·F  │
                              │  Erosion / Sediment   │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼───────────┐
                              │    PROJECTOR          │
                              │  R^N → Text           │
                              │  Nearest Neighbor     │
                              └──────────────────────┘
```

---

## Mathematical Foundations

### Dynamic Metric Tensor G_ij(t)

The semantic space is a Riemannian manifold whose metric evolves over time.

```python
# Initialization: quasi-Euclidean metric
G = I_N + ε · symmetric_perturbation

# Update with guaranteed positive-definiteness
G += dG
# Eigenvalue clamping: λ_i >= 1e-6
```

**Christoffel Symbols** (vectorized via `np.einsum`):

$$\Gamma^k_{ij} = \frac{1}{2} G^{kl} \left( \frac{\partial G_{li}}{\partial q^j} + \frac{\partial G_{lj}}{\partial q^i} - \frac{\partial G_{ij}}{\partial q^l} \right)$$

```python
Gamma = 0.5 * np.einsum('kl,lij->kij', G_inv, dG_combined)  # (N, N, N)
```

**Geodesic Acceleration**:

$$\ddot{q}^k = -\Gamma^k_{ij} \, \dot{q}^i \, \dot{q}^j$$

```python
accel = -np.einsum('kij,i,j->k', Gamma, dq, dq)
```

**Scalar Curvature**:

$$R \approx \frac{1}{N} \text{tr}(G^{-1} \cdot \Delta \cdot \Delta)$$

### Semantic Lagrangian

$$\mathcal{L}(q, \dot{q}, t) = T(q, \dot{q}) - V(q, t)$$

Where kinetic energy is defined by the Riemannian metric:

$$T = \frac{1}{2} \dot{q}^T \, G \, \dot{q}$$

And the potential combines soft confinement + external semantic force:

$$V(q) = V_{\text{confinement}}(q) + V_{\text{semantic}}(q)$$

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

### Syntopy — Topological Operator

The **★** operator (syntopy) fuses two semantic manifolds without gradients:

$$\tau = \nabla \star (M_{\text{query}} \oplus M_{\text{example}})$$

- The example's topology "imprints" instantly on the query
- **Zero gradient, zero fine-tuning**: the deformation is purely geometric
- Enables absolute one-shot: a single example is enough to constrain generation

### Encoding — Deterministic SHA-256

Each word is projected to a unique position in R^N via hashing:

```python
h = sha256(word.encode('utf-8')).hexdigest()
pos[i] = scale * tanh(normalize(h[start:start+4]))
```

Properties:
- **Deterministic**: same word = same position, always
- **Quasi-uniform**: SHA-256 guarantees homogeneous distribution
- **Unlimited**: no fixed vocabulary, any word can be encoded
- **TF-IDF weighted**: rare words carry more weight

---

## Creativity Engine

MVT isn't a deterministic LLM. It has a 5-layer creativity engine:

| Layer | Mechanism | Effect |
|-------|-----------|--------|
| **Temperature** | Gaussian noise in acceleration | Stochastic exploration |
| **Novelty-seeking** | 1/r repulsion from visited regions | Anti-stagnation, anti-repetition |
| **Metaphor Jumps** | Probabilistic discontinuity (10%) | Novel associations |
| **Soft-max Selection** | Probabilistic word election | Lexical diversity |
| **Divergent Thinking** | Parallel trajectories | Multi-path exploration |

```python
# Thermal noise
noise = randn(N) * noise_scale * temperature * (1 + tanh(r/5))

# Novelty force — repels from visited regions
force = sum(novelty_bias / (dist^2 + 0.1) * direction)

# Metaphor jump — random teleport with amplitude control
if random() < metaphor_jump_prob * temperature:
    q = q + amplitude * random_direction
```

**Soft anti-hallucination**: singularities are *detected* (curvature > threshold) but *never blocking*. Generation continues with a marker. Default threshold: 100.0 (relaxed 10-100x vs. conservative models).

```python
if abs(curvature) > threshold:
    return True, step  # Signals, but does NOT block
```

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
    │              AGENTIC LOOP                    │
    │                                               │
    │  OBSERVE ──► PLAN ──► GENERATE ──► REFLECT  │
    │      ▲                                   │   │
    │      │         ◄── REPLAN ◄─── ◄────────┘   │
    │      │                                       │
    │      └──── (dissatisfaction) ──────────────── │
    └──────────────────────────────────────────────┘
```

The agent is **agentic by nature** — it doesn't just generate text, it:

1. **Observes**: analyzes the prompt, identifies key concepts, measures complexity
2. **Plans**: decomposes into semantic sub-goals (Goals) in R^N
3. **Generates**: integrates with goal-directed steering toward targets
4. **Reflects**: evaluates satisfaction, creativity, coherence (scores 0-1)
5. **Replans**: adjusts temperature, metaphor_prob, and target if unsatisfied

```python
agent = MVSAgent(MVTConfig(max_agent_iterations=5))
result = agent.run("Imagine a world where gravity doesn't exist", verbose=True)

# The agent:
# - Observes prompt complexity
# - Creates 1-2 sub-goals (exploration + creativity)
# - Generates, evaluates, and optionally replans
# - Returns the best result after 1-5 iterations
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
                    │  LM_HEAD (weight tie)  │
                    └───────────────────────┘
```

Each expert is a "topological specialist" of a semantic region. The router distributes each token to the `top_k` most relevant experts. Only a fraction of parameters is active per token.

**Weight tying**: `lm_head.weight = embed.tok_embed.weight` — parameter reduction.

### EDT — Expert Decoupled Training

The EDT pipeline trains **each component independently**, then briefly aligns them. Claimed result: **189x speedup** vs. standard training.

```
  PHASE 1                    PHASE 2a                    PHASE 2b
  ┌─────────────┐           ┌─────────────┐           ┌─────────────┐
  │  EXPERTS     │           │  ATTENTION   │           │  EMBEDDING  │
  │  Independent │           │  Independent │           │  Separable   │
  │              │           │              │           │             │
  │  MSE(h_in,   │           │  MSE(h_in +  │           │  Next-token  │
  │   h_target)  │           │   attn,      │           │  prediction │
  │              │           │   h_target)  │           │             │
  │  150 steps   │           │  300 steps   │           │  2M tokens  │
  │  / expert    │           │  / layer     │           │             │
  └──────┬──────┘           └──────┬──────┘           └──────┬──────┘
         │                         │                         │
         └─────────────┬───────────┘                         │
                       │                                     │
              ┌────────▼────────┐                            │
              │  PHASE 3         │◄───────────────────────────┘
              │  JOINT FINE-TUNE │
              │                  │
              │  All unfrozen    │
              │  PGSG rotation   │
              │  CE + aux_loss   │
              │  500K tokens     │
              └──────────────────┘
```

### PGSG — Partial Gradient Sequential Update

Only `n_active_layers` out of `n_layers` total receive gradients per step. Circular rotation: layers skipped this step will be active in subsequent steps.

```python
# Step 0: layers 0,1 active (out of 4)
# Step 1: layers 1,2 active
# Step 2: layers 2,3 active
# Step 3: layers 3,0 active
# Step 4: layers 0,1 active (cycle)
```

**Result**: ~60% backprop reduction, ideal for CPU.

---

## Usage

### Install

```bash
pip install numpy torch
```

### Creative Generation

```python
from mvt import MVTConfig, MVT

config = MVTConfig(
    ambient_dim=16,              # Semantic space dimension
    temperature=0.8,             # 0 = deterministic, 1 = very creative
    novelty_bias=0.4,           # Novelty-seeking force
    metaphor_jump_prob=0.1,     # 10% chance of metaphor jump per step
)

model = MVT(config)
result = model.generate("consciousness emerges from", temperature=0.9)

print(result.text)                          # Generated text
print(f"Creativity : {result.creativity_score:.2f}")
print(f"Diversity  : {result.diversity_score:.2f}")
print(f"Time       : {result.generation_time:.4f}s")
print(f"Singularity: {result.has_singularity}")

# Adjust temperature on the fly
model.set_temperature(1.2)  # Hotter
model.set_temperature(0.3)  # Colder
```

### Autonomous Agent

```python
from mvt import MVTConfig, MVSAgent

config = MVTConfig(
    ambient_dim=16,
    max_agent_iterations=5,
    self_reflection_threshold=0.5,
    goal_steering_strength=0.3,
)

agent = MVSAgent(config)
result = agent.run("Imagine a world where gravity doesn't exist", verbose=True)
```

### One-Shot Learning (Syntopy)

```python
from mvt import MVTConfig, MVT

model = MVT(MVTConfig(ambient_dim=16))

# Learn a style from ONE example — zero gradients
model.set_example("The quantum field oscillates with infinite possibilities")

# The example's topology constrains generation
result = model.generate("consciousness and", temperature=0.8)
print(result.text)
print(f"Syntopy score: {result.syntopy_score:.2f}")

model.clear_example()
```

### MoE-MVT + EDT (PyTorch, CPU)

```python
from mvt.edt import MoEMVT, MoEMVTConfig, EDTConfig, run_edt, generate_synthetic_corpus

# Model configuration
model_cfg = MoEMVTConfig(
    vocab_size=4000,     # Vocabulary size
    d_model=128,         # Model dimension
    n_layers=4,          # Transformer blocks
    n_experts=8,         # Experts per MoE layer
    top_k=2,             # Active experts per token
    d_ff=256,            # Intermediate dimension
    max_seq_len=64,      # Max sequence length
)

# EDT configuration (CPU-optimized)
edt_cfg = EDTConfig(
    phase1_steps_per_expert=50,
    phase1_hidden_samples=500,
    phase2a_steps_per_layer=100,
    phase2b_n_tokens=100_000,
    phase3_n_tokens=50_000,
    phase3_n_active_layers=2,       # PGSG: 2 active / 4 total
    save_dir="./checkpoints",
)

# Train
model = MoEMVT(model_cfg)
corpus = generate_synthetic_corpus(vocab_size=4000, length=50_000)
stats = run_edt(model, corpus, edt_cfg, verbose=True)

total, active = model.count_params()
print(f"Params: {total:,} total, {active:,} active/token")
print(f"Sparsity: {1 - active/total:.1%}")
```

### Topological Plasticity

```python
from mvt import MVTConfig, MVT

model = MVT(MVTConfig(ambient_dim=16))

# Generate multiple times — the metric evolves
for prompt in ["Music is a language", "Colors dance under light"]:
    result = model.generate(prompt)
    stats = result.plasticity_stats
    print(f"Barriers: {stats['total_barriers']}, Channels: {stats['total_channels']}")
    print(f"det(G): {stats['metric_det']:.4f}")

# The metric has self-modified:
# - Erosion where trajectories failed
# - Sedimentation where they succeeded
```

---

## Configuration

```python
@dataclass
class MVTConfig:
    # === Space ===
    ambient_dim: int = 128          # R^N dimension
    intrinsic_dim: int = 64        # Intrinsic dimension

    # === Integration ===
    dt: float = 0.01                # RK4 time step
    num_rk4_steps: int = 200        # Number of steps

    # === Plasticity ===
    alpha_erosion: float = 0.05      # Erosion rate
    beta_sedimentation: float = 0.03  # Sedimentation rate

    # === Lagrangian ===
    damping: float = 0.05           # Low damping = creative
    potential_stiffness: float = 2.0 # Confinement stiffness

    # === Creativity ===
    temperature: float = 0.8         # Stochastic noise
    novelty_bias: float = 0.4        # Novelty-seeking
    metaphor_jump_prob: float = 0.1  # Metaphor jumps

    # === Agent ===
    max_agent_iterations: int = 5     # Max loops
    self_reflection_threshold: float = 0.5
    goal_steering_strength: float = 0.3

    # === Projection ===
    projection_temperature: float = 0.7
    diversity_penalty: float = 0.3
    max_consecutive_repeat: int = 2

    # === Stability (relaxed for creativity) ===
    curvature_threshold: float = 100.0
    divergence_threshold: float = 1e8
```

---

## Tests

```bash
python -m pytest mvt/tests.py -v
```

39 tests covering every component — pass in **0.58s**.

---

## Demos

```bash
python -m mvt.demo
```

5 interactive demos:
1. **Creative generation** — Temperature, diversity, metaphor jumps
2. **Autonomous agent** — Observe/plan/generate/reflect loop
3. **One-shot syntopy** — Learn a style from one example
4. **Plasticity** — Metric self-evolution
5. **Creative agent** — Complex task with replanning

---

## EDT Training

```bash
python -m mvt.edt.run_edt
```

Full Phase 1 → 2a → 2b → 3 pipeline with intermediate checkpoints.

---

## Project Structure

```
mvt/
├── __init__.py                  # MVTConfig, MVT, MVSAgent, CreativityEngine
├── config.py                    # 25+ hyperparameters
├── model.py                     # Main orchestrator v2
├── encoder.py                   # SHA-256 + TF-IDF → R^N
├── syntopy.py                   # Topological fusion operator ★
├── plasticity.py                # dG/dt = -α·C + β·F
├── projector.py                 # R^N → Text (nearest neighbor)
├── creativity.py                # Noise, novelty, jumps, soft-max
├── agent.py                     # Observe → Plan → Generate → Reflect → Replan
├── demo.py                      # 5 interactive demos
├── tests.py                     # 39 unit tests
│
├── core/
│   ├── metric_tensor.py         # G_ij(t), Christoffel, curvature (einsum)
│   └── vector_field.py          # Coulomb N-dim, potential, divergence
│
├── lagrangian/
│   ├── semantic_lagrangian.py   # L = T - V, Euler-Lagrange
│   └── integrator.py            # RK4, action, optimal trajectory
│
└── edt/
    ├── moe_model.py             # MoE-MVT: Router + Experts + Attention
    ├── edt_pipeline.py           # 4 phases + PGSG
    └── run_edt.py                # CPU training script
```

---

## Performance

| Operation | Complexity | Implementation |
|-----------|-----------|---------------|
| Christoffel Symbols | O(N³) | `np.einsum('kl,lij->kij')` — vectorized |
| RK4 Step | O(N²) | 4 evaluations + combination |
| Geodesic Distance | O(N²) | `sqrt(delta^T G delta)` |
| MoE Forward | O(k · d² · f) | k active experts out of n total |
| EDT Phase 1 | O(E · S · d²) | E experts, S steps, parallelizable |
| PGSG | -60% backprop | Circular layer rotation |

**CPU Optimization**: Christoffel vectorized via einsum (73s → 1.1s for N=128), PGSG reduces gradient computation, separable embedding for Phase 2b.

---

## Author

**[AFKmoney](https://github.com/AFKmoney)**

## License

MIT
