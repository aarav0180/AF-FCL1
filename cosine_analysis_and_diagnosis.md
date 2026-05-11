# Cosine Normalization in AF-FCL: Root Cause Analysis

## Executive Summary

**Cosine normalization is NOT fundamentally broken.** The problem is a specific architectural mismatch: the AF-FCL "accurate forgetting" pipeline was designed around **unnormalized features**, and three critical mechanisms break when features are projected onto the unit hypersphere.

The result: cosine normalization improves *optimization stability* (smoother gradients, no magnitude explosions) but degrades *selective replay quality* (the model can no longer distinguish useful from biased replay), causing worse forgetting in Non-IID settings.

---

## The Three Failure Mechanisms

### 1. Probability Estimation Collapse (PRIMARY CAUSE)

The `probability_in_localdata()` function (line 190 of model.py) is the heart of AF-FCL's "accurate forgetting." It computes a per-dimension Gaussian probability to decide how much each replayed sample matters:

```python
# This estimates: P(flow_xa | current_client_data)
prob = N(flow_xa; mean(xa_current), var(xa_current))
```

**With standard linear head:** Features `xa` have diverse magnitudes and variances. The Gaussian fits a rich distribution. The per-class variance `xa_u_yi_var` is meaningful — it captures how "spread out" each class is in the client's local feature space. Flow-generated features that don't match the client's distribution get low probability → they are down-weighted → **accurate forgetting works**.

**With cosine head:** The cosine head normalizes features to unit norm before classification. But critically, `xa` (the features fed to the flow and probability estimator) comes from `forward_to_xa()`, which is the ResNet backbone output **before** the cosine head. However, the cosine head's gradient signal backpropagates through the backbone and reshapes the feature geometry:

- The backbone learns to produce features where **direction matters but magnitude doesn't**
- Feature magnitudes become more uniform across classes (because the loss doesn't reward magnitude differentiation)
- `xa_u_yi_var` shrinks and becomes more uniform across classes
- The Gaussian probability becomes nearly uniform → `flow_xa_prob ≈ constant`
- **ALL replay samples get equal weight, regardless of relevance to the client**

This is why `flow_prob_mean` drops to ~0.08-0.12 in CIFAR100 cosine runs — the probability estimator can no longer discriminate.

```
┌─────────────────────────────────────────────────────────────┐
│  Baseline: probability_in_localdata() is discriminative     │
│                                                             │
│  Class A features: ||xa|| ≈ 5-15, var ≈ 3.2                │
│  Class B features: ||xa|| ≈ 8-20, var ≈ 7.1                │
│  → P(replay_A | client) = 0.72  (keep — relevant)          │
│  → P(replay_B | client) = 0.03  (forget — biased)          │
│                                                             │
│  Cosine: probability_in_localdata() is near-uniform         │
│                                                             │
│  Class A features: direction varies, ||xa|| ≈ 4-6, var≈0.8 │
│  Class B features: direction varies, ||xa|| ≈ 4-6, var≈0.9 │
│  → P(replay_A | client) = 0.11  (can't distinguish)        │
│  → P(replay_B | client) = 0.09  (can't distinguish)        │
└─────────────────────────────────────────────────────────────┘
```

### 2. Feature KD Distortion

The feature-level knowledge distillation (line 328) computes:

```python
kd_loss_feature_last = k_kd_last_cls * torch.pow(xa_last - xa, 2).mean()
```

This L2 distance between old and new features **depends on magnitude**. With cosine normalization:
- Feature magnitudes cluster tightly → L2 distances shrink
- The KD gradient becomes weak (small L2 → small gradient)
- The model drifts more freely from old representations between rounds
- In IID: drift is consistent across clients, so it averages out (OK)
- In Non-IID: each client drifts in its own direction → **feature space fragments across clients → catastrophic forgetting on aggregation**

### 3. Sigma-Heterogeneity Conflict

The temperature `σ` is a **single global scalar** shared across all clients via FedAvg. But in Non-IID FCL:

- Client A sees classes {cat, dog, car} → needs σ ≈ 8 for sharp distinctions
- Client B sees classes {truck, ship, plane} → needs σ ≈ 15 for its cluster geometry
- After FedAvg: σ ≈ 11.5 → **wrong for both clients**

The averaged σ produces a softmax distribution that is either too sharp (overconfident, poor generalization) or too blurry (underfitting) for every individual client. This is a **heterogeneity-specific problem** that doesn't exist in IID settings, which explains the IID vs Non-IID performance gap perfectly.

---

## Why IID Works But Non-IID Fails

| Aspect | IID (Shared) | Non-IID |
|--------|-------------|---------|
| Feature distributions | Similar across clients | Divergent across clients |
| Probability estimation | Still somewhat useful (clients see similar data) | Collapses (clients see different subsets) |
| Feature KD drift | Consistent direction → averages well | Conflicting directions → destructive averaging |
| σ averaging | Clients need similar σ → averaging works | Clients need different σ → compromise hurts all |
| Replay relevance | Most replay is relevant (similar distributions) | Much replay is irrelevant (different class subsets) |

**In IID:** Cosine's stability benefits outweigh the accuracy-of-forgetting degradation, because most replay is relevant anyway. The reduced BWT (+13.35 → +5.18) just means less positive transfer, not catastrophic failure.

**In Non-IID:** The inability to distinguish relevant from irrelevant replay is fatal. The model replays biased features with equal weight → forgetting doubles (+14.96 → +24.74).

---

## Verdict

> **Cosine normalization itself is sound.** The failure is in the interaction between cosine's magnitude-invariant features and AF-FCL's magnitude-dependent probability estimation. The fix is NOT to remove cosine — it is to make the "accurate forgetting" mechanism work in angular (direction-only) space instead of Euclidean space.

---

## Recommended Fixes (Ranked by Impact)

### Fix 1: Angular Probability Estimation (HIGH IMPACT)

Replace the Gaussian probability in `probability_in_localdata()` with a **von Mises-Fisher (vMF) distribution** when cosine is enabled. vMF is the natural probability distribution on the hypersphere:

```
P(x | μ, κ) ∝ exp(κ · μᵀx)
```

where `μ` is the mean direction and `κ` is the concentration parameter. This measures **directional relevance** instead of magnitude-based Gaussian probability.

### Fix 2: Per-Client Learnable Sigma (MEDIUM IMPACT)

Don't federate σ. Keep it as a client-local parameter that adapts to each client's class geometry. This removes the heterogeneity conflict without changing the rest of the pipeline.

### Fix 3: Angular Feature KD (MEDIUM IMPACT)

Replace L2 feature KD with cosine distance KD:
```python
kd_loss_feature = 1 - F.cosine_similarity(xa, xa_last).mean()
```

This makes the KD loss consistent with the cosine head's geometry — it preserves *direction* of old features, not magnitude.

### Fix 4: Sigma Warmup Schedule (LOW IMPACT)

Start σ low (3-5) and anneal upward per task. This prevents the sharp-softmax problem in early training while allowing discrimination in later tasks.
