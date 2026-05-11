# Cosine-Normalized AF-FCL — Experimental Guide

This document is a companion to the main `README.md`.  
It covers only cosine-head experiments, angular-geometry fixes, and ablation setups.  
The original baseline pipeline is **unchanged** — all new mechanisms are opt-in via flags.

> **Note:** On Linux/macOS use `\` continuations. On Windows CMD use `^`. On Windows PowerShell use a backtick `` ` ``.

---

## 1. Overview

### Why cosine normalization was introduced

The standard `nn.Linear` classifier computes logits as a raw dot product:

```
y_k = x · W_k^T + b
```

During sequential task training the optimiser inflates weight magnitudes for new classes. Old-class weights cannot keep up, so new classes dominate predictions purely through magnitude, not learned similarity. This is a well-known problem in class-incremental and federated continual learning.

`CosineLinear` normalises both the feature vector and each prototype weight to the unit hypersphere before computing similarity:

```
y_k = σ · cos(x, W_k)  =  σ · (x/‖x‖) · (W_k/‖W_k‖)
```

Because magnitudes are divided out, class predictions are driven entirely by angular similarity — no class can dominate simply by being "bigger".

### What improved in IID settings

- Replay flow became numerically stable (`flow_prob_mean` ≈ 0.35–0.40 vs erratic)
- Classifier converged faster and more consistently
- Flow explosions eliminated in most EMNIST configurations
- Positive transfer (BWT) remained meaningful: +5.18% (cosine) vs +13.35% (baseline)

### What broke in Non-IID settings

Cosine normalization improved stability but **worsened forgetting** under Non-IID heterogeneity:

| Metric | Baseline Non-IID | Cosine Non-IID |
|---|---|---|
| Avg Forgetting | +14.96% | +24.74% |
| BWT | -14.72% | -29.69% |

The root cause is a **geometry mismatch**: the AF-FCL replay-selection pipeline was designed for Euclidean feature space, but the cosine head trains the backbone to discard magnitude — making all Euclidean distance and variance computations unreliable.

### The three failure mechanisms

**1. Probability estimation collapse (primary)**  
`probability_in_localdata()` uses per-dimension Gaussian variance to weight replayed samples. With cosine, feature magnitudes become uniform across classes → variance collapses → all replay samples get equal weight → the model can no longer distinguish relevant from biased replay.  
*Sign:* `flow_prob_mean` drops to 0.08–0.12 in CIFAR100 cosine runs.

**2. Feature KD distortion**  
The L2 feature KD loss `‖xa_old − xa‖²` becomes weak when cosine compresses magnitude variation. Under Non-IID heterogeneity, each client drifts in a different angular direction and FedAvg averages these conflicting drifts destructively.

**3. Sigma-heterogeneity conflict**  
The temperature scalar σ is federated via FedAvg. In Non-IID settings, different clients need different σ values for their own class geometry — averaging produces a σ that is wrong for every client.

---

## 2. Implemented Flags

All flags default to **OFF**. Existing experiments run identically without them.

### Core cosine head

| Flag | Type | Default | Description |
|---|---|---|---|
| `--cosine` | bool | off | Replace `nn.Linear` with `CosineLinear` |
| `--cosine_sigma` | float | 10.0 | Initial temperature σ (learnable, federated) |

### Angular-geometry fixes *(requires `--cosine`)*

| Flag | Type | Default | Description |
|---|---|---|---|
| `--vmf_prob` | bool | off | Replace Euclidean Gaussian replay-probability with von Mises–Fisher angular probability |
| `--vmf_kappa_min` | float | 0.5 | Minimum concentration κ for vMF estimator |
| `--vmf_kappa_max` | float | 50.0 | Maximum concentration κ for vMF estimator |
| `--angular_kd` | bool | off | Replace L2 feature KD with cosine-distance KD |

### Planned / future flags *(not yet implemented)*

| Flag | Description |
|---|---|
| `--adaptive_sigma` | Per-round σ adaptation based on replay quality |
| `--client_local_sigma` | Do not federate σ — keep it client-local |
| `--sigma_schedule` | Warmup schedule: `linear`, `cosine`, `step` |
| `--sigma_min` | Lower bound for sigma scheduling |
| `--sigma_max` | Upper bound for sigma scheduling |

---

### What each fix does

#### `--vmf_prob` — Angular replay selection

**Problem:** Baseline Gaussian probability measures Euclidean spread of features per class. Cosine head makes magnitudes uninformative, collapsing variance → probabilities become uniform → replay cannot filter biased samples.

**Fix:** Replaces the Gaussian with a von Mises–Fisher (vMF) distribution — the natural probability model on the unit hypersphere:

```
P(x | μ, κ) ∝ exp(κ · cos(x, μ))
```

where `μ` is the mean direction of the class's real features and `κ` is estimated from the mean resultant length. High probability = replay feature aligns well with the client's angular cluster for that class.

**When to enable:** Always when using `--cosine` in Non-IID settings. In IID settings the benefit is smaller but harmless.

**Expected effect:** `flow_prob_mean` stays in a meaningful range (0.2–0.5). Forgetting and BWT improve in Non-IID settings.

**Hyperparameter guidance:**
- `--vmf_kappa_min 0.5` — safe lower bound, prevents degenerate uniform distributions
- `--vmf_kappa_max 50.0` — prevents over-concentration; reduce to 20–30 for high-dimensional features (CIFAR100)

---

#### `--angular_kd` — Cosine-distance feature KD

**Problem:** Feature KD computes `‖xa_old − xa‖²`. When cosine compresses magnitude variation this gradient becomes weak, allowing the backbone to drift freely between rounds — and under Non-IID, clients drift in conflicting angular directions.

**Fix:** Replaces L2 with angular distance:

```
kd_loss_feature = 1 − cos(xa, xa_old)
```

This penalises *directional* drift from the previous/global model, which is the quantity the cosine head actually cares about.

**When to enable:** Recommended alongside `--vmf_prob` for full angular-geometry alignment. Can also be tested standalone to isolate its contribution.

**Expected effect:** Reduced client drift under heterogeneity, more consistent feature geometry across federated rounds.

---

## 3. Recommended Experimental Setups

### A — EMNIST IID (Shared split)

```bash
python main.py \
  --dataset EMNIST-Letters-shuffle \
  --data_split_file data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_shared_s2571.pkl \
  --algorithm PreciseFCL \
  --seed 2571 --device cuda \
  --num_glob_iters 60 --local_epochs 80 \
  --lr 1e-4 --flow_lr 1e-3 \
  --flow_epoch 8 --flow_explore_theta 0.2 \
  --k_loss_flow 0.1 --k_flow_lastflow 0.4 \
  --k_kd_last_cls 0.2 --k_kd_feature 0.5 --k_kd_output 0.1 \
  --cosine --cosine_sigma 10.0 \
  --adaptive \
  --target_dir_name output_emnist_iid_cosine
```

### B — EMNIST Non-IID (Heterogeneous split) with angular fixes

```bash
python main.py \
  --dataset EMNIST-Letters-shuffle \
  --data_split_file data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl \
  --algorithm PreciseFCL \
  --seed 2571 --device cuda \
  --num_glob_iters 60 --local_epochs 80 \
  --lr 1e-4 --flow_lr 1e-3 \
  --flow_epoch 8 --flow_explore_theta 0.2 \
  --k_loss_flow 0.1 --k_flow_lastflow 0.4 \
  --k_kd_last_cls 0.2 --k_kd_feature 0.5 --k_kd_output 0.1 \
  --cosine --cosine_sigma 10.0 \
  --vmf_prob --vmf_kappa_min 0.5 --vmf_kappa_max 50.0 \
  --angular_kd \
  --adaptive \
  --target_dir_name output_emnist_noniid_cosine_angular
```

### C — CIFAR100 IID

```bash
python main.py \
  --dataset CIFAR100 \
  --data_split_file data_split/CIFAR100_IID_cn10_tn4_cet20_s2571.pkl \
  --algorithm PreciseFCL \
  --seed 2571 --device cuda \
  --num_glob_iters 40 --local_epochs 250 \
  --lr 0.001 --flow_lr 1e-3 \
  --flow_epoch 6 --flow_explore_theta 0.1 \
  --k_loss_flow 0.5 --k_flow_lastflow 0.01 \
  --k_kd_last_cls 0.2 --k_kd_feature 0.5 --k_kd_output 0.1 \
  --fedprox_k 0.001 \
  --cosine --cosine_sigma 15.0 \
  --adaptive \
  --target_dir_name output_cifar100_iid_cosine
```

### D — CIFAR100 Non-IID with full angular fixes

```bash
python main.py \
  --dataset CIFAR100 \
  --data_split_file data_split/CIFAR100_NONIID_cn10_tn4_cet20_s2571.pkl \
  --algorithm PreciseFCL \
  --seed 2571 --device cuda \
  --num_glob_iters 40 --local_epochs 250 \
  --lr 0.001 --flow_lr 1.5e-3 \
  --flow_epoch 7 --flow_explore_theta 0.1 \
  --k_loss_flow 0.5 --k_flow_lastflow 0.01 \
  --k_kd_last_cls 0.2 --k_kd_feature 0.5 --k_kd_output 0.1 \
  --fedprox_k 0.001 \
  --cosine --cosine_sigma 15.0 \
  --vmf_prob --vmf_kappa_min 0.5 --vmf_kappa_max 30.0 \
  --angular_kd \
  --adaptive \
  --target_dir_name output_cifar100_noniid_cosine_angular
```

---

## 4. Best Parameter Regions (from experimental observations)

### EMNIST

| Parameter | Recommended range | Notes |
|---|---|---|
| `--cosine_sigma` | 10.0 | Stable; higher values not needed for 26 classes |
| `--local_epochs` | 70–90 | Excessive epochs → client drift |
| `--flow_epoch` | 7–9 | Fewer → underfitting; more → replay rigidity |
| `--flow_lr` | 1e-3 | Lower → replay stagnation; higher → instability |
| `--flow_explore_theta` | 0.1–0.3 | Controls explore/exploit balance in replay weighting |
| `--gmm` | Optional | Anchors flow geometry per task; adds compute |
| `--adaptive` | Recommended | Stabilizes KD without cost |
| `--vmf_kappa_max` | 50.0 | Safe for 26-class, lower-dimensional EMNIST |

### CIFAR100

| Parameter | Recommended range | Notes |
|---|---|---|
| `--cosine_sigma` | 12–18 | **Do not exceed 25** — causes softmax collapse |
| `--local_epochs` | 220–300 | 400+ causes severe client drift |
| `--flow_epoch` | 6–8 | Below 5 → replay underfitting |
| `--flow_lr` | 1e-3 to 1.8e-3 | Below 5e-4 → replay stagnation |
| `--flow_explore_theta` | 0.1 | Lower is better for 100 classes |
| `--vmf_kappa_max` | 25–35 | Reduce for high-dimensional features |
| `--fedprox_k` | 0.001 | Mild proximal regularization helps stability |

### Known failure configurations

| Configuration | Symptom | Cause |
|---|---|---|
| `--cosine_sigma 30-45` (CIFAR100) | Softmax near-zero for most classes; loss explosion | σ too large → overconfident logits → saturated gradients |
| `--flow_lr 2e-5` (CIFAR100) | `flow_prob_mean` → 0.06, replay flat | Flow never learns meaningful density → replay useless |
| `--local_epochs 400+` | BWT worsens significantly, high forgetting | Client drift dominates; aggregation destructive |
| `--cosine` alone in Non-IID | High forgetting despite stable training | Euclidean probability/KD incompatible with angular geometry |

---

## 5. Ablation Commands

All commands are reproducible from the terminal alone. No code edits required.

### Baseline (no cosine)

```bash
python main.py \
  --dataset EMNIST-Letters-shuffle \
  --data_split_file data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl \
  --algorithm PreciseFCL --seed 2571 --device cuda \
  --num_glob_iters 60 --local_epochs 80 \
  --lr 1e-4 --flow_lr 1e-3 --flow_epoch 8 \
  --k_loss_flow 0.1 --k_flow_lastflow 0.4 --flow_explore_theta 0.2 \
  --k_kd_last_cls 0.2 --k_kd_feature 0.5 --k_kd_output 0.1 \
  --target_dir_name ablation_baseline
```

### Cosine only (original, no angular fixes)

```bash
python main.py \
  --dataset EMNIST-Letters-shuffle \
  --data_split_file data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl \
  --algorithm PreciseFCL --seed 2571 --device cuda \
  --num_glob_iters 60 --local_epochs 80 \
  --lr 1e-4 --flow_lr 1e-3 --flow_epoch 8 \
  --k_loss_flow 0.1 --k_flow_lastflow 0.4 --flow_explore_theta 0.2 \
  --k_kd_last_cls 0.2 --k_kd_feature 0.5 --k_kd_output 0.1 \
  --cosine --cosine_sigma 10.0 \
  --target_dir_name ablation_cosine_only
```

### Cosine + vMF probability only

```bash
python main.py \
  --dataset EMNIST-Letters-shuffle \
  --data_split_file data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl \
  --algorithm PreciseFCL --seed 2571 --device cuda \
  --num_glob_iters 60 --local_epochs 80 \
  --lr 1e-4 --flow_lr 1e-3 --flow_epoch 8 \
  --k_loss_flow 0.1 --k_flow_lastflow 0.4 --flow_explore_theta 0.2 \
  --k_kd_last_cls 0.2 --k_kd_feature 0.5 --k_kd_output 0.1 \
  --cosine --cosine_sigma 10.0 \
  --vmf_prob --vmf_kappa_min 0.5 --vmf_kappa_max 50.0 \
  --target_dir_name ablation_cosine_vmf
```

### Cosine + angular KD only

```bash
python main.py \
  --dataset EMNIST-Letters-shuffle \
  --data_split_file data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl \
  --algorithm PreciseFCL --seed 2571 --device cuda \
  --num_glob_iters 60 --local_epochs 80 \
  --lr 1e-4 --flow_lr 1e-3 --flow_epoch 8 \
  --k_loss_flow 0.1 --k_flow_lastflow 0.4 --flow_explore_theta 0.2 \
  --k_kd_last_cls 0.2 --k_kd_feature 0.5 --k_kd_output 0.1 \
  --cosine --cosine_sigma 10.0 \
  --angular_kd \
  --target_dir_name ablation_cosine_angkd
```

### Cosine + all implemented fixes

```bash
python main.py \
  --dataset EMNIST-Letters-shuffle \
  --data_split_file data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl \
  --algorithm PreciseFCL --seed 2571 --device cuda \
  --num_glob_iters 60 --local_epochs 80 \
  --lr 1e-4 --flow_lr 1e-3 --flow_epoch 8 \
  --k_loss_flow 0.1 --k_flow_lastflow 0.4 --flow_explore_theta 0.2 \
  --k_kd_last_cls 0.2 --k_kd_feature 0.5 --k_kd_output 0.1 \
  --cosine --cosine_sigma 10.0 \
  --vmf_prob --vmf_kappa_min 0.5 --vmf_kappa_max 50.0 \
  --angular_kd \
  --adaptive \
  --target_dir_name ablation_cosine_all_fixes
```

---

## 6. Replay Diagnostics

### Key metrics to monitor in `run.log`

| Metric | Healthy range | Problem if... |
|---|---|---|
| `flow_loss` | Decreasing, stabilises | Still decreasing at final epoch → underfitting; constant from epoch 1 → collapsed |
| `flow_prob_mean` | 0.20–0.55 | Below 0.10 → probability collapse; above 0.80 → overfitting to current client |
| `c_loss_flow` | Decreasing with `c_loss` | Diverging from `c_loss` → replay and current task are conflicting |
| BWT | Positive (IID), slightly negative (Non-IID) | Strongly negative (< −20%) → replay is damaging old tasks |
| Avg Forgetting | Near 0% (ideal) | Increasing monotonically → replay failing entirely |

### Diagnosing specific failure modes

**Replay collapse** (`flow_prob_mean` → 0.06–0.12)
- All replay samples receive near-equal, near-zero weight
- Model ignores replay — effectively no continual learning
- Cause: Euclidean variance collapsed by cosine head
- Fix: `--vmf_prob`

**Replay stagnation** (`flow_loss` plateaus immediately)
- Flow never learns to model task-specific feature density
- Replayed features are near-random; replay loss is uninformative
- Cause: `--flow_lr` too low, or `--flow_epoch` too few
- Fix: increase `--flow_lr` to 1e-3, `--flow_epoch` to 7–8

**Replay rigidity** (stable flow loss, but poor BWT)
- Flow overfits tightly to seen data, cannot generalise to replay from other tasks
- Cause: `--local_epochs` too high → client drift; flow adapts to biased local distribution
- Fix: reduce `--local_epochs`, consider `--flow_explore_theta 0.3`

**Exploding flow** (`flow_loss` → NaN or >> 100)
- Cause: high `--flow_lr`, no gradient clipping
- Fix: add `--klreg --klreg_clip 1.0`

**Feature KD too weak** (features drift rapidly across rounds)
- `kd_loss_feature` near zero despite high forgetting
- Cause: cosine head compresses magnitudes; L2 KD loses sensitivity
- Fix: `--angular_kd`

---

## 7. Research Notes

### Key finding

> **Cosine normalization is not fundamentally incompatible with FCL.**  
> The failure in Non-IID settings arises entirely from a geometry mismatch: the replay-selection pipeline was designed for Euclidean feature space, but cosine training makes magnitudes uninformative.

### Summary of discovered interactions

| Mechanism | Effect in IID | Effect in Non-IID |
|---|---|---|
| Cosine head alone | ✅ More stable training, better convergence | ❌ Worse forgetting — Euclidean replay breaks |
| `--vmf_prob` | Neutral (Gaussian already works in IID) | ✅ Restores replay-selection discriminability |
| `--angular_kd` | Slightly stronger KD | ✅ Prevents conflicting directional drift under heterogeneity |
| `--adaptive` | ✅ Dynamic KD scales gracefully | ✅ Scales back when local distributions are unfamiliar |

### Why IID works without the fixes

In IID settings, all clients see the same class distribution. Most flow-generated replay samples are already relevant to every client regardless of their weight, so the probability collapse does not matter much. The Gaussian weights all samples equally — but equal weighting is approximately correct when all clients share the same data distribution.

Under Non-IID, equal weighting is wrong: a sample from a class the current client never saw should be down-weighted (accurate forgetting). Gaussian collapse removes this discrimination.

### Open problems

- **Per-client sigma:** The federated σ remains a compromise; a local σ per client would remove the heterogeneity conflict entirely.
- **Flow warm-up:** Starting with lower `--flow_explore_theta` and annealing may help the flow converge before being used for replay.
- **vMF κ estimation quality:** The current κ estimator works well in practice but is an approximation; exact MLE-based κ estimation may improve quality for very high-dimensional features.

---

## 8. Engineering Requirements & Reproducibility

### Principles

- **No code edits between runs** — every mechanism is controlled via CLI flags
- **All defaults preserve existing behavior** — experiments without new flags are byte-identical to pre-fix runs
- **All parameters saved in `args.json`** — output directory contains full hyperparameter record
- **Backward compatible** — old checkpoints, logs, and scripts continue to work

### Verifying a run is using the new mechanisms

Check `run.log` startup output for:

```
[CosineMixin] Angular replay-selection ENABLED (vMF, kappa_range=[0.5, 50.0])
[CosineMixin] Angular feature KD ENABLED (cosine distance)
```

These log lines are printed at model construction. Absence of both lines confirms the original cosine pipeline is running unchanged.

### args.json records all flags

Every output directory contains `args.json` which records all CLI values including the new flags. A run is fully reproducible by reading `args.json` and reconstructing the command.

### Recommended flag combinations for paper tables

| Row label | Flags |
|---|---|
| Baseline | *(no cosine flags)* |
| +Cosine | `--cosine --cosine_sigma 10.0` |
| +Cosine+vMF | `--cosine --cosine_sigma 10.0 --vmf_prob` |
| +Cosine+AngKD | `--cosine --cosine_sigma 10.0 --angular_kd` |
| +Cosine+All | `--cosine --cosine_sigma 10.0 --vmf_prob --angular_kd` |
| +Cosine+All+Adaptive | `--cosine --cosine_sigma 10.0 --vmf_prob --angular_kd --adaptive` |
