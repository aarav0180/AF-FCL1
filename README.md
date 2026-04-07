# Accurate Forgetting for Heterogeneous Federated Continual Learning

The implementation of AF-FCL.

## Requirements

Install all required packages:

```bash
pip install -r requirements.txt
pip install torch torchvision glog scikit-learn
```

## Dataset Preparation

### EMNIST-Letters / EMNIST-Letters-malicious

The EMNIST dataset is downloaded automatically by torchvision on first run.
The data-split `.pkl` file must be placed at:

```
datasets/PreciseFCL/data_split/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl
```

No generator script is bundled for this variant; supply the file manually.

### EMNIST-Letters-shuffle

Two split files are available for this setting.

#### Variant 1 — Heterogeneous classes (`cs2`, default)

Each client is assigned a **different random set of 12 classes** across 6 tasks.
This is the fully-heterogeneous variant used in most experiments.

The split file is **auto-generated** the first time the experiment is run.
No manual preparation needed:

```
datasets/PreciseFCL/data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl
```

#### Variant 2 — Shared task pool (`shared`)

All 8 clients share the **same 12 classes and the same 6 task pairs**, but
each client receives those tasks in a **different random arrival order**.
This matches the "shuffle" setting described in the paper (~75.8% accuracy
for AF-FCL), where task content is consistent but temporal ordering is
heterogeneous across clients.

**Dataset composition:**

| Property | Value |
|---|---|
| Source | EMNIST-Letters (145,600 samples, 26 letters, upper+lower mapped to same label) |
| Clients (N) | 8 |
| Tasks per client (T) | 6 |
| Classes per task (C) | 2 |
| Total classes used | 12 (same for all clients) |
| Training samples per class per client | 500 |
| Task order | Same 6 tasks, independently shuffled per client |

Generate the split file once with:

```bash
python generate_emnist_shuffle_split_shared.py
```

This produces:

```
datasets/PreciseFCL/data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_shared_s2571.pkl
```

The two `.pkl` files are independent — generating the shared variant does not
affect the existing `cs2` file. Raw EMNIST image files are shared between both.

### CIFAR100

The dataset is downloaded automatically. Generate the split file once with:

```bash
python generate_cifar100_split.py
```

### MNIST-SVHN-FASHION

The code loads MNIST, SVHN and FashionMNIST with `download=False`, so the
three datasets must be downloaded manually into `datasets/PreciseFCL/` first.
The split file must also be placed at:

```
datasets/PreciseFCL/data_split/MNISTSVHNFASHION_split_cn10_tn6_cet3_s2571.pkl
```

---

## Experiments

> **Note:**
> - On **Linux / macOS** use `python` or `python3`.
> - On **Windows CMD** replace `\` continuation with `^`.
> - On **Windows PowerShell** replace `\` continuation with `` ` ``.

---

### EMNIST-Letters

#### Linux / macOS
```bash
python main.py \
  --dataset EMNIST-Letters \
  --data_split_file data_split/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl \
  --num_glob_iters 60 --local_epochs 100 \
  --lr 1e-4 --flow_lr 1e-4 \
  --k_loss_flow 0.5 --k_flow_lastflow 0.4 --flow_explore_theta 0
```

#### Windows CMD
```cmd
python main.py ^
  --dataset EMNIST-Letters ^
  --data_split_file data_split/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl ^
  --num_glob_iters 60 --local_epochs 100 ^
  --lr 1e-4 --flow_lr 1e-4 ^
  --k_loss_flow 0.5 --k_flow_lastflow 0.4 --flow_explore_theta 0
```

#### Windows PowerShell
```powershell
python main.py `
  --dataset EMNIST-Letters `
  --data_split_file data_split/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl `
  --num_glob_iters 60 --local_epochs 100 `
  --lr 1e-4 --flow_lr 1e-4 `
  --k_loss_flow 0.5 --k_flow_lastflow 0.4 --flow_explore_theta 0
```

---

### EMNIST-Letters-shuffle — Split A: Heterogeneous classes (`cs2`)

**Split type:** Each client is assigned its own **different random set of 12 classes** across 6 tasks.
Class overlap between clients is possible; temporal arrival order is also independent per client.
This is the more challenging and fully heterogeneous variant.

**Split file** (auto-generated on first run, or via `python generate_emnist_shuffle_split.py`):
```
datasets/PreciseFCL/data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl
```

**Best known configuration** (from experimental runs):

| Flag combination | Effect |
|---|---|
| `--gmm --gmm_k 4 --adaptive` | **Best accuracy** — GMM prior anchors latent geometry per task; adaptive KD scales regularisation by batch accuracy |
| `--klreg` | Prevents flow-loss explosion via gradient clipping; stabilises training but trades off a small amount of accuracy |
| `--klreg --klreg_beta 0.01` | Adds Jacobian KL term on top of clipping; further stabilises flow but does not improve accuracy vs clip-only |

> **Summary:** `--gmm --gmm_k 4 --adaptive` is the recommended combination for this split.
> `--klreg` is useful when flow loss explodes but expect a slight accuracy cost.

#### Linux / macOS
```bash
python main.py \
  --dataset EMNIST-Letters-shuffle \
  --data_split_file data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl \
  --num_glob_iters 60 --local_epochs 100 \
  --lr 1e-4 --flow_lr 1e-3 \
  --k_loss_flow 0.05 --k_flow_lastflow 0.02 --flow_explore_theta 0.5 \
  --gmm --gmm_k 4 --adaptive
```

#### Windows CMD
```cmd
python main.py ^
  --dataset EMNIST-Letters-shuffle ^
  --data_split_file data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl ^
  --num_glob_iters 60 --local_epochs 100 ^
  --lr 1e-4 --flow_lr 1e-3 ^
  --k_loss_flow 0.05 --k_flow_lastflow 0.02 --flow_explore_theta 0.5 ^
  --gmm --gmm_k 4 --adaptive
```

#### Windows PowerShell
```powershell
python main.py `
  --dataset EMNIST-Letters-shuffle `
  --data_split_file data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl `
  --num_glob_iters 60 --local_epochs 100 `
  --lr 1e-4 --flow_lr 1e-3 `
  --k_loss_flow 0.05 --k_flow_lastflow 0.02 --flow_explore_theta 0.5 `
  --gmm --gmm_k 4 --adaptive
```

---

### EMNIST-Letters-shuffle — Split B: Shared task pool (`shared`)

**Split type:** All 8 clients share the **same 12 classes and the same 6 task pairs**, but each
client receives those tasks in a **different random arrival order**.
This matches the "shuffle" setting in the paper (~75.8 % for AF-FCL baseline).

**Split file** (generate once with `python generate_emnist_shuffle_split_shared.py`):
```
datasets/PreciseFCL/data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_shared_s2571.pkl
```

**Best known configuration:** baseline PreciseFCL (no add-on flags).

> The add-ons (`--gmm`, `--klreg`, `--adaptive`) have not shown improvement over the
> baseline on this split in current experiments.  Use the plain baseline command below.

#### Linux / macOS
```bash
python main.py \
  --dataset EMNIST-Letters-shuffle \
  --data_split_file data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_shared_s2571.pkl \
  --num_glob_iters 60 --local_epochs 100 \
  --lr 1e-4 --flow_lr 1e-3 \
  --k_loss_flow 0.05 --k_flow_lastflow 0.02 --flow_explore_theta 0.5
```

#### Windows CMD
```cmd
python main.py ^
  --dataset EMNIST-Letters-shuffle ^
  --data_split_file data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_shared_s2571.pkl ^
  --num_glob_iters 60 --local_epochs 100 ^
  --lr 1e-4 --flow_lr 1e-3 ^
  --k_loss_flow 0.05 --k_flow_lastflow 0.02 --flow_explore_theta 0.5
```

#### Windows PowerShell
```powershell
python main.py `
  --dataset EMNIST-Letters-shuffle `
  --data_split_file data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_shared_s2571.pkl `
  --num_glob_iters 60 --local_epochs 100 `
  --lr 1e-4 --flow_lr 1e-3 `
  --k_loss_flow 0.05 --k_flow_lastflow 0.02 --flow_explore_theta 0.5
```

---

### EMNIST-Letters-malicious (M noisy clients)

Replace `$M` / `%M%` / `$M` with the desired number of malicious clients.

#### Linux / macOS
```bash
python main.py \
  --dataset EMNIST-Letters-malicious \
  --data_split_file data_split/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl \
  --num_glob_iters 60 --local_epochs 100 \
  --lr 1e-4 --flow_lr 1e-3 \
  --k_loss_flow 0.5 --k_flow_lastflow 0.1 --flow_explore_theta 0.5 \
  --malicious_client_num $M
```

---

## Experimental Extensions

The following implementation flags are available in `main.py` and are off by default:

- `--acta`: attention-based server aggregation with client task embeddings.
- `--gcar`: gradient conflict-aware replay for classifier updates.
- `--hmce`: multi-branch normalizing flow replay.
- `--cpr`: contrastive prototype replay on top of flow replay.
- `--maft`: learnable replay/KD gate based on batch statistics.

These flags are intended to be used as modular experiments on top of the existing AF-FCL pipeline.

#### Windows CMD
```cmd
python main.py ^
  --dataset EMNIST-Letters-malicious ^
  --data_split_file data_split/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl ^
  --num_glob_iters 60 --local_epochs 100 ^
  --lr 1e-4 --flow_lr 1e-3 ^
  --k_loss_flow 0.5 --k_flow_lastflow 0.1 --flow_explore_theta 0.5 ^
  --malicious_client_num %M%
```

#### Windows PowerShell
```powershell
python main.py `
  --dataset EMNIST-Letters-malicious `
  --data_split_file data_split/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl `
  --num_glob_iters 60 --local_epochs 100 `
  --lr 1e-4 --flow_lr 1e-3 `
  --k_loss_flow 0.5 --k_flow_lastflow 0.1 --flow_explore_theta 0.5 `
  --malicious_client_num $M
```

---

### MNIST-SVHN-FASHION

#### Linux / macOS
```bash
python main.py \
  --dataset MNIST-SVHN-FASHION \
  --data_split_file data_split/MNISTSVHNFASHION_split_cn10_tn6_cet3_s2571.pkl \
  --num_glob_iters 60 --local_epochs 100 \
  --lr 1e-4 --flow_lr 1e-3 \
  --k_loss_flow 0.1 --k_flow_lastflow 0 --flow_explore_theta 0 --fedprox_k 0.001
```

#### Windows CMD
```cmd
python main.py ^
  --dataset MNIST-SVHN-FASHION ^
  --data_split_file data_split/MNISTSVHNFASHION_split_cn10_tn6_cet3_s2571.pkl ^
  --num_glob_iters 60 --local_epochs 100 ^
  --lr 1e-4 --flow_lr 1e-3 ^
  --k_loss_flow 0.1 --k_flow_lastflow 0 --flow_explore_theta 0 --fedprox_k 0.001
```

#### Windows PowerShell
```powershell
python main.py `
  --dataset MNIST-SVHN-FASHION `
  --data_split_file data_split/MNISTSVHNFASHION_split_cn10_tn6_cet3_s2571.pkl `
  --num_glob_iters 60 --local_epochs 100 `
  --lr 1e-4 --flow_lr 1e-3 `
  --k_loss_flow 0.1 --k_flow_lastflow 0 --flow_explore_theta 0 --fedprox_k 0.001
```

---

### CIFAR100

#### Linux / macOS
```bash
python main.py \
  --dataset CIFAR100 \
  --data_split_file data_split/CIFAR100_split_cn10_tn4_cet20_s2571.pkl \
  --num_glob_iters 40 --local_epochs 400 \
  --lr 1e-3 --flow_lr 5e-3 \
  --k_loss_flow 0.5 --k_flow_lastflow 0.1 --flow_explore_theta 0.1 --fedprox_k 0.001
```

#### Windows CMD
```cmd
python main.py ^
  --dataset CIFAR100 ^
  --data_split_file data_split/CIFAR100_split_cn10_tn4_cet20_s2571.pkl ^
  --num_glob_iters 40 --local_epochs 400 ^
  --lr 1e-3 --flow_lr 5e-3 ^
  --k_loss_flow 0.5 --k_flow_lastflow 0.1 --flow_explore_theta 0.1 --fedprox_k 0.001
```

#### Windows PowerShell
```powershell
python main.py `
  --dataset CIFAR100 `
  --data_split_file data_split/CIFAR100_split_cn10_tn4_cet20_s2571.pkl `
  --num_glob_iters 40 --local_epochs 400 `
  --lr 1e-3 --flow_lr 5e-3 `
  --k_loss_flow 0.5 --k_flow_lastflow 0.1 --flow_explore_theta 0.1 --fedprox_k 0.001
```

---

## Running on GPU

Add `--device cuda` to any command above. To select a specific GPU, set
`CUDA_VISIBLE_DEVICES` before the command:

#### Linux / macOS
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --device cuda ...
```

#### Windows CMD
```cmd
set CUDA_VISIBLE_DEVICES=0 && python main.py --device cuda ...
```

#### Windows PowerShell
```powershell
$env:CUDA_VISIBLE_DEVICES=0; python main.py --device cuda ...
```

---

## CPU Smoke Tests

Run the minimal test suite to verify the installation is correct.
Each test uses `--num_glob_iters N_TASKS --local_epochs 1` so that the full
training loop is exercised quickly on CPU.
Tests whose required split file is missing are automatically skipped.

```bash
# Install pytest (once)
pip install pytest

# Run all CPU tests
pytest test_cpu.py -v
```

Without pytest:
```bash
python test_cpu.py
```

---

## GMM Prior for Normalizing Flow (`--gmm`)

Adding `--gmm` replaces the standard Normal base distribution inside the
normalizing flow with a **task-adaptive Gaussian Mixture Model (GMM) prior**.

**How it works:**

| Event | What happens |
|---|---|
| End of task T | Each client runs its full training set through the flow transform to collect latent codes `z = f(xa)`, then fits a K-component diagonal-covariance GMM on those codes via scikit-learn |
| The fitted GMM is frozen and **deepcopied** into `last_copy` alongside the flow weights | |
| Task T+1 flow training | Loss becomes `−log p_GMM(z) − log|det J_f|` instead of `−log N(0,I)(z) − log|det J_f|` |

The result: new-task features are pushed into the same **multi-modal cluster
geometry** as the previous task, preventing semantic drift.  The GMM buffers
(means, variances, weights) are **not federated** — each client keeps its own
local distribution.

> **Note:** `--gmm` is an add-on to `--algorithm PreciseFCL`. All other
> PreciseFCL flags (`--k_loss_flow`, `--flow_lr`, etc.) apply unchanged.

### Example — EMNIST-Letters-shuffle

#### Linux / macOS
```bash
python main.py \
  --dataset EMNIST-Letters-shuffle \
  --data_split_file data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl \
  --num_glob_iters 60 --local_epochs 100 \
  --lr 1e-4 --flow_lr 1e-3 \
  --k_loss_flow 0.05 --k_flow_lastflow 0.02 --flow_explore_theta 0.5 \
  --gmm --gmm_k 4
```

#### Windows CMD
```cmd
python main.py ^
  --dataset EMNIST-Letters-shuffle ^
  --data_split_file data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl ^
  --num_glob_iters 60 --local_epochs 100 ^
  --lr 1e-4 --flow_lr 1e-3 ^
  --k_loss_flow 0.05 --k_flow_lastflow 0.02 --flow_explore_theta 0.5 ^
  --gmm --gmm_k 4
```

#### Windows PowerShell
```powershell
python main.py `
  --dataset EMNIST-Letters-shuffle `
  --data_split_file data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl `
  --num_glob_iters 60 --local_epochs 100 `
  --lr 1e-4 --flow_lr 1e-3 `
  --k_loss_flow 0.05 --k_flow_lastflow 0.02 --flow_explore_theta 0.5 `
  --gmm --gmm_k 4
```

Add `--gmm_k N` to change the number of GMM components (default: 4).

---

## Stabilised Flow Training (`--klreg`)

The baseline `train_a_batch_flow` has no gradient clipping on the flow
optimizer (unlike the classifier, which clips at `max_norm=1.0`).  This was
confirmed to cause flow-loss explosion from task 0 round 6 onward in longer
runs.  `--klreg` fixes this with two changes and adds an optional third:

| Change | Always active with `--klreg` | Flag |
|---|---|---|
| NaN/Inf guard on `loss_last_flow` | ✅ | — |
| `clip_grad_norm_` on flow parameters | ✅ | `--klreg_clip` (default `1.0`) |
| Hutchinson-estimator Jacobian KL regularisation | optional | `--klreg_beta` (default `0.0`) |

The Jacobian KL term approximates $\|J\|_F^2 - \log|\det J|$ with a single
random probe $v \sim \mathcal{N}(0,I)$:

$$\mathcal{L}_\text{JacKL} = \mathbb{E}_v\left[\|J^\top v\|^2\right] - \log|\det J|$$

This regularises the coupling-transform Jacobian toward orthogonal-like
behaviour, preventing individual entries from blowing up while keeping the
determinant in range.

> Composable with all other flags.  `--klreg --klreg_beta 0` = clip only
> (safest starting point).  `--klreg --klreg_beta 0.01` = clip + Jacobian KL.

### Example

```bash
# Fix explosion, everything else unchanged:
python main.py --klreg

# GMM + stabilised flow:
python main.py --gmm --klreg --klreg_beta 0.01
```

---

## Adaptive KD Weighting (`--adaptive`)

The KD loss weight is normally a fixed hyperparameter.  `--adaptive` scales
the combined KD loss each mini-batch by:

$$\alpha = \sigma(\text{batch\_acc} - 0.5)$$

where $\sigma$ is the sigmoid function and `batch_acc` is the fraction of
correctly-classified samples in the current mini-batch (computed from the
already-available `softmax_output`, no extra forward pass).

| batch\_acc | α | Effect |
|---|---|---|
| 0.0 | ≈ 0.38 | Low accuracy → relax KD, let model learn |
| 0.5 | 0.50 | Balanced |
| 1.0 | ≈ 0.62 | High accuracy → tighten KD regularisation |

When `last_classifier is None` (task 0) α is forced to 1.0 — behaviour is
identical to the baseline since all KD terms are 0 anyway.

> Composable with `--gmm` and `--klreg`.

### Example

```bash
# Full stack — GMM prior + stabilised flow + adaptive KD:
python main.py --gmm --klreg --adaptive --klreg_beta 0.01
```

---

## Cosine Classifier Head (`--cosine`)

`--cosine` swaps the final linear classification layer for a **CosineLinear** head:

$$\hat{y} = \sigma \cdot \frac{W}{\|W\|} \cdot \frac{x_a}{\|x_a\|}$$

where $\sigma$ is a **learnable** per-head temperature scalar (initialised via
`--cosine_sigma`, default `10.0`) that is federated alongside the weights.

| Argument | Default | Description |
|---|---|---|
| `--cosine` | off | Replace `nn.Linear` head with L2-normalised `CosineLinear` |
| `--cosine_sigma` | `10.0` | Initial temperature σ |

**Why use it?**  Cosine classifiers decouple feature magnitude from the
classification decision, which can improve cross-task weight transfer and reduce
feature drift when replaying flow-generated samples whose magnitude distribution
may differ from real data.

**Implementation notes:**
- Only `__init__` is overridden — `forward`, `forward_to_xa`, and
  `forward_from_xa` are fully inherited from `S_ConvNet` / `Resnet_plus`.
- The mixin pattern (`CosineMixin`) calls `super().__init__(args)` first (builds
  the full model), then replaces `self.classifier.fc_classifier` in-place and
  rebuilds both optimisers (`classifier_optimizer`, `classifier_fb_optimizer`)
  to point at the new module.
- `sigma` lives in `named_parameters()` and is therefore aggregated by
  FedAvg with zero special casing.
- Because the existing loss uses `NLLLoss(log(softmax_output), y)` and the
  cosine head still returns a `(softmax_p, xa, logits)` triple, the swap is
  **fully transparent** to the training loop.

### Example

```bash
# Recommended starting point — cosine head + GMM prior + adaptive KD:
python main.py --cosine --gmm --gmm_k 4 --adaptive

# Full stack:
python main.py --cosine --gmm --gmm_k 4 --klreg --klreg_clip 1.0 --adaptive
```

---

## Flag Combinations

All four add-ons are independently composable (2⁴ = 16 classes, all wired):

### Without `--cosine`

| Flags | Model class used |
|---|---|
| *(none)* | `PreciseModel` |
| `--gmm` | `GMMPreciseModel` |
| `--klreg` | `KLRegPreciseModel` |
| `--adaptive` | `AdaptivePreciseModel` |
| `--gmm --klreg` | `KLRegGMMPreciseModel` |
| `--gmm --adaptive` | `AdaptiveGMMPreciseModel` |
| `--klreg --adaptive` | `AdaptiveKLRegPreciseModel` |
| `--gmm --klreg --adaptive` | `AdaptiveKLRegGMMPreciseModel` |

### With `--cosine` (CosineLinear head)

Prepend `--cosine [--cosine_sigma 10.0]` to any row above:

| Flags | Model class used |
|---|---|
| `--cosine` | `CosinePreciseModel` |
| `--cosine --gmm` | `CosineGMMPreciseModel` |
| `--cosine --klreg` | `CosineKLRegPreciseModel` |
| `--cosine --adaptive` | `CosineAdaptivePreciseModel` |
| `--cosine --gmm --klreg` | `CosineKLRegGMMPreciseModel` |
| `--cosine --gmm --adaptive` | `CosineAdaptiveGMMPreciseModel` |
| `--cosine --klreg --adaptive` | `CosineAdaptiveKLRegPreciseModel` |
| `--cosine --gmm --klreg --adaptive` | `CosineAdaptiveKLRegGMMPreciseModel` |

---

## Reference

The code structure is based on the code in [FedCIL](https://github.com/daiqing98/FedCIL).

The normalizing flow code refers to [nflows](https://github.com/bayesiains/nflows).

trying command :

python main.py --dataset EMNIST-Letters-shuffle --data_split_file data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl --num_glob_iters 60 --local_epochs 100 --lr 1e-4 --flow_lr 1e-3 --k_loss_flow 0.05 --k_flow_lastflow 0.02 --flow_explore_theta 0.5 --fedprox_k 0.01 --k_kd_feature 0.3 --k_kd_output 0.3 --gmm --gmm_k 12 --adaptive