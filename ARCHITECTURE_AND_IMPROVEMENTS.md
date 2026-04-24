# AF-FCL1 Architecture and Implemented Improvements

## 1. Project Architecture

AF-FCL1 is a federated continual learning system built around one central idea:
client models learn task-by-task, the server aggregates them, and a normalizing
flow models the latent feature space of the classifier so replay and
regularization can be done without storing all past data.

### Main execution flow

1. `main.py` parses CLI arguments, selects the model/server variant, creates the
   run directory, sets the random seed, and starts training.
2. `FLAlgorithms/servers/serverPreciseFCL.py` coordinates the federated loop.
   It loads the dataset, creates one user object per client, selects users for
   each round, broadcasts the global model, collects local updates, aggregates
   them, and evaluates.
3. `FLAlgorithms/users/userPreciseFCL.py` handles client-side continual
   learning. Each user tracks current task data, past tasks, class history, and
   a frozen copy of the previous task model.
4. `FLAlgorithms/PreciseFCLNet/model.py` defines the core learning model:
   - a feature extractor and classifier
   - a flow over latent features
   - classifier and flow losses
   - knowledge distillation between current, last-task, and global models
5. `utils/dataset.py` loads the split pickle file and converts raw torchvision
   datasets into client/task sequences.

### Core model structure

Each client model contains two main parts:

- Classifier: learns class prediction for the current task
- Flow: learns the distribution of latent features `xa` conditioned on class
  labels

The classifier produces:

- softmax probabilities
- latent features `xa`
- logits

The flow learns on `xa` so the system can:

- generate replay features from past tasks
- estimate feature likelihoods
- stabilize continual learning with generative regularization

### Dataset and task structure

The code supports task-structured federated datasets such as:

- EMNIST-Letters
- EMNIST-Letters-shuffle
- EMNIST-Letters-malicious
- CIFAR100
- MNIST-SVHN-FASHION

Each client has multiple tasks, and each task contains a subset of classes.
The split files define which client sees which labels at each task.

---

## 2. Implemented Improvements

I added five new feature families and wired them into the existing flag-based
composition style used by GMM, KLReg, Adaptive, and Cosine.

### A. ACTA: Adaptive Client-Task Attention Aggregation

What it does:

- Replaces vanilla FedAvg with attention-weighted aggregation.
- Client updates are weighted by the similarity between a client task embedding
  and a global prototype summary.
- A prototype bank is updated over time so the server has a memory of task
  structure.

How it works:

- Each client computes a task embedding from the mean latent feature `xa` of
  its current task data.
- The server computes cosine similarity between each client embedding and a
  global prototype vector.
- Attention weights are computed with a softmax temperature.
- The server aggregates model parameters using those weights instead of sample
  count alone.

Where it lives:

- `FLAlgorithms/ACTAModule/acta_server.py`
- `main.py` selects `ACTAServerPreciseFCL` when `--acta` is passed.

### B. GCAR: Gradient Conflict-Aware Replay

What it does:

- Splits current-task loss and replay loss.
- Computes gradients separately for both terms.
- Detects gradient conflict when the dot product is negative.
- Projects the replay gradient before combining it with the current gradient.

How it works:

- Current batch classification loss is computed as usual.
- Replay loss comes from generated flow samples and optional prototype replay.
- If the current and replay gradients conflict, the replay gradient is projected
  away from the current gradient direction.
- The final gradient is `g_current + beta * g_replay_proj`.

Where it lives:

- `FLAlgorithms/PreciseFCLNet/model.py`
- Enabled with `--gcar`

### C. HMCE: Hierarchical Multi-Scale Correlation Estimation

What it does:

- Replaces the single flow with a multi-branch flow wrapper.
- Uses multiple feature permutations / flow branches to estimate replay across
  multiple scales.

How it works:

- Several flow instances are created for the same latent dimension.
- Each branch uses a different permutation pattern.
- A lightweight wrapper combines the branches behind the same `log_prob()` and
  `sample()` interface.
- This keeps the rest of the training code compatible.

Where it lives:

- `FLAlgorithms/PreciseFCLNet/model.py`
- Enabled with `--hmce`

### D. CPR: Contrastive Prototype Replay

What it does:

- Adds a prototype memory for classes seen by each client.
- Uses contrastive loss on current features and replay features.
- Runs as an additive replay regularizer on top of the existing flow replay.

How it works:

- At task transitions, each user computes class prototypes from its latent
  features.
- Prototypes are stored in a per-user prototype bank.
- During classifier training, the current feature vector and replay feature
  vector are compared against stored prototypes.
- A contrastive loss is added to the task objective.

Where it lives:

- `FLAlgorithms/users/userPreciseFCL.py`
- `FLAlgorithms/PreciseFCLNet/model.py`
- Enabled with `--cpr`

### E. MAFT: Meta-Learned Forgetting Threshold

What it does:

- Adds a learnable gate that scales replay / KD strength using batch statistics.
- Keeps the current synchronous task loop but learns a data-driven replay gate.

How it works:

- A small MLP takes batch accuracy, replay probability, and exploration state
  as input.
- The output is a scalar replay scale in `[0, 1]`.
- That scale modulates the replay loss before the optimizer step.

Where it lives:

- `FLAlgorithms/PreciseFCLNet/model.py`
- Enabled with `--maft`

---

## 3. Model Composition Strategy

The project keeps the original compositional pattern:

- `GMMPreciseModel` adds a task-adaptive GMM base distribution
- `KLRegMixin` stabilizes flow training with gradient clipping and optional
  Jacobian regularization
- `AdaptiveMixin` scales KD loss by batch accuracy
- `CosineMixin` replaces the classifier head with cosine similarity

The new features follow the same idea where possible:

- ACTA is server-side only
- GCAR modifies the classifier/replay update path
- HMCE modifies the flow construction
- CPR adds prototype-based replay state
- MAFT adds a learnable replay gate

---

## 4. How To Run

Use PowerShell line continuation with backticks, not `^`.

Example:

```powershell
python main.py `
  --dataset EMNIST-Letters-shuffle `
  --data_split_file data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_shared_s2571.pkl `
  --num_glob_iters 60 `
  --local_epochs 100 `
  --lr 1e-4 `
  --flow_lr 1e-3 `
  --k_loss_flow 0.05 `
  --k_flow_lastflow 0.02 `
  --flow_explore_theta 0.5 `
  --gcar `
  --device cuda
```

---

## 5. Notes

- The new flags are off by default.
- The existing baseline remains available when no new flags are used.
- Some combinations are more mature than others, so if you are experimenting,
  start with one new flag at a time.
- The most important follow-up is runtime validation for each flag combination
  on CPU or CUDA.
