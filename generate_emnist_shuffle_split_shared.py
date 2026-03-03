"""
Generator for EMNIST-Letters-shuffle split with SHARED task pool.

Produces:
  datasets/PreciseFCL/data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_shared_s2571.pkl

Shuffle semantics (as described in the paper):
  - All 8 clients share the SAME set of 6 tasks (same class pairs).
  - Each client sees those 6 tasks in a DIFFERENT random order.
  - This is the "shuffle" variant: consistent task content, heterogeneous arrival order.

Compared to the original generator (EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl):
  - Original: each client gets different classes (fully heterogeneous).
  - This file: all clients share the same 12 classes and the same 6 tasks, but shuffled order.

Parameters encoded in filename:
  cn8    - 8 clients
  tn6    - 6 tasks per client
  cet2   - 2 classes per task
  shared - all clients share the same task pool (the distinguishing trait)
  s2571  - random seed

Approximate expected accuracy (from paper): ~75.8% for AF-FCL on this setting.
"""

import os
import pickle
import numpy as np
from torchvision import datasets, transforms

# ── Configuration ────────────────────────────────────────────────────────────
N_CLIENTS         = 8
N_TASKS           = 6
CLASSES_PER_TASK  = 2
N_EMNIST_CLASSES  = 26
CLASSES_PER_CLIENT = N_TASKS * CLASSES_PER_TASK   # = 12
SAMPLES_PER_CLASS  = 500   # training samples per class per client
SEED               = 2571

DATA_DIR  = "datasets/PreciseFCL/"
SPLIT_DIR = os.path.join(DATA_DIR, "data_split")
SAVE_PATH = os.path.join(SPLIT_DIR,
            "EMNIST_letters_shuffle_split_cn8_tn6_cet2_shared_s2571.pkl")

rng = np.random.RandomState(SEED)
os.makedirs(SPLIT_DIR, exist_ok=True)

# ── Load EMNIST Letters ───────────────────────────────────────────────────────
print("Downloading / loading EMNIST Letters …")
emnist_train = datasets.EMNIST(DATA_DIR, split='letters', train=True,
                                download=True, transform=transforms.ToTensor(),
                                target_transform=lambda x: x - 1)  # labels → 0-25
emnist_test  = datasets.EMNIST(DATA_DIR, split='letters', train=False,
                                download=True, transform=transforms.ToTensor(),
                                target_transform=lambda x: x - 1)

train_labels = np.array(emnist_train.targets) - 1
test_labels  = np.array(emnist_test.targets)  - 1

# Collect indices per class
train_class_inds = {c: np.where(train_labels == c)[0].tolist()
                    for c in range(N_EMNIST_CLASSES)}
test_class_inds  = {c: np.where(test_labels  == c)[0].tolist()
                    for c in range(N_EMNIST_CLASSES)}

# ── Define the SHARED task pool (same for all clients) ───────────────────────
# Pick 12 classes at random (shared across all clients).
shared_classes = rng.choice(N_EMNIST_CLASSES, size=CLASSES_PER_CLIENT, replace=False)

# Split into 6 task pairs — fixed pairing, only the ORDER changes per client.
shared_tasks = [
    shared_classes[t * CLASSES_PER_TASK : (t + 1) * CLASSES_PER_TASK].tolist()
    for t in range(N_TASKS)
]

print(f"Shared task pool (same for all clients):")
for t_i, tc in enumerate(shared_tasks):
    print(f"  Task {t_i}: classes {tc}")

# ── Build per-client splits with shuffled task ORDER ─────────────────────────
train_inds    = []
test_inds     = []
client_y_list = []

print(f"\nGenerating shuffle order for {N_CLIENTS} clients …")

for client_id in range(N_CLIENTS):
    # Each client gets its own random permutation of the 6 task indices
    client_rng = np.random.RandomState(SEED + client_id * 137)
    task_order = client_rng.permutation(N_TASKS).tolist()

    client_train_inds  = []
    client_test_inds   = []
    client_task_labels = []

    print(f"  Client {client_id}: task order {task_order}  "
          f"→ classes per step {[shared_tasks[t] for t in task_order]}")

    for t_slot in range(N_TASKS):
        task_idx     = task_order[t_slot]
        task_classes = shared_tasks[task_idx]
        client_task_labels.append(task_classes)

        task_train = []
        task_test  = []
        for c in task_classes:
            avail    = train_class_inds[c]
            n_sample = min(SAMPLES_PER_CLASS, len(avail))
            chosen   = client_rng.choice(avail, n_sample, replace=False).tolist()
            task_train.extend(chosen)
            task_test.extend(test_class_inds[c])

        client_train_inds.append(task_train)
        client_test_inds.append(task_test)

    train_inds.append(client_train_inds)
    test_inds.append(client_test_inds)
    client_y_list.append(client_task_labels)

# ── Verify ───────────────────────────────────────────────────────────────────
print("\nVerifying split consistency …")
for c_i in range(N_CLIENTS):
    for t_i in range(N_TASKS):
        actual   = set(train_labels[np.array(train_inds[c_i][t_i])].tolist())
        expected = set(client_y_list[c_i][t_i])
        assert actual == expected, \
            f"Client {c_i}, Task {t_i}: label mismatch {actual} != {expected}"
print("  All checks passed ✓")

# ── Save ─────────────────────────────────────────────────────────────────────
split_data = {
    "train_inds"   : train_inds,
    "test_inds"    : test_inds,
    "client_y_list": client_y_list,
}

with open(SAVE_PATH, "wb") as f:
    pickle.dump(split_data, f)

print(f"\nSaved → {SAVE_PATH}")
print("\nTo use this split:")
print("  python main.py \\")
print("    --dataset EMNIST-Letters-shuffle \\")
print(f"    --data_split_file data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_shared_s2571.pkl \\")
print("    --num_glob_iters 60 --local_epochs 100 \\")
print("    --lr 1e-4 --flow_lr 1e-3 \\")
print("    --k_loss_flow 0.05 --k_flow_lastflow 0.02 --flow_explore_theta 0.5")
