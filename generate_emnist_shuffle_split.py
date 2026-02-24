"""
Generator for EMNIST-Letters-shuffle federated continual learning split.

Produces: datasets/PreciseFCL/data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl

Parameters (encoded in filename):
  cn8  - 8 clients
  tn6  - 6 tasks per client
  cet2 - 2 classes per task (12 classes per client)
  cs2  - 2 sample sets per class (index into EMNIST sampling)
  s2571 - random seed

Shuffle semantics: each client gets a DIFFERENT random class-to-task assignment,
so the temporal arrival order of classes is heterogeneous across clients.
"""

import os
import pickle
import numpy as np
from torchvision import datasets, transforms

# ── Configuration ────────────────────────────────────────────────────────────
N_CLIENTS         = 8
N_TASKS           = 6
CLASSES_PER_TASK  = 2      # each client sees 2 new classes per task
N_EMNIST_CLASSES  = 26     # EMNIST-Letters: a–z (labels 0–25 after transform)
CLASSES_PER_CLIENT = N_TASKS * CLASSES_PER_TASK   # = 12
SAMPLES_PER_CLASS  = 500   # training samples per class per client
SEED               = 2571

DATA_DIR   = "datasets/PreciseFCL/"
SPLIT_DIR  = os.path.join(DATA_DIR, "data_split")
SAVE_PATH  = os.path.join(SPLIT_DIR, "EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl")

np.random.seed(SEED)
os.makedirs(SPLIT_DIR, exist_ok=True)

# ── Load EMNIST Letters ───────────────────────────────────────────────────────
print("Downloading / loading EMNIST Letters …")
emnist_train = datasets.EMNIST(DATA_DIR, split='letters', train=True,
                                download=True, transform=transforms.ToTensor(),
                                target_transform=lambda x: x - 1)   # labels → 0-25
emnist_test  = datasets.EMNIST(DATA_DIR, split='letters', train=False,
                                download=True, transform=transforms.ToTensor(),
                                target_transform=lambda x: x - 1)

# .targets holds the raw labels (1-26); subtract 1 to get 0-25, matching target_transform
train_labels = np.array(emnist_train.targets) - 1
test_labels  = np.array(emnist_test.targets)  - 1

# Collect indices per class
train_class_inds = {c: np.where(train_labels == c)[0].tolist() for c in range(N_EMNIST_CLASSES)}
test_class_inds  = {c: np.where(test_labels  == c)[0].tolist() for c in range(N_EMNIST_CLASSES)}

# ── Build per-client class pools ──────────────────────────────────────────────
# To maximise coverage across clients, we cycle through all 26 classes evenly.
# Each client independently draws CLASSES_PER_CLIENT unique classes at random
# (shuffle = each client has a DIFFERENT random ordering).

train_inds    = []
test_inds     = []
client_y_list = []

print(f"Generating split for {N_CLIENTS} clients × {N_TASKS} tasks × {CLASSES_PER_TASK} classes …")

for client_id in range(N_CLIENTS):
    # Random set of 12 classes, unique per client (with different seed offsets)
    rng = np.random.RandomState(SEED + client_id * 137)
    # Pick CLASSES_PER_CLIENT random classes without replacement
    client_classes = rng.choice(N_EMNIST_CLASSES, size=CLASSES_PER_CLIENT, replace=False)

    client_train_inds  = []
    client_test_inds   = []
    client_task_labels = []

    for t in range(N_TASKS):
        task_classes = client_classes[t * CLASSES_PER_TASK: (t + 1) * CLASSES_PER_TASK].tolist()
        client_task_labels.append(task_classes)

        task_train = []
        task_test  = []
        for c in task_classes:
            # Sample training indices
            avail = train_class_inds[c]
            n_sample = min(SAMPLES_PER_CLASS, len(avail))
            chosen = rng.choice(avail, n_sample, replace=False).tolist()
            task_train.extend(chosen)

            # Use all test indices for this class
            task_test.extend(test_class_inds[c])

        client_train_inds.append(task_train)
        client_test_inds.append(task_test)

    train_inds.append(client_train_inds)
    test_inds.append(client_test_inds)
    client_y_list.append(client_task_labels)

    print(f"  Client {client_id}: classes = {client_classes.tolist()}")

# ── Verify correctness ───────────────────────────────────────────────────────
print("\nVerifying split consistency …")
for c_i in range(N_CLIENTS):
    for t_i in range(N_TASKS):
        actual_labels = set(train_labels[np.array(train_inds[c_i][t_i])].tolist())
        expected      = set(client_y_list[c_i][t_i])
        assert actual_labels == expected, \
            f"Client {c_i}, Task {t_i}: label mismatch {actual_labels} != {expected}"
print("  All checks passed ✓")

# ── Save ─────────────────────────────────────────────────────────────────────
split_data = {
    "train_inds"   : train_inds,
    "test_inds"    : test_inds,
    "client_y_list": client_y_list,
}

with open(SAVE_PATH, "wb") as f:
    pickle.dump(split_data, f)

print(f"\nSplit file saved → {SAVE_PATH}")
print(f"  Clients: {N_CLIENTS}  |  Tasks: {N_TASKS}  |  Classes/task: {CLASSES_PER_TASK}")
print(f"  Train samples/class: {SAMPLES_PER_CLASS}")
