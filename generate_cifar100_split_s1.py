"""
Generator: CIFAR100 Setting 1 (main paper setting).
Paper config: N=10 clients, T=4 tasks, C=20 classes/task, 400 samples/class.
Each client gets independent random permutation of 80 classes (LTP).
Output: datasets/PreciseFCL/data_split/CIFAR100_cn10_tn4_cet20_s2571.pkl
"""
import os, pickle, numpy as np
from torchvision.datasets import CIFAR100

N_CLIENTS = 10
N_TASKS = 4
CLASSES_PER_TASK = 20
SAMPLES_PER_CLASS = 400
SEED = 2571

DATA_DIR = "datasets/PreciseFCL/"
SPLIT_DIR = os.path.join(DATA_DIR, "data_split")
SAVE_PATH = os.path.join(SPLIT_DIR, "CIFAR100_cn10_tn4_cet20_s2571.pkl")

os.makedirs(SPLIT_DIR, exist_ok=True)

# ── Load CIFAR100 ──
print("Loading CIFAR100 ...")
train_ds = CIFAR100(DATA_DIR, train=True, download=True)
test_ds  = CIFAR100(DATA_DIR, train=False, download=True)

train_labels = np.array(train_ds.targets)
test_labels  = np.array(test_ds.targets)

train_class_inds = {c: np.where(train_labels == c)[0].tolist() for c in range(100)}
test_class_inds  = {c: np.where(test_labels  == c)[0].tolist() for c in range(100)}

# ── Build split ──
np.random.seed(SEED)
train_inds, test_inds, client_y_list = [], [], []

print(f"Generating split: {N_CLIENTS} clients × {N_TASKS} tasks × {CLASSES_PER_TASK} classes ...")
for client_id in range(N_CLIENTS):
    classes = np.random.permutation(100)  # LTP: each client gets independent permutation
    c_train, c_test, c_labels = [], [], []
    for t in range(N_TASKS):
        task_classes = classes[t * CLASSES_PER_TASK:(t + 1) * CLASSES_PER_TASK]
        c_labels.append(task_classes.tolist())
        t_train, t_test = [], []
        for c in task_classes:
            inds = np.random.choice(train_class_inds[c], SAMPLES_PER_CLASS, replace=False)
            t_train.extend(inds.tolist())
            t_test.extend(test_class_inds[c])
        c_train.append(t_train)
        c_test.append(t_test)

    train_inds.append(c_train)
    test_inds.append(c_test)
    client_y_list.append(c_labels)
    print(f"  Client {client_id}: {N_TASKS} tasks, {CLASSES_PER_TASK} classes each")

# ── Verify ──
print("\nVerifying ...")
for ci in range(N_CLIENTS):
    for ti in range(N_TASKS):
        assert len(client_y_list[ci][ti]) == CLASSES_PER_TASK
        assert len(train_inds[ci][ti]) == CLASSES_PER_TASK * SAMPLES_PER_CLASS
        actual = set(train_labels[np.array(train_inds[ci][ti])].tolist())
        expected = set(client_y_list[ci][ti])
        assert actual == expected, f"Client {ci}, Task {ti}: mismatch"
    # No class overlap across tasks within same client
    all_cls = [c for t in client_y_list[ci] for c in t]
    assert len(all_cls) == len(set(all_cls)), f"Client {ci}: class overlap"
print("  All checks passed ✓")

# ── Save ──
split_data = {"train_inds": train_inds, "test_inds": test_inds, "client_y_list": client_y_list}
with open(SAVE_PATH, "wb") as f:
    pickle.dump(split_data, f)
print(f"\nSaved → {SAVE_PATH}")
