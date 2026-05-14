"""
Generator: EMNIST-LTP (Local Task Permutation) split.
Paper config: N=8 clients, T=6 tasks, C=2 classes/task.
Each client independently draws random classes (LTP = heterogeneous classes).
Output: datasets/PreciseFCL/data_split/EMNIST_letters_LTP_cn8_tn6_cet2_s2571.pkl
"""
import os, pickle, numpy as np
from torchvision import datasets, transforms

N_CLIENTS = 8
N_TASKS = 6
CLASSES_PER_TASK = 2
N_EMNIST_CLASSES = 26
CLASSES_PER_CLIENT = N_TASKS * CLASSES_PER_TASK  # 12
SAMPLES_PER_CLASS = 500
SEED = 2571

DATA_DIR = "datasets/PreciseFCL/"
SPLIT_DIR = os.path.join(DATA_DIR, "data_split")
SAVE_PATH = os.path.join(SPLIT_DIR, "EMNIST_letters_LTP_cn8_tn6_cet2_s2571.pkl")

os.makedirs(SPLIT_DIR, exist_ok=True)

# ── Load EMNIST Letters ──
print("Loading EMNIST Letters ...")
emnist_train = datasets.EMNIST(DATA_DIR, split='letters', train=True,
                                download=True, transform=transforms.ToTensor(),
                                target_transform=lambda x: x - 1)
emnist_test  = datasets.EMNIST(DATA_DIR, split='letters', train=False,
                                download=True, transform=transforms.ToTensor(),
                                target_transform=lambda x: x - 1)

train_labels = np.array(emnist_train.targets) - 1
test_labels  = np.array(emnist_test.targets)  - 1

train_class_inds = {c: np.where(train_labels == c)[0].tolist() for c in range(N_EMNIST_CLASSES)}
test_class_inds  = {c: np.where(test_labels  == c)[0].tolist() for c in range(N_EMNIST_CLASSES)}

# ── Build LTP split (each client gets independently random classes) ──
train_inds, test_inds, client_y_list = [], [], []

print(f"Generating LTP split: {N_CLIENTS} clients × {N_TASKS} tasks × {CLASSES_PER_TASK} classes ...")
for client_id in range(N_CLIENTS):
    rng = np.random.RandomState(SEED + client_id * 137)
    client_classes = rng.choice(N_EMNIST_CLASSES, size=CLASSES_PER_CLIENT, replace=False)

    c_train, c_test, c_labels = [], [], []
    for t in range(N_TASKS):
        task_classes = client_classes[t * CLASSES_PER_TASK:(t + 1) * CLASSES_PER_TASK].tolist()
        c_labels.append(task_classes)
        t_train, t_test = [], []
        for c in task_classes:
            avail = train_class_inds[c]
            chosen = rng.choice(avail, min(SAMPLES_PER_CLASS, len(avail)), replace=False).tolist()
            t_train.extend(chosen)
            t_test.extend(test_class_inds[c])
        c_train.append(t_train)
        c_test.append(t_test)

    train_inds.append(c_train)
    test_inds.append(c_test)
    client_y_list.append(c_labels)
    print(f"  Client {client_id}: classes = {client_classes.tolist()}")

# ── Verify ──
print("\nVerifying ...")
for ci in range(N_CLIENTS):
    for ti in range(N_TASKS):
        actual = set(train_labels[np.array(train_inds[ci][ti])].tolist())
        expected = set(client_y_list[ci][ti])
        assert actual == expected, f"Client {ci}, Task {ti}: mismatch {actual} != {expected}"
print("  All checks passed ✓")

# ── Save ──
split_data = {"train_inds": train_inds, "test_inds": test_inds, "client_y_list": client_y_list}
with open(SAVE_PATH, "wb") as f:
    pickle.dump(split_data, f)
print(f"\nSaved → {SAVE_PATH}")
