"""
Generator: EMNIST-shuffle split (paper-exact).
Paper config: N=8 clients, T=6 tasks, C=2 classes/task.
Shuffle: all clients share the SAME 6 tasks (same class pairs), but in DIFFERENT random orders.
Output: datasets/PreciseFCL/data_split/EMNIST_letters_shuffle_cn8_tn6_cet2_s2571.pkl
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
SAVE_PATH = os.path.join(SPLIT_DIR, "EMNIST_letters_shuffle_cn8_tn6_cet2_s2571.pkl")

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

# ── Define shared task pool ──
rng_global = np.random.RandomState(SEED)
shared_classes = rng_global.choice(N_EMNIST_CLASSES, size=CLASSES_PER_CLIENT, replace=False)
shared_tasks = [shared_classes[t * CLASSES_PER_TASK:(t + 1) * CLASSES_PER_TASK].tolist()
                for t in range(N_TASKS)]

print("Shared task pool:")
for ti, tc in enumerate(shared_tasks):
    print(f"  Task {ti}: classes {tc}")

# ── Build per-client splits (shuffled order) ──
train_inds, test_inds, client_y_list = [], [], []

print(f"\nGenerating shuffle for {N_CLIENTS} clients ...")
for client_id in range(N_CLIENTS):
    client_rng = np.random.RandomState(SEED + client_id * 137)
    task_order = client_rng.permutation(N_TASKS).tolist()

    c_train, c_test, c_labels = [], [], []
    for t_slot in range(N_TASKS):
        task_classes = shared_tasks[task_order[t_slot]]
        c_labels.append(task_classes)
        t_train, t_test = [], []
        for c in task_classes:
            avail = train_class_inds[c]
            chosen = client_rng.choice(avail, min(SAMPLES_PER_CLASS, len(avail)), replace=False).tolist()
            t_train.extend(chosen)
            t_test.extend(test_class_inds[c])
        c_train.append(t_train)
        c_test.append(t_test)

    train_inds.append(c_train)
    test_inds.append(c_test)
    client_y_list.append(c_labels)
    print(f"  Client {client_id}: order {task_order} → {[shared_tasks[t] for t in task_order]}")

# ── Verify ──
print("\nVerifying ...")
for ci in range(N_CLIENTS):
    for ti in range(N_TASKS):
        actual = set(train_labels[np.array(train_inds[ci][ti])].tolist())
        expected = set(client_y_list[ci][ti])
        assert actual == expected, f"Client {ci}, Task {ti}: mismatch"
print("  All checks passed ✓")

# ── Save ──
split_data = {"train_inds": train_inds, "test_inds": test_inds, "client_y_list": client_y_list}
with open(SAVE_PATH, "wb") as f:
    pickle.dump(split_data, f)
print(f"\nSaved → {SAVE_PATH}")
