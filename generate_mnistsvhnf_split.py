"""
Generator: MNIST-SVHN-FashionMNIST mixed dataset split.
Paper config: N=10 clients, T=6 tasks, C=3 classes/task.
Total 20 classes: MNIST 0-9, SVHN 0-9 (mapped to 10-19 via FashionMNIST target_transform).
Actually: MNIST(0-9) + SVHN(0-9, labels kept as-is) + FashionMNIST(0-9, mapped to 10-19).
So unique_labels = 20 (MNIST & SVHN share 0-9, FashionMNIST uses 10-19).
Each client has 6 tasks × 3 classes = 18 classes drawn from 20.
Output: datasets/PreciseFCL/data_split/MNISTSVHNFASHION_split_cn10_tn6_cet3_s2571.pkl
"""
import os, pickle, numpy as np
from torchvision import datasets, transforms

N_CLIENTS = 10
N_TASKS = 6
CLASSES_PER_TASK = 3
UNIQUE_LABELS = 20  # MNIST(0-9) + FashionMNIST(10-19); SVHN shares 0-9 with MNIST
CLASSES_PER_CLIENT = N_TASKS * CLASSES_PER_TASK  # 18
SAMPLES_PER_CLASS = 500
SEED = 2571

DATA_DIR = "datasets/PreciseFCL/"
SPLIT_DIR = os.path.join(DATA_DIR, "data_split")
SAVE_PATH = os.path.join(SPLIT_DIR, "MNISTSVHNFASHION_split_cn10_tn6_cet3_s2571.pkl")

os.makedirs(SPLIT_DIR, exist_ok=True)

# ── Load datasets ──
print("Loading MNIST, SVHN, FashionMNIST ...")
repeat_transform = transforms.Lambda(lambda x: x.repeat(3, 1, 1))
mnist_mean, mnist_std = (0.1,), (0.2752,)

mnist_train = datasets.MNIST(DATA_DIR, train=True, download=True,
    transform=transforms.Compose([transforms.Pad(2, fill=0), transforms.ToTensor(),
                                  transforms.Normalize(mnist_mean, mnist_std), repeat_transform]))
mnist_test = datasets.MNIST(DATA_DIR, train=False, download=True,
    transform=transforms.Compose([transforms.Pad(2, fill=0), transforms.ToTensor(),
                                  transforms.Normalize(mnist_mean, mnist_std), repeat_transform]))

svhn_mean, svhn_std = [0.4377, 0.4438, 0.4728], [0.198, 0.201, 0.197]
svhn_train = datasets.SVHN(DATA_DIR, split='train', download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(svhn_mean, svhn_std)]))
svhn_test = datasets.SVHN(DATA_DIR, split='test', download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(svhn_mean, svhn_std)]))

fmnist_mean, fmnist_std = (0.2190,), (0.3318,)
fmnist_train = datasets.FashionMNIST(DATA_DIR, train=True, download=True,
    transform=transforms.Compose([transforms.Pad(2, fill=0), transforms.ToTensor(),
                                  transforms.Normalize(fmnist_mean, fmnist_std), repeat_transform]),
    target_transform=lambda x: x + 10)
fmnist_test = datasets.FashionMNIST(DATA_DIR, train=False, download=True,
    transform=transforms.Compose([transforms.Pad(2, fill=0), transforms.ToTensor(),
                                  transforms.Normalize(fmnist_mean, fmnist_std), repeat_transform]),
    target_transform=lambda x: x + 10)

# Concatenate all datasets
data_train = []
data_test = []
for ds in [mnist_train, svhn_train, fmnist_train]:
    data_train += [ds[i] for i in range(len(ds))]
for ds in [mnist_test, svhn_test, fmnist_test]:
    data_test += [ds[i] for i in range(len(ds))]

train_y = np.array([data_train[i][1] for i in range(len(data_train))])
test_y  = np.array([data_test[i][1]  for i in range(len(data_test))])

train_class_inds = {c: np.where(train_y == c)[0].tolist() for c in range(UNIQUE_LABELS)}
test_class_inds  = {c: np.where(test_y  == c)[0].tolist() for c in range(UNIQUE_LABELS)}

# ── Build split ──
np.random.seed(SEED)
train_inds, test_inds, client_y_list = [], [], []

print(f"Generating split: {N_CLIENTS} clients × {N_TASKS} tasks × {CLASSES_PER_TASK} classes ...")
for client_id in range(N_CLIENTS):
    rng = np.random.RandomState(SEED + client_id * 137)
    # Each client draws 18 unique classes from 20
    client_classes = rng.choice(UNIQUE_LABELS, size=CLASSES_PER_CLIENT, replace=False)

    c_train, c_test, c_labels = [], [], []
    for t in range(N_TASKS):
        task_classes = client_classes[t * CLASSES_PER_TASK:(t + 1) * CLASSES_PER_TASK].tolist()
        c_labels.append(task_classes)
        t_train, t_test = [], []
        for c in task_classes:
            avail = train_class_inds[c]
            n_sample = min(SAMPLES_PER_CLASS, len(avail))
            chosen = rng.choice(avail, n_sample, replace=False).tolist()
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
        actual = set(train_y[np.array(train_inds[ci][ti])].tolist())
        expected = set(client_y_list[ci][ti])
        assert actual == expected, f"Client {ci}, Task {ti}: mismatch {actual} != {expected}"
    all_cls = [c for t in client_y_list[ci] for c in t]
    assert len(all_cls) == len(set(all_cls)), f"Client {ci}: class overlap"
print("  All checks passed ✓")

# ── Save ──
split_data = {"train_inds": train_inds, "test_inds": test_inds, "client_y_list": client_y_list}
with open(SAVE_PATH, "wb") as f:
    pickle.dump(split_data, f)
print(f"\nSaved → {SAVE_PATH}")
