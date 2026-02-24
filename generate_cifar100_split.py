import os
import pickle
import numpy as np
from torchvision.datasets import CIFAR100

# Paper settings
N_CLIENTS = 10
N_TASKS = 4
CLASSES_PER_TASK = 20
SAMPLES_PER_CLASS = 400
SEED = 2571

DATA_DIR = "datasets/PreciseFCL/"
SAVE_PATH = os.path.join(DATA_DIR, "data_split/CIFAR100_split_cn10_tn4_cet20_s2571.pkl")

np.random.seed(SEED)

# Load CIFAR100
train_dataset = CIFAR100(DATA_DIR, train=True, download=True)
test_dataset = CIFAR100(DATA_DIR, train=False, download=True)

train_labels = np.array(train_dataset.targets)
test_labels = np.array(test_dataset.targets)

# Collect indices per class
train_class_inds = {c: np.where(train_labels == c)[0].tolist() for c in range(100)}
test_class_inds = {c: np.where(test_labels == c)[0].tolist() for c in range(100)}

train_inds = []
test_inds = []
client_y_list = []

for client in range(N_CLIENTS):
    # Random class order for each client (LTP setting)
    classes = np.random.permutation(100)
    
    client_train = []
    client_test = []
    client_tasks_classes = []
    
    for t in range(N_TASKS):
        task_classes = classes[t * CLASSES_PER_TASK:(t + 1) * CLASSES_PER_TASK]
        client_tasks_classes.append(task_classes.tolist())
        
        task_train_inds = []
        task_test_inds = []
        
        for c in task_classes:
            # Sample 400 train samples per class
            inds = np.random.choice(train_class_inds[c], SAMPLES_PER_CLASS, replace=False)
            task_train_inds.extend(inds.tolist())
            
            # Use all test samples of that class
            task_test_inds.extend(test_class_inds[c])
        
        client_train.append(task_train_inds)
        client_test.append(task_test_inds)
    
    train_inds.append(client_train)
    test_inds.append(client_test)
    client_y_list.append(client_tasks_classes)

split_data = {
    "train_inds": train_inds,
    "test_inds": test_inds,
    "client_y_list": client_y_list
}

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

with open(SAVE_PATH, "wb") as f:
    pickle.dump(split_data, f)

print("Split file saved to:", SAVE_PATH)
