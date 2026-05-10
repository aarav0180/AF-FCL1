import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.datasets import CIFAR100
from collections import Counter

# ======================================================================
#  PAPER SETTINGS
# ======================================================================
N_CLIENTS = 10
N_TASKS = 4
CLASSES_PER_TASK = 20
SAMPLES_PER_CLASS = 400
SEED = 2571

DATA_DIR = "datasets/PreciseFCL/"
PLOTS_DIR = "./plots/"

# ======================================================================
#  HELPER FUNCTIONS
# ======================================================================
def load_data():
    """Load CIFAR100 and group indices by class."""
    print("Downloading/Loading CIFAR100 datasets...")
    train_dataset = CIFAR100(DATA_DIR, train=True, download=True)
    test_dataset = CIFAR100(DATA_DIR, train=False, download=True)
    
    train_labels = np.array(train_dataset.targets)
    test_labels = np.array(test_dataset.targets)
    
    train_class_inds = {c: np.where(train_labels == c)[0].tolist() for c in range(100)}
    test_class_inds = {c: np.where(test_labels == c)[0].tolist() for c in range(100)}
    return train_class_inds, test_class_inds

def generate_split(train_class_inds, test_class_inds, setting="iid"):
    """
    Generate FCL data split.
    
    IID: One global random permutation of classes used by ALL clients.
         (All clients learn the exact same tasks in the same order).
    NON-IID: Local Task Permutation (LTP) where each client gets a completely
             independent random permutation of classes.
    """
    # Reset seed to ensure deterministic but unique splits per setting
    np.random.seed(SEED if setting == "iid" else SEED + 1)
    
    train_inds = []
    test_inds = []
    client_y_list = []
    
    if setting == "iid":
        # IID: One global permutation for all clients
        global_classes = np.random.permutation(100)
    
    for client in range(N_CLIENTS):
        if setting == "noniid":
            # Non-IID: Local permutation for each client (LTP)
            classes = np.random.permutation(100)
        else:
            # IID: Share the same permutation
            classes = global_classes
            
        client_train = []
        client_test = []
        client_tasks_classes = []
        
        for t in range(N_TASKS):
            # Select 20 classes for this task
            task_classes = classes[t * CLASSES_PER_TASK:(t + 1) * CLASSES_PER_TASK]
            client_tasks_classes.append(task_classes.tolist())
            
            task_train_inds = []
            task_test_inds = []
            
            for c in task_classes:
                # IMPORTANT: Sample 400 train samples per class independently.
                # Overlap across clients is expected and ALLOWED per AF-FCL protocol.
                inds = np.random.choice(train_class_inds[c], SAMPLES_PER_CLASS, replace=False)
                task_train_inds.extend(inds.tolist())
                
                # Use all test samples for the class
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
    return split_data

def validate_split(split_data, setting):
    """Rigorous validation of the split against paper requirements."""
    print(f"\n{'='*50}\nVALIDATING {setting.upper()} SPLIT\n{'='*50}")
    train_inds = split_data["train_inds"]
    test_inds = split_data["test_inds"]
    client_y = split_data["client_y_list"]
    
    assert len(train_inds) == N_CLIENTS, "Wrong number of clients in train_inds"
    assert len(test_inds) == N_CLIENTS, "Wrong number of clients in test_inds"
    assert len(client_y) == N_CLIENTS, "Wrong number of clients in client_y"
    
    for c in range(N_CLIENTS):
        assert len(train_inds[c]) == N_TASKS, f"Client {c} has wrong number of tasks (train)"
        assert len(test_inds[c]) == N_TASKS, f"Client {c} has wrong number of tasks (test)"
        assert len(client_y[c]) == N_TASKS, f"Client {c} has wrong number of tasks (labels)"
        
        all_client_classes = []
        for t in range(N_TASKS):
            assert len(client_y[c][t]) == CLASSES_PER_TASK, f"Task {t} has wrong number of classes"
            assert len(train_inds[c][t]) == CLASSES_PER_TASK * SAMPLES_PER_CLASS, f"Task {t} has wrong number of train samples"
            
            # Check for no overlap between tasks within the SAME client
            for cls in client_y[c][t]:
                assert cls not in all_client_classes, f"Class {cls} overlaps across tasks in client {c}"
            all_client_classes.extend(client_y[c][t])
            
    print(f"✅ Structural Validation Passed for {setting.upper()}")

def analyze_and_plot(split_data, setting):
    """Compute statistics and generate visual distributions."""
    client_y = split_data["client_y_list"]
    
    # 1. Class frequency heatmap (Client vs Class)
    class_matrix = np.zeros((N_CLIENTS, 100))
    for c in range(N_CLIENTS):
        for t in range(N_TASKS):
            for cls in client_y[c][t]:
                class_matrix[c, cls] = t + 1 # Encode task ID (1 to 4)
                
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(class_matrix, cmap="viridis", cbar_kws={'label': 'Task ID (0=Not Present)'})
    plt.title(f"AF-FCL {setting.upper()} Setting - Client-Task Class Allocation")
    plt.xlabel("CIFAR-100 Class ID")
    plt.ylabel("Client ID")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{setting}_class_allocation_heatmap.png"), dpi=200)
    plt.close()
    
    # 2. Pairwise Client Distribution Similarity (Jaccard Index)
    client_sets = [set(np.concatenate(client_y[c])) for c in range(N_CLIENTS)]
    sim_matrix = np.zeros((N_CLIENTS, N_CLIENTS))
    for i in range(N_CLIENTS):
        for j in range(N_CLIENTS):
            intersection = len(client_sets[i].intersection(client_sets[j]))
            union = len(client_sets[i].union(client_sets[j]))
            sim_matrix[i, j] = intersection / union
            
    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_matrix, annot=True, cmap="coolwarm", vmin=0, vmax=1, fmt=".2f")
    plt.title(f"{setting.upper()} Setting - Pairwise Client Class Similarity (Jaccard)")
    plt.xlabel("Client ID")
    plt.ylabel("Client ID")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{setting}_client_similarity.png"), dpi=200)
    plt.close()
    
    # Print Stats
    print(f"\n--- {setting.upper()} Distribution Statistics ---")
    print(f"Average Pairwise Client Jaccard Similarity: {np.mean(sim_matrix):.4f}")
    
    # Entropy surrogate: check global class appearances
    all_classes_flat = np.concatenate([np.concatenate(client_y[c]) for c in range(N_CLIENTS)])
    counts = Counter(all_classes_flat)
    
    # In IID, exactly 80 classes appear 10 times, 20 appear 0 times.
    # In Non-IID, class appearances follow a binomial distribution.
    print(f"Total unique classes used globally (out of 100): {len(counts)}")
    print(f"Min appearances of a class across clients: {min(counts.values())}")
    print(f"Max appearances of a class across clients: {max(counts.values())}")
    print(f"Mean appearances of used classes: {np.mean(list(counts.values())):.2f}")

# ======================================================================
#  MAIN EXECUTION
# ======================================================================
def main():
    print("Initializing AF-FCL Split Generator...")
    train_class_inds, test_class_inds = load_data()
    
    for setting in ["iid", "noniid"]:
        print(f"\n" + "*"*60)
        print(f" PROCESSING {setting.upper()} SPLIT")
        print("*"*60)
        
        split_data = generate_split(train_class_inds, test_class_inds, setting)
        
        validate_split(split_data, setting)
        analyze_and_plot(split_data, setting)
        
        filename = f"CIFAR100_{setting.upper()}_cn10_tn4_cet20_s2571.pkl"
        save_path = os.path.join(DATA_DIR, "data_split", filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, "wb") as f:
            pickle.dump(split_data, f)
        print(f"✅ Saved perfectly formatted {setting.upper()} split to:\n   {save_path}")

if __name__ == "__main__":
    main()
