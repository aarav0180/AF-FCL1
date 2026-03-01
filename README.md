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

The dataset is downloaded automatically and the split file is **auto-generated**
the first time the experiment is run. No manual preparation needed.

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

### EMNIST-Letters-shuffle

#### Linux / macOS
```bash
python main.py \
  --dataset EMNIST-Letters-shuffle \
  --data_split_file data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl \
  --num_glob_iters 60 --local_epochs 100 \
  --lr 1e-4 --flow_lr 1e-3 \
  --k_loss_flow 0.05 --k_flow_lastflow 0.02 --flow_explore_theta 0.5
```

#### Windows CMD
```cmd
python main.py ^
  --dataset EMNIST-Letters-shuffle ^
  --data_split_file data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl ^
  --num_glob_iters 60 --local_epochs 100 ^
  --lr 1e-4 --flow_lr 1e-3 ^
  --k_loss_flow 0.05 --k_flow_lastflow 0.02 --flow_explore_theta 0.5
```

#### Windows PowerShell
```powershell
python main.py `
  --dataset EMNIST-Letters-shuffle `
  --data_split_file data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl `
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

## Reference

The code structure is based on the code in [FedCIL](https://github.com/daiqing98/FedCIL).

The normalizing flow code refers to [nflows](https://github.com/bayesiains/nflows).