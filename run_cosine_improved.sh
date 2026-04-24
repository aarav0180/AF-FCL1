#!/bin/bash
# =============================================================================
# Run script for EMNIST-Letters-shuffle with improved Cosine classifier
# New features: Angular Margin + Feature Calibration + EWC regularization
# =============================================================================

# --- Configuration ---
DATASET="EMNIST-Letters-shuffle"
DATA_SPLIT="data_split/EMNIST_split_cn10_tn7_cet3_s2571.pkl"
DEVICE="cuda"
SEED=2571

# --- Recommended configurations to try ---
# Config 1: Cosine + Angular Margin (start here)
# Config 2: Cosine + Angular Margin + Feature Calibration
# Config 3: Cosine + Angular Margin + Feature Calibration + EWC
# Config 4: Full stack (Cosine + Margin + Calibration + EWC + GMM + KLReg + Adaptive)

echo "=== Config 1: Cosine + Angular Margin ==="
python main.py \
    --dataset "$DATASET" \
    --data_split_file "$DATA_SPLIT" \
    --algorithm PreciseFCL \
    --seed $SEED \
    --device $DEVICE \
    --cosine \
    --cosine_sigma 10.0 \
    --cosine_margin 0.15 \
    --num_glob_iters 60 \
    --local_epochs 80 \
    --batch_size 64 \
    --lr 1e-4 \
    --flow_lr 1e-4 \
    --k_loss_flow 0.1 \
    --k_kd_last_cls 0.2 \
    --k_kd_feature 0.5 \
    --k_kd_output 0.1 \
    --k_flow_lastflow 0.4 \
    --flow_explore_theta 0.2 \
    --flow_epoch 5 \
    --target_dir_name "output_cosine_margin"

# echo "=== Config 2: Cosine + Margin + Calibration ==="
# python main.py \
#     --dataset "$DATASET" \
#     --data_split_file "$DATA_SPLIT" \
#     --algorithm PreciseFCL \
#     --seed $SEED \
#     --device $DEVICE \
#     --cosine \
#     --cosine_sigma 10.0 \
#     --cosine_margin 0.15 \
#     --cosine_calibration \
#     --num_glob_iters 60 \
#     --local_epochs 80 \
#     --batch_size 64 \
#     --lr 1e-4 \
#     --flow_lr 1e-4 \
#     --k_loss_flow 0.1 \
#     --k_kd_last_cls 0.2 \
#     --k_kd_feature 0.5 \
#     --k_kd_output 0.1 \
#     --k_flow_lastflow 0.4 \
#     --flow_explore_theta 0.2 \
#     --flow_epoch 5 \
#     --target_dir_name "output_cosine_margin_calib"

# echo "=== Config 3: Cosine + Margin + Calibration + EWC ==="
# python main.py \
#     --dataset "$DATASET" \
#     --data_split_file "$DATA_SPLIT" \
#     --algorithm PreciseFCL \
#     --seed $SEED \
#     --device $DEVICE \
#     --cosine \
#     --cosine_sigma 10.0 \
#     --cosine_margin 0.15 \
#     --cosine_calibration \
#     --cosine_ewc 500.0 \
#     --num_glob_iters 60 \
#     --local_epochs 80 \
#     --batch_size 64 \
#     --lr 1e-4 \
#     --flow_lr 1e-4 \
#     --k_loss_flow 0.1 \
#     --k_kd_last_cls 0.2 \
#     --k_kd_feature 0.5 \
#     --k_kd_output 0.1 \
#     --k_flow_lastflow 0.4 \
#     --flow_explore_theta 0.2 \
#     --flow_epoch 5 \
#     --target_dir_name "output_cosine_full"

# echo "=== Config 4: Full stack ==="
# python main.py \
#     --dataset "$DATASET" \
#     --data_split_file "$DATA_SPLIT" \
#     --algorithm PreciseFCL \
#     --seed $SEED \
#     --device $DEVICE \
#     --cosine \
#     --cosine_sigma 10.0 \
#     --cosine_margin 0.15 \
#     --cosine_calibration \
#     --cosine_ewc 500.0 \
#     --gmm --gmm_k 4 \
#     --klreg --klreg_clip 1.0 \
#     --adaptive \
#     --num_glob_iters 60 \
#     --local_epochs 80 \
#     --batch_size 64 \
#     --lr 1e-4 \
#     --flow_lr 1e-4 \
#     --k_loss_flow 0.1 \
#     --k_kd_last_cls 0.2 \
#     --k_kd_feature 0.5 \
#     --k_kd_output 0.1 \
#     --k_flow_lastflow 0.4 \
#     --flow_explore_theta 0.2 \
#     --flow_epoch 5 \
#     --target_dir_name "output_cosine_fullstack"
