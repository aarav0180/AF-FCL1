#!/bin/bash
# CIFAR-100 run with cosine head + stability fixes
# Key changes vs original:
#   --flow_lr 0.001        (was 0.005 in the failing run)
#   --k_flow_lastflow 0.01 (was 0.1 — reduces replay pressure on flow)
#   --cosine_sigma 5.0     (was 10.0 — gentler gradient scaling for 100 classes)

CUDA_VISIBLE_DEVICES=0 nohup python main.py \
  --dataset CIFAR100 \
  --algorithm PreciseFCL \
  --data_split_file data_split/CIFAR100_split_cn10_tn4_cet20_s2571.pkl \
  --seed 2571 \
  --device cuda \
  --num_glob_iters 40 \
  --local_epochs 550 \
  --batch_size 64 \
  --lr 0.001 \
  --flow_lr 0.001 \
  --flow_epoch 15 \
  --k_loss_flow 0.5 \
  --k_kd_last_cls 0.2 \
  --k_kd_feature 0.5 \
  --k_kd_output 0.1 \
  --k_flow_lastflow 0.01 \
  --flow_explore_theta 0.1 \
  --fedprox_k 0.001 \
  --cosine \
  --cosine_sigma 5.0 \
  --adaptive \
  --target_dir_name output_dir \
  &
