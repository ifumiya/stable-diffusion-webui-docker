#!/bin/bash

dataset_config=/data/dataset.toml
output_name=$(date "+%Y%m%d-%H%M%S")_$1

# https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network_README-ja.md#%E5%AD%A6%E7%BF%92%E3%81%AE%E5%AE%9F%E8%A1%8C

accelerate launch \
  --num_cpu_threads_per_process 1 \
  train_network.py \
  --pretrained_model_name_or_path=/data/pretrain/anything-v3-fp16-pruned.safetensors \
  --dataset_config=${dataset_config} \
  --output_dir=/data/output/ \
  --output_name=${output_name} \
  --save_model_as=safetensors \
  --prior_loss_weight=1.0 \
  --max_train_steps=400 \
  --learning_rate=1e-4 \
  --optimizer_type="AdamW8bit" \
  --xformers \
  --mixed_precision="fp16" \
  --cache_latents \
  --gradient_checkpointing \
  --save_every_n_epochs=1 \
  --network_module=networks.lora
