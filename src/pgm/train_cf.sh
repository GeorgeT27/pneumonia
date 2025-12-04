#!/bin/bash
exp_name='cf_mimic192'
parents='f_a_s_r'
mkdir -p "../../checkpoints/$parents/$exp_name"

# Model checkpoint paths
predictor_path="/workspace/checkpoints/aux_60k-aux/checkpoint.pt"
pgm_path="/workspace/checkpoints/pgm_60k-pgmg/checkpoint.pt"
vae_path="/workspace/checkpoints/mimic192/checkpoint.pt"

# Data directory
data_dir="/workspace/causal-gen/pneumonia/"

run_cmd="python train_cf.py \
    --exp_name=$exp_name \
    --data_dir=$data_dir \
    --predictor_path=$predictor_path \
    --pgm_path=$pgm_path \
    --vae_path=$vae_path \
    --epochs=50 \
    --bs=16 \
    --lr=1e-4 \
    --lr_lagrange=1e-2 \
    --ema_rate=0.999 \
    --alpha=1 \
    --lmbda_init=0 \
    --damping=100 \
    --cf_particles=1 \
    --eval_freq=1 \
    --plot_freq=500 \
    --imgs_plot=10 \
    --seed=7"

if [ "$1" = "nohup" ]
then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  ${run_cmd}
fi

