#!/bin/bash
model_name='pgm_60k'
exp_name=$model_name'-pgmg'
parents='f_a_s_r'
mkdir -p "../../checkpoints/$parents/$exp_name"


run_cmd="python train_pgm.py \
    --exp_name=$exp_name \
    --dataset mimic \
    --data_dir=/workspace/causal-gen/pneumonia/ \
    --hps mimic192 \
    --setup sup_pgm \
    --parents_x finding age sex race \
    --context_dim=6 \
    --concat_pa \
    --lr=0.001 \
    --bs=32 \ "

if [ "$2" = "nohup" ]
then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  ${run_cmd}
fi