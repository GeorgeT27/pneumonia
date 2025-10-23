#!/bin/bash
exp_name="mimic192"
run_cmd="python main.py \
    --exp_name=$exp_name \
    --data_dir=/workspace/causal-gen/pneumonia/ \
    --csv_dir=/workspace/causal-gen/pneumonia/ \
    --hps mimic192 \
    --parents_x finding age sex race \
    --context_dim=6 \
    --concat_pa \
    --lr=0.001 \
    --epochs=500 \
    --bs=32 \
    --wd=0.05 \
    --beta=9 \
    --x_like=diag_dgauss \
    --z_max_res=96 \
    --resume=/workspace/causal-gen/checkpoints/f_a_s_r/mimic192/checkpoint.pt \
    --eval_freq=4" 




# run_cmd="python main.py \
#     --exp_name=$exp_name \
#     --data_dir=/data2/ukbb \
#     --hps ukbb192 \
#     --parents_x mri_seq brain_volume ventricle_volume sex \
#     --context_dim=4 \
#     --concat_pa \
#     --lr=0.001 \
#     --bs=32 \
#     --wd=0.05 \
#     --beta=5 \
#     --x_like=diag_dgauss \
#     --z_max_res=96 \
#     --eval_freq=4"

if [ "$2" = "nohup" ]
then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  ${run_cmd}
fi