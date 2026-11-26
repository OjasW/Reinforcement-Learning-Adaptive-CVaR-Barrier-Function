#usr/bin/env bash
for s in 0 1 2; do
  RUN_NAME=lidar_diff_cbf_cnn_s${s} bash scripts/run_ppo.sh model=diff_cbf_cnn seed=${s} \
  model.geometry=lidar model.lidar.range_noise_std=0.02 model.lidar.dropout_prob=0.01
done
