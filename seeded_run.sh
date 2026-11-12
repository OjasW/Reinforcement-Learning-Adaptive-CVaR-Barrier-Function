#usr/bin/env bash
for s in 1 2; do
  RUN_NAME=diff_cbf_cnn_s${s} bash scripts/run_ppo.sh model=diff_cbf_cnn seed=${s}
done
