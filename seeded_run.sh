#usr/bin/env bash
for s in 0 1 2; do
  RUN_NAME=diff_cbf_gnn_s${s} bash scripts/run_ppo.sh model=diff_cbf_gnn seed=${s}
done
