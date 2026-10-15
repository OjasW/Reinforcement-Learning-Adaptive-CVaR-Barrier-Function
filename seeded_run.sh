#usr/bin/env bash
for m in ppo_gnn diff_cvar_mlp diff_cvar_gnn; do
  for s in 0 1 2; do
    RUN_NAME=${m}_s${s} bash scripts/run_ppo.sh model=${m} seed=${s}
  done
done