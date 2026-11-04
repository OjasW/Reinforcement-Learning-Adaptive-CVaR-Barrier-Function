| Model | Seeds | Success | Collision | Timeout | Return | Travel time (s) | Closest dist (m) |
|---|---|---|---|---|---|---|---|
| ppo_mlp | 3 | 0.60 ± 0.01 | 0.40 ± 0.01 | 0.00 ± 0.00 | 2.34 ± 0.05 | 6.2 ± 0.1 | 0.93 ± 0.02 |
| ppo_gnn | 3 | 0.92 ± 0.01 | 0.08 ± 0.01 | 0.00 ± 0.00 | 3.76 ± 0.05 | 6.7 ± 0.1 | 0.71 ± 0.02 |
| diff_cvar_mlp | 3 | 0.60 ± 0.03 | 0.36 ± 0.04 | 0.04 ± 0.02 | 1.25 ± 0.16 | 15.6 ± 0.5 | 0.67 ± 0.08 |
| diff_cvar_gnn | 3 | 0.93 ± 0.02 | 0.07 ± 0.02 | 0.00 ± 0.00 | 3.30 ± 0.07 | 12.0 ± 0.2 | 0.75 ± 0.01 |

Checkpoints:
- `ppo_mlp`: `outputs/crowd_dyn_var_num_env/runs/ppo_mlp_s0-ppo_mlp-bs4096-ep4-lr1.0e-04-ent0.01-braxppo/ckpt_19660800.pkl`
- `ppo_mlp`: `outputs/crowd_dyn_var_num_env/runs/ppo_mlp_s1-ppo_mlp-bs4096-ep4-lr1.0e-04-ent0.01-braxppo/ckpt_20000768.pkl`
- `ppo_mlp`: `outputs/crowd_dyn_var_num_env/runs/ppo_mlp_s2-ppo_mlp-bs4096-ep4-lr1.0e-04-ent0.01-braxppo/ckpt_20000768.pkl`
- `ppo_gnn`: `outputs/crowd_dyn_var_num_env/runs/ppo_gnn_s0-ppo_gnn-bs4096-ep4-lr1.0e-04-ent0.01-braxppo/ckpt_20000768.pkl`
- `ppo_gnn`: `outputs/crowd_dyn_var_num_env/runs/ppo_gnn_s1-ppo_gnn-bs4096-ep4-lr1.0e-04-ent0.01-braxppo/ckpt_19660800.pkl`
- `ppo_gnn`: `outputs/crowd_dyn_var_num_env/runs/ppo_gnn_s2-ppo_gnn-bs4096-ep4-lr1.0e-04-ent0.01-braxppo/ckpt_20000768.pkl`
- `diff_cvar_mlp`: `outputs/crowd_dyn_var_num_env/runs/diff_cvar_mlp_s0-diff_cvar_mlp-bs4096-ep4-lr1.0e-04-ent0.01-braxppo/ckpt_20000768.pkl`
- `diff_cvar_mlp`: `outputs/crowd_dyn_var_num_env/runs/diff_cvar_mlp_s1-diff_cvar_mlp-bs4096-ep4-lr1.0e-04-ent0.01-braxppo/ckpt_20000768.pkl`
- `diff_cvar_mlp`: `outputs/crowd_dyn_var_num_env/runs/diff_cvar_mlp_s2-diff_cvar_mlp-bs4096-ep4-lr1.0e-04-ent0.01-braxppo/ckpt_20000768.pkl`
- `diff_cvar_gnn`: `outputs/crowd_dyn_var_num_env/runs/diff_cvar_gnn_s0-diff_cvar_gnn-bs4096-ep4-lr1.0e-04-ent0.01-braxppo/ckpt_20000768.pkl`
- `diff_cvar_gnn`: `outputs/crowd_dyn_var_num_env/runs/diff_cvar_gnn_s1-diff_cvar_gnn-bs4096-ep4-lr1.0e-04-ent0.01-braxppo/ckpt_20000768.pkl`
- `diff_cvar_gnn`: `outputs/crowd_dyn_var_num_env/runs/diff_cvar_gnn_s2-diff_cvar_gnn-bs4096-ep4-lr1.0e-04-ent0.01-braxppo/ckpt_20000768.pkl`
