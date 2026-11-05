| Model | Seeds | Success | Collision | Timeout | Return | Travel time (s) | Closest dist (m) |
|---|---|---|---|---|---|---|---|
| ppo_mlp | 1 | 0.60 | 0.40 | 0.00 | 2.29 | 6.1 | 0.95 |
| ppo_cnn | 1 | 0.81 | 0.19 | 0.00 | 2.87 | 7.0 | 0.85 |

Checkpoints:
- `ppo_mlp`: `outputs/crowd_dyn_var_num_env/runs/ppo_mlp_s0-ppo_mlp-bs4096-ep4-lr1.0e-04-ent0.01-braxppo/ckpt_19660800.pkl`
- `ppo_cnn`: `outputs/crowd_dyn_var_num_env/runs/ppo_cnn0_window4-ppo_cnn-bs4096-ep4-lr1.0e-04-ent0.01-braxppo/ckpt_20000768.pkl`
