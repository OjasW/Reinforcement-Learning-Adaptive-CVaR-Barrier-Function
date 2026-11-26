| Model | Seeds | Success | Collision | Timeout | Return | Travel time (s) | Closest dist (m) |
|---|---|---|---|---|---|---|---|
| ppo_cnn | 1 | 0.81 | 0.19 | 0.00 | 2.87 | 7.0 | 0.85 |
| diff_cbf_cnn | 3 | 0.83 ± 0.01 | 0.17 ± 0.01 | 0.00 ± 0.00 | 3.13 ± 0.04 | 8.3 ± 0.3 | 0.64 ± 0.01 |
| diff_cbf_cnn + lidar | 3 | 0.83 ± 0.02 | 0.17 ± 0.02 | 0.00 ± 0.00 | 3.08 ± 0.06 | 8.6 ± 0.1 | 0.65 ± 0.02 |

Checkpoints:
- `ppo_cnn`: `outputs/crowd_dyn_var_num_env/runs/ppo_cnn0_window4-ppo_cnn-bs4096-ep4-lr1.0e-04-ent0.01-braxppo/ckpt_20000768.pkl`
- `diff_cbf_cnn`: `outputs/crowd_dyn_var_num_env/runs/diff_cbf_cnn_s0-diff_cbf_cnn-bs4096-ep4-lr1.0e-04-ent0.01-braxppo/ckpt_20000768.pkl`
- `diff_cbf_cnn`: `outputs/crowd_dyn_var_num_env/runs/diff_cbf_cnn_s1-diff_cbf_cnn-bs4096-ep4-lr1.0e-04-ent0.01-braxppo/ckpt_20000768.pkl`
- `diff_cbf_cnn`: `outputs/crowd_dyn_var_num_env/runs/diff_cbf_cnn_s2-diff_cbf_cnn-bs4096-ep4-lr1.0e-04-ent0.01-braxppo/ckpt_20000768.pkl`
- `diff_cbf_cnn + lidar`: `outputs/crowd_dyn_var_num_env/runs/lidar_diff_cbf_cnn_s0-diff_cbf_cnn-bs4096-ep4-lr1.0e-04-ent0.01-braxppo/ckpt_19660800.pkl`
- `diff_cbf_cnn + lidar`: `outputs/crowd_dyn_var_num_env/runs/lidar_diff_cbf_cnn_s1-diff_cbf_cnn-bs4096-ep4-lr1.0e-04-ent0.01-braxppo/ckpt_20000768.pkl`
- `diff_cbf_cnn + lidar`: `outputs/crowd_dyn_var_num_env/runs/lidar_diff_cbf_cnn_s2-diff_cbf_cnn-bs4096-ep4-lr1.0e-04-ent0.01-braxppo/ckpt_20000768.pkl`
