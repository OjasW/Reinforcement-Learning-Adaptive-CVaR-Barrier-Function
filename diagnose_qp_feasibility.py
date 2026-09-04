"""
diagnose_qp_feasibility.py

Run this in a SEPARATE terminal while training continues. Read-only:
loads the latest checkpoint from disk, builds a fresh env instance, and
checks how often the CVaR-CBF-QP safety constraints are jointly
infeasible -- comparing "all obstacles visible" (what the GNN run is
actually using) against "only the closest obstacle" (the original
top_k=1 baseline's constraint count) on the SAME observations.

This does not import or touch anything from the running training
process -- it's a fresh Python process, CPU only, using cvxopt (which
reports infeasibility explicitly), not qpth.

Usage:
    python diagnose_qp_feasibility.py \
        --run-dir outputs/social_nav_var_num/runs/<your_gnn_run_name> \
        --episodes 20
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OmegaConf.register_new_resolver("math", lambda expr: eval(str(expr)), replace=True)

from crowd_sim.utils import build_env, resolve_env_name, absolute_obs_to_relative
from model.factory import build_model
from model.qp_solver import solve_qp_cvxopt


def latest_checkpoint(run_dir):
    ckpts = sorted(glob.glob(os.path.join(run_dir, "ckpt_*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    return max(ckpts, key=os.path.getmtime)


def count_active_constraints(mask_row):
    return int(mask_row.sum().item())


def check_feasibility(model, obs_row, top_k=None):
    """
    obs_row: (1, obs_dim) tensor, full-width (6 + max_humans*6) relative obs.
    top_k:   if set, zero out all but the top_k closest obstacle slots
             before checking (to compare against the original top-1 setup).
    Returns: (is_infeasible: bool, n_active_constraints: int)
    """
    obs = obs_row.clone()
    max_humans = model.max_humans_hint  # set by caller
    blocks = obs[:, 6:].view(1, max_humans, 6)

    if top_k is not None:
        rel = blocks[0, :, 0:2]
        dist = (rel ** 2).sum(-1)
        mask = blocks[0, :, 5] > 0.5
        dist = torch.where(mask, dist, torch.full_like(dist, float("inf")))
        keep = torch.argsort(dist)[:top_k]
        new_mask = torch.zeros_like(blocks[0, :, 5])
        new_mask[keep] = blocks[0, keep, 5]
        # zero-out obstacles not kept (mimics select_top_k_obs's effect on the mask)
        blocks = blocks.clone()
        blocks[0, :, 5] = new_mask
        obs = torch.cat([obs[:, :6], blocks.reshape(1, -1)], dim=1)

    with torch.no_grad():
        policy_feat = model.gnn_encoder(obs) if model.use_gnn else obs
        x = model.act(model.fc1(policy_feat))
        x21 = model.act(model.fc21(x))
        x22 = model.act(model.fc22(x))
        x23 = model.act(model.fc23(x))

        u_nom = model.fc31(x21)
        from model.diff_cvar import cvar_coeff_from_beta
        beta_raw = torch.sigmoid(model.fc32(x22)).squeeze(-1)
        beta = model.beta_min + (model.beta - model.beta_min) * beta_raw

        r_scale = 1.0 + 1.5 * torch.sigmoid(model.fc33(x23)).squeeze(-1)
        r_safe_learned = model.safe_dist * r_scale

        rel, human_vel, mask = model._extract_obstacle_blocks(obs)
        rel_x, rel_y = rel[:, :, 0], rel[:, :, 1]
        dist_sq = rel_x ** 2 + rel_y ** 2
        h = dist_sq - r_safe_learned.unsqueeze(1) ** 2

        cvar_coeff = cvar_coeff_from_beta(beta).unsqueeze(1).unsqueeze(2)
        means, variances = model._predict_gmm_multi(human_vel)
        rel_norm_sq = (rel ** 2).sum(dim=2, keepdim=True)
        sigma_f = torch.sqrt(4.0 * variances * rel_norm_sq + 1e-8)
        rel_dot_mu = 2.0 * (means * rel.unsqueeze(2)).sum(dim=3)
        rhs = (-model.alpha * h.unsqueeze(2)) + rel_dot_mu + (sigma_f * cvar_coeff)
        tau = 0.1
        rhs_wc = tau * torch.logsumexp(rhs / tau, dim=2)

        G = (-2.0 * rel * mask.unsqueeze(-1)).contiguous()
        h_qp = (-(rhs_wc * mask)).contiguous()

        Q = 2 * torch.eye(model.action_dim).unsqueeze(0)
        p = -2 * u_nom

        _, infeas = solve_qp_cvxopt(
            (Q[0]).to(torch.float64), (p[0]).to(torch.float64),
            G[0].to(torch.float64), h_qp[0].to(torch.float64),
            device="cpu", dtype=torch.float32, warm_start_x=None,
        )
        n_active = count_active_constraints(mask[0])
    return bool(infeas), n_active


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=100)
    args = ap.parse_args()

    ckpt_path = latest_checkpoint(args.run_dir)
    print(f"Using checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg = OmegaConf.load(os.path.join(args.run_dir, "config.yaml"))
    obs_dim = int(cfg.model.obs_dim)
    act_dim = int(cfg.model.act_dim)

    model = build_model(cfg, obs_dim=obs_dim, act_dim=act_dim)
    model.load_state_dict(state["model"], strict=True)
    model.actor.eval()
    actor = model.actor
    actor.max_humans_hint = (obs_dim - 6) // 6

    env = build_env(resolve_env_name(cfg), render_mode=None, config=cfg)

    results_full = {"infeasible": 0, "total": 0, "active_constraints": []}
    results_top1 = {"infeasible": 0, "total": 0}

    for ep in range(args.episodes):
        obs, info = env.reset(seed=1000 + ep)
        for t in range(args.max_steps):
            obs_t = torch.tensor(absolute_obs_to_relative(np.asarray(obs)),
                                  dtype=torch.float32).unsqueeze(0)

            infeas_full, n_active = check_feasibility(actor, obs_t, top_k=None)
            results_full["infeasible"] += int(infeas_full)
            results_full["total"] += 1
            results_full["active_constraints"].append(n_active)

            infeas_top1, _ = check_feasibility(actor, obs_t, top_k=1)
            results_top1["infeasible"] += int(infeas_top1)
            results_top1["total"] += 1

            # step the env using the actor's actual (safe) output, ignoring
            # our diagnostic re-solves above, so the trajectory matches
            # what the policy would really do
            with torch.no_grad():
                action = actor(obs_t).squeeze(0).numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    env.close()

    n = max(results_full["total"], 1)
    avg_active = np.mean(results_full["active_constraints"]) if results_full["active_constraints"] else 0.0
    print("\n=== QP feasibility check (all obstacle constraints active) ===")
    print(f"Infeasible fraction: {results_full['infeasible']}/{results_full['total']} "
          f"({100.0*results_full['infeasible']/n:.1f}%)")
    print(f"Average simultaneous active constraints per step: {avg_active:.2f}")

    print("\n=== Same observations, re-checked with only top-1 obstacle (original baseline) ===")
    print(f"Infeasible fraction: {results_top1['infeasible']}/{results_top1['total']} "
          f"({100.0*results_top1['infeasible']/n:.1f}%)")

    print("\nIf the top-20 infeasible rate is much higher than the top-1 rate, "
          "the collision-rate regression is very likely explained by the QP "
          "going infeasible more often during training (where there is no "
          "fallback), not by the GNN encoder producing bad features.")


if __name__ == "__main__":
    main()
