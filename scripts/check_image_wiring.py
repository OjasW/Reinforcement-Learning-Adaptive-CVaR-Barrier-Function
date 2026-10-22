"""End-to-end check of the [IMG] wiring: rollout -> _update_batch -> PPO update -> eval.

[IMG] New file. Uses a stub image model (ppo_mlp + a tiny learned image term) so it runs before ppo_cnn exists.
Run from the repo root:
    python scripts/check_image_wiring.py
"""
import importlib.util, math, os, subprocess, sys, types

# repo root = parent of scripts/; an explicit path argument still works but is no longer required
repo = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo)
os.chdir(repo)

import jax, jax.numpy as jnp, numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("add", lambda *v: sum(v), replace=True)
OmegaConf.register_new_resolver("mul", lambda *v: math.prod(v), replace=True)

from env.jax_image_renderer import RenderConfig, history_to_cnn_input, init_image_history, render_policy_frame
from env.jax_social_nav import config_from_omegaconf, reset_env
from trainer.jax_ppo_rollout import init_obs_history, make_rollout_collector
from trainer.jax_ppo_core import compute_brax_gae, init_ppo_opt_state, make_ppo_update_fn
from trainer.jax_ppo_base_trainer import JaxPPOBaseTrainer
from trainer.jax_ppo_eval import evaluate_jax_env
import model.jax_ppo_mlp as mlp

failed = []
def ok(name, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        failed.append(name)

def load_original(relpath, modname):
    """Load the unmodified HEAD version of a repo file, for regression comparison."""
    src = subprocess.check_output(["git", "show", f"HEAD:{relpath}"], text=True)
    spec = importlib.util.spec_from_loader(modname, loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, f"HEAD:{relpath}", "exec"), mod.__dict__)
    return mod

with initialize_config_dir(config_dir=os.path.abspath("config"), version_base=None):
    cfg = compose(config_name="config", overrides=["model=ppo_mlp", "env.max_steps=10"])
env_cfg = config_from_omegaconf(cfg)
hp = mlp.hparams_from_config(cfg.model, cfg.model.critic, act_dim=2)
params = mlp.init_params_from_config(jax.random.PRNGKey(0), cfg.model, cfg.model.critic, 2,
                                     np.array([-1.5, -1.57], np.float32), np.array([1.5, 1.57], np.float32))
B, STEPS, T = 8, 16, 4
rc = RenderConfig()

# ---------------------------------------------------------------- 1. MLP path is unchanged
state0 = reset_env(env_cfg, jax.random.PRNGKey(1), B)
hist0 = init_obs_history(env_cfg, state0, hp.history_len)
es_vec = (state0, hist0, jnp.zeros((B,), jnp.float32))
def jit_collect(collector):  # close over static args, as JaxPPOBaseTrainer._make_jitted_train_step does
    return jax.jit(lambda params, es, key: collector(params, hp, env_cfg, es, key, STEPS))
new = jit_collect(make_rollout_collector(mlp, use_qp=False))
old_mod = load_original("trainer/jax_ppo_rollout.py", "old_rollout")
old = jit_collect(old_mod.make_rollout_collector(mlp, use_qp=False))
b_new, es_new, _ = new(params, es_vec, jax.random.PRNGKey(2))
b_old, es_old, _ = old(params, es_vec, jax.random.PRNGKey(2))
same = all(np.array_equal(np.asarray(b_new[k]), np.asarray(b_old[k]), equal_nan=True) for k in b_old)
ok("MLP rollout bit-identical to HEAD", same and set(b_new) == set(b_old) and len(es_new) == 3)

# ---------------------------------------------------------------- 2. image path, with a stub image model
seen = {}
def _img_term(params, image_obs, where):
    assert image_obs is not None, f"{where}: image_obs missing"
    assert image_obs.dtype == jnp.uint8 and image_obs.ndim == 5, (where, image_obs.dtype, image_obs.shape)
    seen[where] = tuple(image_obs.shape)
    x = history_to_cnn_input(image_obs)                       # (N, H, W, 3T) float in [0, 1]
    return params["critic"]["img_w"] * jnp.mean(x, axis=(1, 2, 3))   # tiny learned use of the image

stub = types.SimpleNamespace(
    get_action_and_value=lambda params, hparams, obs, image_obs=None, **kw: (
        lambda out: out[:3] + (out[3] + _img_term(params, image_obs, "get_action_and_value"),) + out[4:]
    )(mlp.get_action_and_value(params, hparams, obs, **kw)),
    critic_value=lambda params, hparams, obs, image_obs=None:
        mlp.critic_value(params, hparams, obs) + _img_term(params, image_obs, "critic_value"),
    policy_action_mean=lambda params, hparams, obs, image_obs=None:
        mlp.policy_action_mean(params, hparams, obs) + 0.0 * _img_term(params, image_obs, "policy_action_mean")[:, None],
    policy_action_to_env_action=mlp.policy_action_to_env_action,
)
params_img = jax.tree_util.tree_map(lambda x: x, params)
params_img["critic"] = dict(params["critic"], img_w=jnp.asarray(0.5, jnp.float32))

img0 = init_image_history(render_policy_frame(env_cfg, state0, rc), T)       # same as trainer.train()
es_img = (state0, hist0, img0, jnp.zeros((B,), jnp.float32))
coll = jit_collect(make_rollout_collector(stub, use_qp=False, render_cfg=rc))
batch, es_next, _ = coll(params_img, es_img, jax.random.PRNGKey(2))
ok("env_state is 4-tuple", len(es_next) == 4 and es_next[2].shape == (B, T, 84, 84, 3))
ok("batch['img'] shape/dtype", batch["img"].shape == (STEPS * B, T, 84, 84, 3) and batch["img"].dtype == jnp.uint8,
   f"{batch['img'].shape} {batch['img'].dtype}")
ok("vector obs still identical to MLP run", np.array_equal(np.asarray(batch["obs"]), np.asarray(b_old["obs"])))

# step t's newest image must equal step t+1's second-newest image (unless the env reset in between)
img = np.asarray(batch["img"]).reshape(STEPS, B, T, 84, 84, 3)
done = np.asarray(batch["done"]) > 0.5
chk = [np.array_equal(img[t, b, -1], img[t + 1, b, -2]) for t in range(STEPS - 1) for b in range(B) if not done[t, b]]
ok("image history shifts by one per step", all(chk), f"({len(chk)} transitions)")
rst = [all(np.array_equal(img[t + 1, b, 0], img[t + 1, b, k]) for k in range(T))
       for t in range(STEPS - 1) for b in range(B) if done[t, b]]
ok("image history cleared on reset", len(rst) > 0 and all(rst), f"({len(rst)} resets)")
ok("first stored image = initial render", np.array_equal(img[0], np.asarray(img0)))

# ---------------------------------------------------------------- 3. _update_batch + PPO update
adv, ret = compute_brax_gae(batch["rew"], batch["val"], batch["done"], batch["next_value"], gamma=0.99, lam=0.95)
ub = JaxPPOBaseTrainer._update_batch(batch, adv, ret)
ok("_update_batch keeps img (uint8)", "img" in ub and ub["img"].dtype == jnp.uint8)
upd = jax.jit(make_ppo_update_fn(hp, num_epochs=2, num_minibatches=4, get_action_and_value_fn=stub.get_action_and_value))
opt = init_ppo_opt_state(params_img)
new_params, _, _, m = upd(params_img, opt, ub, jax.random.PRNGKey(3), 3e-4, 0.2, 0.0, 1.0, 0.5, 0.2)
ok("minibatches got image slices", seen.get("get_action_and_value", (0,))[0] == STEPS * B // 4,
   f"last image_obs shape={seen.get('get_action_and_value')}")
dw = float(new_params["critic"]["img_w"] - params_img["critic"]["img_w"])
ok("gradient flows through images", dw != 0.0, f"img_w change={dw:.2e}")

# ---------------------------------------------------------------- 4. eval
ev_old_mod = load_original("trainer/jax_ppo_eval.py", "old_eval")
kw = dict(episodes=1, seeds=(100, 200), policy_action_mean_fn=mlp.policy_action_mean,
          policy_action_to_env_action_fn=mlp.policy_action_to_env_action)
m_new = evaluate_jax_env(params, hp, env_cfg, **kw)
m_old = ev_old_mod.evaluate_jax_env(params, hp, env_cfg, **kw)
ok("MLP eval identical to HEAD", m_new == m_old, str(m_new))
m_img = evaluate_jax_env(params_img, hp, env_cfg, episodes=1, seeds=(100, 200),
                         policy_action_mean_fn=stub.policy_action_mean,
                         policy_action_to_env_action_fn=stub.policy_action_to_env_action,
                         render_cfg=rc, image_history_len=T)
ok("image eval runs, policy got images", seen.get("policy_action_mean", (0,))[1:] == (T, 84, 84, 3)
   and m_img == m_old, "(stub ignores image in actions, so metrics must match)")

print("ALL PASS" if not failed else f"FAILED: {failed}")
sys.exit(1 if failed else 0)