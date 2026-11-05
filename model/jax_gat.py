import math

import jax
import jax.numpy as jnp

from model.jax_ppo_base import activation, init_linear

def _lin(layer, x):
    return x @ layer["weight"].T + layer["bias"]

def _mlp(layers, x, act):
    for layer in layers[:-1]:
        x = _lin(layer, x)
        x = activation(x, act)
    return _lin(layers[-1], x)

def _init_gat_params(key, node_dim, edge_dim, hidden_dim, embed_dim, num_layers, use_init_weights=True):
    sqrt2 = math.sqrt(2.0)
    keys = iter(jax.random.split(key, 8 * num_layers + 3))  # Check dim

    def lin(i, o, gain=sqrt2):
        return init_linear(next(keys), i, o, gain, use_init_weights)

    params = {"node_enc": lin(node_dim, hidden_dim), "layers": []}
    
    for _ in range(num_layers):
        params["layers"].append({
            "psi1": [lin(2 * hidden_dim + edge_dim, hidden_dim), lin(hidden_dim, hidden_dim)],
            "psi2": [lin(hidden_dim, hidden_dim), lin(hidden_dim, 1)],
            "psi3": [lin(hidden_dim, hidden_dim), lin(hidden_dim, hidden_dim)],
            "upd": [lin(2 * hidden_dim, hidden_dim), lin(hidden_dim, hidden_dim)],
        })

    params["psi4"] = [lin(hidden_dim, hidden_dim), lin(hidden_dim, embed_dim)]        
    return params

def _gat_layer(p, h, src, dst, edge_attr, act_name):
    n = h.shape[0]
    valid = edge_attr[:, -1]            # (E,) 0/1

    # z_ij = [v_i, v_j, e_ij] , i = receiver (dst), j = sender (src)
    z = jnp.concatenate([h[dst], h[src], edge_attr], axis=-1)  # (E, 2*hidden_dim + edge_dim)
    q = _mlp(p["psi1"], z, act_name)  # (E, hidden_dim)

    # masked softmax over the incoming edges of each receiver
    logits = _mlp(p["psi2"], q, act_name)[:, 0]  # (E,)
    logits = jnp.where(valid > 0.5, logits, -1e9)
    m = jax.lax.stop_gradient(jax.ops.segment_max(logits, dst, num_segments=n))  # (n,)
    w = jnp.exp(logits - m[dst]) * valid  # (E,)
    denom = jax.ops.segment_sum(w, dst, num_segments=n)  
    alpha = w / (denom[dst] + 1e-9) # all masked receivers will have alpha=0

    msg = _mlp(p["psi3"], q, act_name)  # (E, hidden_dim)
    agg = jax.ops.segment_sum(alpha[:, None] * msg, dst, num_segments=n)  # (n, hidden_dim)


    # update step so that nodes without incoming edges keep their own state
    return h + _mlp(p["upd"], jnp.concatenate([h, agg], axis=-1), act_name)

def gat_encode(params, node_features, edge_attr, src, dst, robot_idx, act_name="relu", return_nodes=False):
    def single(nf, ea):
        h = activation(_lin(params["node_enc"], nf), act_name)
        for layer in params["layers"]:
            h = _gat_layer(layer, h, src, dst, ea, act_name)
        robot = _mlp(params["psi4"], h[robot_idx], act_name)
        return (robot, h) if return_nodes else robot

    
    return jax.vmap(single)(node_features, edge_attr)