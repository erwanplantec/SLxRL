import jax.numpy as jnp
from jax import vmap, jit
import jax
import numpy as np
import chex
from distrax import Categorical
from functools import partial
import typing as t

@jit
def softmax(x:jnp.ndarray):
	return jnp.exp(x) / jnp.sum(jnp.exp(x))

@jit
def sum_norm(x:jnp.ndarray):
    return x / (x.sum() + 1e-8)

KL = lambda d1, d2 : d1.kl_divergence(d2)

def KL_vmap(logits1, logits2):
    return Categorical(logits1).kl_divergence(
            Categorical(logits2)
        )

@chex.dataclass
class TrainState:
    params : t.Collection
    opt_state : t.Collection
    training_steps : int = 0

@chex.dataclass
class Trajectory:
    s : jnp.ndarray
    a : jnp.ndarray
    lp : jnp.ndarray
    v : jnp.ndarray
    r : jnp.ndarray
    d : jnp.ndarray

@chex.dataclass
class ProcessedTrajectory:
    s : jnp.ndarray
    a : jnp.ndarray
    lp : jnp.ndarray
    ret : jnp.ndarray
    adv : jnp.ndarray

def stack_trees(trees: t.Iterable[t.Collection])->t.Collection:
    return jax.tree_map(
        lambda *trees : jnp.stack(trees) , *trees)

def pick_action(params, apply_fn, s):
	"""
	Out : action index, log_prob, value
	"""
	logits, v = apply_fn(params, s)
	dist = Categorical(logits=logits)
	return dist, v		

def process_trajectory(traj:Trajectory, gamma:float, 
                       lambd:float)->ProcessedTrajectory:
    adv = gae_advantages(
        traj.r[:-1, None],
        1 - traj.d[:-1, None],
        traj.v[:, None],
        gamma,
        lambd
    )[:, 0]
    ret = adv + traj.v[:-1]
    return ProcessedTrajectory(s=traj.s[:-1],
                                a=traj.a[:-1], 
                                lp=traj.lp[:-1],
                                ret=ret, 
                                adv=adv)

#https://github.com/google/flax/blob/main/examples/ppo/ppo_lib.py
def gae_advantages(
    rewards: np.ndarray,
    terminal_masks: np.ndarray,
    values: np.ndarray,
    discount: float,
    lambd: float):
	assert rewards.shape[0] + 1 == values.shape[0], ('One more value needed; Eq. '
	                                               '(12) in PPO paper requires '
	                                               'V(s_{t+1}) for delta_t')
	advantages = []
	gae = 0.
	for t in reversed(range(len(rewards))):
		# Masks used to set next state value to 0 for terminal states.
		value_diff = discount * values[t + 1] * terminal_masks[t] - values[t]
		delta = rewards[t] + value_diff
		# Masks[t] used to ensure that values before and after a terminal state
		# are independent of each other.
		gae = delta + discount * lambd * terminal_masks[t] * gae
		advantages.append(gae)
	advantages = advantages[::-1]
	return jnp.array(advantages)

if __name__ == '__main__':
    logits1 = jnp.array(np.random.rand(10, 4))
    logits2 = jnp.array(np.random.rand(2, 10, 4))

    d1 = Categorical(logits1)
    d2 = Categorical(logits2)

    kls = vmap(KL_vmap2, in_axes=(None, 0))(logits1, logits2)

    print(kls.shape)
