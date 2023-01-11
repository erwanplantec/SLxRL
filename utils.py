import jax
import chex
from distrax import Categorical
import typing as t

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