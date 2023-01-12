"""Summary
"""
import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad
import chex
from distrax import Categorical
import rlax
import numpy as np
import typing as t
import utils as ut
from ppo import ppo_loss



@chex.dataclass
class Config:

    """Summary
    """

    training_steps: int
    T: int
    epochs: int
    # agents
    BC_fn: t.callable
    n_agents: int  # number of agents
    N: jnp.ndarray  # Neighboring relations : N[i, j] = 1. -> i and j neighbors
    B_succ: jnp.ndarray  # success biases shape = (n_agents,)
    B_nov: jnp.ndarray  # novelty bias shape = (n_agents,)
    alpha: jnp.ndarray  # social learning rate = (n_agents,)

    # Env params
    env_name: str
    # network params
    network_config: t.Collection = {}
    learning_rate: float
    seed: int
    # Loss function params
    c1: float
    c2: float
    gamma: float
    lambd: float
    epsilon: float

    batch_size: int


class SL_PPO:

    """Summary
    
    Attributes:
        batch_maker: callable
        config: callable
        exp_collector: TYPE
        rng (TYPE): Description
        train: TYPE
        train_state: TYPE
        train_step: TYPE
        Description
        Description
        Description
        Description
        Description
        Description
    """

    # -------------------------------------------------------------------------
    def __init__(self, config: Config):
        """Summary
        
        Args:
            config (Config): Description
        """
        self.config = config

        self.rng = jax.random.PRNGKey(config.seed)

        self.env_reset, self.env_step, state_dims, action_dims = self._init_env()
        self.apply_fn, self.apply_fn_vmap, params = self._init_network(state_dims, action_dims)
        opt_state, self.opt_update = self._init_opt(params)

        self.train_state = ut.TrainState(
            params=params,
            opt_state=opt_state
        )

        self.exp_collector = self._build_rollout()
        self.process_trajectory = vmap(ut.process_trajectory)
        self.batch_maker = self._build_batch_maker()
        self.train_step = self._build_train_step()
        self.train = self._build_train()

    # -------------------------------------------------------------------------

    def _build_rollout(self)->t.Callable:
        """Summary
        """
        @partial(vmap, in_axes = (0, 0))
        def rollout(params:t.Collection, key)->Trajectory:
            """Summary
            
            Args:
                params (t.Collection): Description
                key (TYPE): Description
            
            Returns:
                Trajectory: Description
            """
            infos = {'episodes':0}
            
            traj_s = jnp.zeros((self.config.T+1, state_dims))
            traj_a = jnp.zeros((self.config.T+1,))
            traj_lp = jnp.zeros((self.config.T+1,))
            traj_v = jnp.zeros((self.config.T+1,))
            traj_r = jnp.zeros((self.config.T+1,))
            traj_d = jnp.zeros((self.config.T+1,))

            sample_key, reset_key, step_key = jax.random.split(self.rng, 3)
            
            s, env_state = self.env_reset(reset_key)

            episodes = 0
            ep_ret = 0
            
            for step in range(config.T+1):
                
                logits, v = self.apply_fn(params, s)
                v = v[0]
                dist = Categorical(logits)
                a, lp = dist.sample_and_log_prob(seed=sample_key)
                s_, env_state, r, d, _ = self.env_step(step_key, env_state, a)
                ep_ret += r

                episodes += d.astype(int)

                traj_s = ts.at[step].set(s)
                traj_a = ta.at[step].set(a)
                traj_lp = tlp.at[step].set(lp)
                traj_v = tv.at[step].set(v)
                traj_r = tr.at[step].set(r)
                traj_d = td.at[step].set(d.astype(float))

                s = s_
            
            infos['episodes'] = episodes

            return ut.Trajectory(s=traj_s, a=traj_a, lp=traj_lp, v=traj_v, r=traj_r, d=traj_d), infos

        return rollout

    # -------------------------------------------------------------------------

    def _build_train_step(self)->t.Callable:
        """Summary
        """
        pass

    # -------------------------------------------------------------------------

    def _build_train(self)->t.Callable:
        """Summary
        """
        pass

    # -------------------------------------------------------------------------

    def _build_batch_maker(self)->t.Callable:
        """Summary
        """
        pass

    # -------------------------------------------------------------------------

    def _build_grad_loss(self)->t.Callable:
    	"""Summary
    	"""
    	def _ppo_loss(params, minibatch):
    		return ppo_loss(params, apply_fn, minibatch, 
    			self.config.c1, self.config.c2, self.config.epsilon)

    	@partial(vmap, in_axes=(0, None, 0, 0, 0, 0, 0, None, None, 0))
    	@value_and_grad
		def slppo_loss(params: t.Collection, other_params: t.Collection,
		               N: jnp.array, batch: t.Tuple, alpha: float, 
		               B_succ: float, B_nov: float, BC_dists: jnp.ndarray, 
		               R: jnp.ndarray, I: jnp.ndarray)->float:
		    """
		    Args:
		        params (t.Collection): Description
		        other_params (t.Collection): Description
		        N (jnp.array): Description
		        batch (t.Tuple): Description
		        alpha (TYPE): Description
		        B_succ (TYPE): Description
		        B_nov (TYPE): Description
		        BC_dists (TYPE): Description
		        R (TYPE): Description
		        I (TYPE): Description
		    
		    Returns:
		        TYPE: Description
		    """
		    s, a, olp, v, ret, adv = batch

		    # Compute the PPO loss
		    loss_ppo = _ppo_loss(params, batch)

		    # Compute KL divergences
		    logits, _ = self.apply_fn(params, s)
		    dist = Categorical(logits)
		    logits_others, _ = vmap(self.apply_fn, in_axes=(0, None))(other_params, s) # To check
		    dist_others = Categorical(logits_others)

		    kls = ut.KL_vmap(dist, dist_others)  # (n_agents,)

		    # Compute phi
		    # ..success bias
		    phi_succ = B_succ * (R / (jnp.sum(R * (N+I)) + 1e-8))  # (n_agents,)
		    # ..novelty bias
		    avg_BC_dists = (1 / (jnp.sum(N)+1)) * jnp.sum(BC_dists * (N+I)[None, :],
		                                                  axis=-1)
		    phi_nov = B_nov * (avg_BC_dists / (jnp.sum(avg_BC_dists)+1e-8))

		    phi = phi_succ + phi_nov  # (n_agents,)

		    sl_loss = jnp.sum(phi * kls * N)

		    return loss_ppo + alpha * sl_loss

		return slppo_loss

    # -------------------------------------------------------------------------

    def _init_network(self, state_dims: int, action_dims: int)->t.Tuple[t.Callable, t.Callable, t.Collection]:
        """Summary
        
        Args:
            state_dims (int): Description
            action_dims (int): Description
        
        Returns:
            t.Tuple[t.Callable, t.Callable, t.Collection]: Description
        """
        net = ActorCritic(state_dims, action_dims, **self.config.network_config)
        
        apply_fn = net.apply
        apply_fn_mm = vmap(apply_fn)
        appply_fn_mn = vmap(apply_fn, in_axes=(0, None))
        
        init = vmap(net.init, in_axes = (0, None))
        keys = jax.random.split(self.rng, self.config.n_agents)
        params = init(keys, jnp.ones(state_dims))

        return (
        	apply_fn,
        	vmap(apply_fn),
        	params
        )

    # -------------------------------------------------------------------------

    def _init_env(self)->t.Tuple[t.Callable, t.Callable]:
        """Summary
        """
        pass

    # -------------------------------------------------------------------------

    def _init_opt(self, params: t.Collection)->t.Tuple[t.Collection, t.Callable]:
        """Summary
        
        Args:
            params (t.Collection): Description
        """
        pass
