"""Summary
"""
import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad
import chex
from distrax import Categorical
import rlax
import gymnax
import optax
import numpy as np
from functools import partial
import typing as t
from utils import (
    Trajectory,
    TrainState,
    ProcessedTrajectory,
    process_trajectory,
    KL_vmap,
    stack_trees
)
from models import ActorCritic
from ppo import ppo_loss


@chex.dataclass
class SL_PPO_Config:

    """Summary
    """

    T: int
    epochs: int
    # agents
    BC_fn: t.Callable
    n_agents: int  # number of agents
    N: jnp.ndarray  # Neighboring relations : N[i, j] = 1. -> i and j neighbors
    B_succ: jnp.ndarray  # success biases shape = (n_agents,)
    B_nov: jnp.ndarray  # novelty bias shape = (n_agents,)
    alpha: jnp.ndarray  # social learning rate = (n_agents,)

    # Env params
    env_name: str
    # network params
    network_config: t.Collection
    learning_rate: float
    seed: int
    # Loss function params
    c1: float
    c2: float
    gamma: float
    lambd: float
    epsilon: float

    batch_size: int


@chex.dataclass
class SL_PPO_Config:

    """Summary
    """

    T: int
    epochs: int
    # agents
    BC_fn: t.Callable
    n_agents: int  # number of agents
    N: jnp.ndarray  # Neighboring relations : N[i, j] = 1. -> i and j neighbors
    B_succ: jnp.ndarray  # success biases shape = (n_agents,)
    B_nov: jnp.ndarray  # novelty bias shape = (n_agents,)
    alpha: jnp.ndarray  # social learning rate = (n_agents,)

    # Env params
    env_name: str
    # network params
    network_config: t.Collection
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
        process_trajectory (TYPE): Description
        rng (TYPE): Description
        train: TYPE
        train_state: TYPE
        train_step: TYPE
    """

    # -------------------------------------------------------------------------
    def __init__(self, config: SL_PPO_Config):
        """Summary

        Args:
            config (Config): Description
        """
        self.config = config

        self.rng = jax.random.PRNGKey(config.seed)

        self.env_reset, self.env_step, state_dims, action_dims = self._init_env()
        self.apply_fn, self.apply_fn_vmap, params = self._init_network(
            state_dims, action_dims)
        opt_state, self.opt_update = self._init_opt(params)

        self.train_state = TrainState(
            params=params,
            opt_state=opt_state
        )

        self.grad_loss = self._build_grad_loss()
        self.exp_collector = jit(self._build_rollout(state_dims))
        self.process_trajectory = vmap(
            lambda traj: process_trajectory(traj,
                                            config.gamma,
                                            config.lambd),
            in_axes=1, out_axes=1
        )
        self.batch_maker = self._build_batch_maker()
        self.train_step = jit(self._build_train_step())
        self.train = self._build_train()

    # -------------------------------------------------------------------------

    def _build_rollout(self, state_dims: int)->t.Callable:
        """Summary

        Returns:
            t.Callable: Description
        """
        @vmap
        def a_and_lp(logits, key):
            dist = Categorical(logits)
            return dist.sample_and_log_prob(seed = key)

        def scannable_step(carry, key):
            s, e_s, params = carry
            key, sample_key, step_key = jax.random.split(key, 3)
            logits, v = self.apply_fn_vmap(params, s)
            v = v[:, 0]
            dist = Categorical(logits)
            a, lp = a_and_lp(logits, jax.random.split(sample_key, self.config.n_agents))
            s_, e_s, r, d, _ = vmap(self.env_step)(jax.random.split(step_key, self.config.n_agents), e_s, a)
            return (s_, e_s, params), Trajectory(s=s, a=a, lp=lp, v=v, r=r, d=d)
            
            

        def rollout(params: t.Collection, key)->Trajectory:
            """Summary

            Args:
                params (t.Collection): Description
                key (TYPE): Description

            Returns:
                Trajectory: Description
            """

            infos = {'episodes': 0}

            # sample_keys, reset_keys, step_keys = jax.random.split(keys, 3)

            key, reset_key = jax.random.split(key)

            s, env_state = vmap(self.env_reset)(jax.random.split(reset_key, self.config.n_agents))

            episodes = jnp.zeros((self.config.n_agents, ))

            step_keys = jax.random.split(key, self.config.T+1)

            final, traj = jax.lax.scan(scannable_step, (s, env_state, params),
                                       step_keys)
            episodes = jnp.sum(traj.d.astype(float), axis = 0)
            avg_return = jnp.sum(traj.r, axis = 0) / (episodes+1)
            
            return traj, {"episodes":episodes, "avg_ret":avg_return}

        return rollout

    # -------------------------------------------------------------------------

    def _build_train_step(self)->t.Callable:
        """Summary
        """
        def train_step(train_state: TrainState,
                       batches : t.Collection)->t.Tuple[TrainState, t.Collection]:
            """Summary

            Args:
                train_state (TrainState): Description
                batches (t.Iterable): Description

            Returns:
                TYPE: Description
            """
            loss_infos = {"loss": 0.}
            #batches = self.batch_maker(traj, key)
            params = train_state.params
            opt_state = train_state.opt_state
            for mb in batches:
                loss, grads = self.grad_loss(params, mb, 
                                            jnp.zeros((self.config.n_agents, 
                                                       self.config.n_agents)),
                                            jnp.zeros((self.config.n_agents,)))
                loss_infos['loss'] += loss
                updates, opt_state = self.opt_update(grads, opt_state)
                params = optax.apply_updates(params, updates)
            train_state.params = params
            train_state.opt_state = opt_state

            return train_state, loss_infos

        return train_step

    # -------------------------------------------------------------------------

    def _build_train(self)->t.Callable:
        """Summary
        """
        @jit
        def inner_step(train_state, key):
            
            step_key, roll_key, batch_key = jax.random.split(key, 3) 
            
            traj, roll_infos = self.exp_collector(train_state.params, roll_key)
            traj = self.process_trajectory(traj)
            batches = [self.batch_maker(traj, key) for key in jax.random.split(batch_key, self.config.epochs)]
            batches = jax.tree_map(lambda *trees : jnp.stack(trees), 
                                   *batches)
            train_state, infos = jax.lax.scan(
                lambda train_state, batches : self.train_step(train_state, batches),
                train_state, batches
            )

            train_state.training_steps+=1
            
            return train_state, {**roll_infos, **infos}

        def train(train_state: TrainState, steps: int, verb: bool = True):
            """
            ppo train function

            Args:
                train_state (TrainState): Description
                steps (int): Description

            Returns:
                TYPE: Description
            """
            keys = jax.random.split(self.rng, 20)
            logs = []
            for step in tqdm(range(steps)):
                train_state, log = inner_step(train_state, keys[step])
                logs.append(log)
            logs = jax.tree_map(lambda *trees : jnp.stack(list(trees)), *logs)

            return train_state, logs

        return train

    # -------------------------------------------------------------------------

    def _build_batch_maker(self)->t.Callable:
        """Summary
        """
        def make_batches(traj: ProcessedTrajectory, key)->t.Iterable:
            """Summary

            Args:
                traj (ProcessedTrajectory): Description

            Returns:
                t.Iterable: Description
            """
            batches = []
            keys = jax.random.split(key, self.config.n_agents)
            permut = vmap(jax.random.permutation, in_axes=(0,None))(
                keys, self.config.T
            )
            s, a, lp, ret, adv = traj.s, traj.a, traj.lp, traj.ret, traj.adv
            # (T, n_agents, s_dims)
            s = jnp.stack([s[permut[i], i] for i in range(self.config.n_agents)], axis=1)
            a = jnp.stack([a[permut[i], i] for i in range(self.config.n_agents)], axis=1)  # (n_agents, T)
            lp = jnp.stack([lp[permut[i], i] for i in range(self.config.n_agents)], axis=1)
            ret = jnp.stack([ret[permut[i], i] for i in range(self.config.n_agents)], axis=1)
            adv = jnp.stack([adv[permut[i], i] for i in range(self.config.n_agents)], axis=1)

            batch_size = self.config.batch_size
            n_batch = self.config.T // batch_size
            for i in range(n_batch):
                batches.append((
                    s[i*batch_size:(i+1)*batch_size, :, :],
                    a[i*batch_size:(i+1)*batch_size, :],
                    lp[i*batch_size:(i+1)*batch_size, :],
                    ret[i*batch_size:(i+1)*batch_size, :],
                    adv[i*batch_size:(i+1)*batch_size, :]
                ))
            return batches

        return make_batches

    # -------------------------------------------------------------------------

    def _build_grad_loss(self)->t.Callable:
        """Summary
        """
        def _ppo_loss(params, minibatch):
            """Summary

            Args:
                    params (TYPE): Description
                    minibatch (TYPE): Description

            Returns:
                TYPE: Description
            """
            return ppo_loss(params, self.apply_fn, minibatch,
                            self.config.c1, self.config.c2, self.config.epsilon)

        @partial(vmap, in_axes=(0, None, 0, 1, 0, 0, 0, None, None, 0))
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
                alpha (float): Description
                B_succ (float): Description
                B_nov (float): Description
                BC_dists (jnp.ndarray): Description
                R (jnp.ndarray): Description
                I (jnp.ndarray): Description

            Returns:
                float: Description
            """
            s, a, olp, ret, adv = batch

            # Compute the PPO loss
            loss_ppo = _ppo_loss(params, batch)

            # Compute KL divergences
            logits, _ = self.apply_fn(params, s)
            logits_others, _ = vmap(self.apply_fn, in_axes=(
                0, None))(other_params, s)  # To check

            kls = jnp.mean(KL_vmap(logits, logits_others), axis=-1)  # (n_agents,)

            # Compute phi
            # ..success bias
            phi_succ = B_succ * \
                (R / (jnp.sum(R * (N+I)) + 1e-8))  # (n_agents,)
            # ..novelty bias
            avg_BC_dists = (1 / (jnp.sum(N)+1)) * jnp.sum(BC_dists * (N+I)[None, :],
                                                          axis=-1)
            phi_nov = B_nov * (avg_BC_dists / (jnp.sum(avg_BC_dists)+1e-8))

            phi = phi_succ + phi_nov  # (n_agents,)

            sl_loss = jnp.sum(phi * kls * N)

            return loss_ppo + alpha * sl_loss

        def loss_fn(params: t.Collection, batch, BC_dists: jnp.ndarray, R: jnp.ndarray):
            return slppo_loss(
                params, params, self.config.N, batch, self.config.alpha, 
                self.config.B_succ, self.config.B_nov, BC_dists, R, 
                jnp.identity(self.config.n_agents)
            )

        return jit(loss_fn)

    # -------------------------------------------------------------------------

    def _init_network(self, state_dims: int, action_dims: int)->t.Tuple[t.Callable, t.Callable, t.Collection]:
        """Summary

        Args:
            state_dims (int): Description
            action_dims (int): Description

        Returns:
            t.Tuple[t.Callable, t.Callable, t.Collection]: Description
        """
        net = ActorCritic(state_dims, action_dims, **
                          self.config.network_config)

        apply_fn = net.apply

        init = vmap(net.init, in_axes=(0, None))
        keys = jax.random.split(self.rng, self.config.n_agents)
        params = init(keys, jnp.ones(state_dims))

        return (
            jit(apply_fn),
            jit(vmap(apply_fn)),
            params
        )

    # -------------------------------------------------------------------------

    def _init_env(self)->t.Tuple[t.Callable, t.Callable, int, int]:
        """Summary
        """
        env, env_params = gymnax.make(self.config.env_name)
        action_dims = env.num_actions
        state_dims = env.observation_space(env_params).shape[0]
        return env.reset, jit(env.step), state_dims, action_dims

    # -------------------------------------------------------------------------

    def _init_opt(self, params: t.Collection)->t.Tuple[t.Collection, t.Callable]:
        """Summary

        Args:
            params (t.Collection): Description
        """
        opt = optax.adam(self.config.learning_rate)
        opt_state = vmap(opt.init)(params)
        opt_update = vmap(opt.update)

        return opt_state, jit(opt_update)


if __name__ == '__main__':

    _config = Config(
        training_steps=200,
        T=16,
        epochs=10,
        BC_fn=lambda x: 0.,
        n_agents=2,
        N=jnp.array(
            [[0., 1.],
             [1., 0.]]
        ),  # Neighboring relations : N[i, j] = 1. -> i and j neighbors
        B_succ=jnp.zeros((2,)),  # success biases shape = (n_agents,)
        B_nov=jnp.zeros((2,)),
        alpha=jnp.zeros((2,)),
        env_name="CartPole-v1",
        network_config={'hidden_dims': (64, 64)},
        learning_rate=5e-4,
        seed=42,
        c1=.5,
        c2=0.,
        gamma=.99,
        lambd=.95,
        epsilon=.2,
        batch_size=4
    )

    sl_ppo = SL_PPO(_config)
    train_state = sl_ppo.train_state
    sl_ppo.train(train_state, 5)




