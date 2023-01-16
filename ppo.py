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
from utils import (
        Trajectory, 
        TrainState,
        ProcessedTrajectory, 
        process_trajectory
    )

@chex.dataclass
class Config:

    """Summary
    """
    
    training_steps:int
    T:int
    epochs:int
    n_actors:int

    loss_fn:t.Callable
    # Env params
    env_name:str
    # network params
    network_config : t.Collection
    learning_rate : float
    seed:int
    # Loss function params
    c1:float
    c2:float
    gamma:float
    lambd:float
    epsilon:float

    batch_size:int


def ppo_loss(params, apply_fn, minibatch, c1, c2, eps):
    """Summary
    
    Args:
        params (TYPE): Description
        apply_fn (TYPE): Description
        minibatch (TYPE): Description
        c1 (TYPE): Description
        c2 (TYPE): Description
        eps (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    s, a, olp, ret, adv = minibatch
    
    logits, vs = apply_fn(params, s)
    vs = vs[:, 0]
    dist = Categorical(logits)
    lp = dist.log_prob(a)
    # Compute the critic loss
    val_loss = jnp.mean(jnp.square(ret - vs))
    # Compute the policy loss
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ratios = jnp.exp(lp - olp)
    pi_loss = rlax.clipped_surrogate_pg_loss(
        ratios, adv, eps
    )
    #Compute entropy loss
    entropy_loss = jnp.mean(dist.entropy())

    return (pi_loss) + (c1 * val_loss) - (c2 * entropy_loss)

class PPO:

    """Summary
    
    Attributes:
        config (TYPE): Description
        exp_collector (TYPE): Description
        grad_loss (TYPE): Description
        make_batches (TYPE): Description
        process_trajectory (TYPE): Description
        rng (TYPE): Description
        train_fn (TYPE): Description
        train_state (TYPE): Description
        train_step (TYPE): Description
    """
    
    #-------------------------------------------------------------------------
    
    def __init__(self, config:Config):
        """Summary
        
        Args:
            config (Config): Description
        """
        self.config = config

        self.rng = jax.random.PRNGKey(config.seed)
        self.env_reset, self.env_step, state_dims, action_dims = self._init_env()
        self.apply_fn, params = self._init_network(state_dims, n_actions, 
            state_dims, action_dims)
        opt_state, self.opt_update = self._init_opt(params)

        self.grad_loss = self._build_grad_loss()

        self.exp_collector = self._build_rollout(config)
        self.make_batches = self._build_batch_maker(config)
        self.train_step = self._build_train_step()
        self.process_trajectory = vmap(lambda traj : process_trajectory(traj, 
                                                                config.gamma, 
                                                                config.lambd), 
                                       in_axes = 1,
                                       out_axes = 1)
        self.train_fn = self.build_train(config)

        self.train_state = TrainState(params=params, opt_state=opt_state)
    
    #-------------------------------------------------------------------------
    
    def _build_rollout(self)->t.Callable:
        """Summary
        
        Returns:
            t.Callable: Description
        """
        @partial(vmap, in_axes = (None, 0), out_axes = (1, 0))
        def rollout(params:t.Collection, key)->Trajectory:
            """Summary
            
            Args:
                params (t.Collection): Description
                key (TYPE): Description
            
            Returns:
                Trajectory: Description
            """
            infos = {'episodes':0}
            
            ts = jnp.zeros((self.config.T+1, state_dims))
            ta = jnp.zeros((self.config.T+1,))
            tlp = jnp.zeros((self.config.T+1,))
            tv = jnp.zeros((self.config.T+1,))
            tr = jnp.zeros((self.config.T+1,))
            td = jnp.zeros((self.config.T+1,))

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

                ts = ts.at[step].set(s)
                ta = ta.at[step].set(a)
                tlp = tlp.at[step].set(lp)
                tv = tv.at[step].set(v)
                tr = tr.at[step].set(r)
                td = td.at[step].set(d.astype(float))

                s = s_
            
            infos['episodes'] = episodes

            return Trajectory(s=ts, a=ta, lp=tlp, v=tv, r=tr, d=td), infos

        return rollout
    
    #------------------------------------------------------------------------- 
    
    def _build_batch_maker(self)->t.Callable:
        """Summary
        
        Returns:
            t.Callable: Description
        """
        def make_batches(traj:ProcessedTrajectory)->t.Iterable:
            """Summary
            
            Args:
                traj (ProcessedTrajectory): Description
            
            Returns:
                t.Iterable: Description
            """
            batches = []
            
            permut = jax.random.permutation(rng, self.config.T * self.config.n_actors)
            shape = (self.config.T * self.config.n_actors,)
            s = traj.s.reshape(shape+(-1,))[permut]
            a = traj.a.reshape(shape)[permut]
            lp = traj.lp.reshape(shape)[permut]
            ret = traj.ret.reshape(shape)[permut]
            adv = traj.adv.reshape(shape)[permut]
            
            batch_size = self.config.batch_size * self.config.n_actors
            n_batch = (self.config.T * self.config.n_actors) // batch_size
            for i in range(n_batch):
                batches.append((
                    s[i*batch_size:(i+1)*batch_size],
                    a[i*batch_size:(i+1)*batch_size],
                    lp[i*batch_size:(i+1)*batch_size],
                    ret[i*batch_size:(i+1)*batch_size],
                    adv[i*batch_size:(i+1)*batch_size]
                ))
            return batches

        return make_batches
    
    #-------------------------------------------------------------------------  
    
    def _build_train_step(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        def train_step(train_state:TrainState, batches:t.Iterable):
            """Summary
            
            Args:
                train_state (TrainState): Description
                batches (t.Iterable): Description
            
            Returns:
                TYPE: Description
            """
            infos = {"loss" : 0.}
            params = train_state.params
            opt_state = train_state.opt_state
            for mb in batches:
                loss, grads = self.grad_loss(params, mb)
                infos['loss'] += loss
                updates, opt_state = self.opt_update(grads, opt_state) 
                params = optax.apply_updates(params, updates)
            train_state.params = params
            train_state.opt_state = opt_state

            return train_state, infos

        return train_step
    
    #-------------------------------------------------------------------------
    
    def _build_train(self)->t.Callable:
        """Summary
        
        Returns:
            t.Callable: Description
        """
        def train(train_state:TrainState, steps:int):
            """
            ppo train function
            
            Args:
                train_state (TrainState): Description
                steps (int): Description
            
            Returns:
                TYPE: Description
            """
            for step in range(steps):
                keys = jax.random.split(self.rng, self.config.n_actors)
                traj, roll_infos = self.exp_collector(train_state.params, keys)
                traj = self.process_trajectory(traj)
                for epoch in range(self.config.epochs):
                    batches = make_batches(traj)
                    train_state, infos = self.train_step(train_state, batches)
                train_state.training_steps += 1
                print('='*70)
                print(f"training step n°{train_state.training_steps}")
                print(f"n_eps : {np.mean(roll_infos['episodes'])}")
                print(f"mean loss = {infos['loss']}")

            return train_state

        return train
    
    #-------------------------------------------------------------------------
    
    def _build_grad_loss(self)->t.Callable:
        """Summary
        
        Returns:
            t.Callable: Description
        """
        def loss_fn(params, minibatch):
            """Summary
            
            Args:
                params (TYPE): Description
                minibatch (TYPE): Description
            
            Returns:
                TYPE: Description
            """
            return config.loss_fn(params, self.apply_fn, minibatch, 
                self.config.c1, self.config.c2, self.config.epsilon)
        return jit(value_and_grad(loss_fn))
    
    #-------------------------------------------------------------------------
    
    def _init_network(self, state_dims:int, action_dims:int)->t.Tuple[t.Callable, t.Collection]:
        """Summary
        
        Args:
            state_dims (int): Description
            action_dims (int): Description
        
        Returns:
            t.Tuple[t.Callable, t.Collection]: Description
        """
        net = ActorCritic(state_dims, action_dims, **self.config.network_config)
        params = net.init(rng, jnp.zeros((state_dims,)))
        apply_fn = jit(net.apply)

        return apply_fn, params
    
    #-------------------------------------------------------------------------
    
    def _init_env(self)->t.Tuple[t.Callable, t.Callable, int, int]:
        """Summary
        
        Returns:
            t.Tuple[t.Callable, t.Callable, int, int]: Description
        """
        env, env_params = gymnax.make(self.config.env_name)
        action_dims = env.num_actions
        state_dims = env.observation_space(env_params).shape[0]
        env_step = jit(env.step)
        env_reset = jit(env.reset)

        return env_reset, env_step, state_dims, action_dims
    
    #-------------------------------------------------------------------------
    
    def _init_opt(self, params:t.Collection)->t.Tuple[t.Collection, t.Callable]:
        """Summary
        
        Args:
            params (t.Collection): Description
        
        Returns:
            t.Tuple[t.Collection, t.Callable]: Description
        """
        opt = optax.adam(config.learning_rate)
        opt_state = opt.init(params)
        opt_update = opt.update

        return opt_state, opt_update


# def init_ppo(config:Config)->t.Tuple[TrainState, t.Callable]:

#     """
#     Initialize the ppo training procedures

#     In :
#         config [Config] - collection specifying the training configuration
#     Out :
#         train_state [TrainState] : initial train state
#         train [Callable] : train function
#     """
    
#     rng = jax.random.PRNGKey(config.seed)
#     # Init env
#     env_name = config.env_name
#     env, env_params = gymnax.make(env_name)
#     action_dims = env.num_actions
#     state_dims = env.observation_space(env_params).shape[0]
#     env_step = jit(env.step)
#     env_reset = jit(env.reset)
#     # Init network
#     net = ActorCritic(state_dims, action_dims, **config.network_config)
#     params = net.init(rng, jnp.zeros((state_dims,)))
#     apply_fn = jit(net.apply)
#     # Init optimizer
#     opt = optax.adam(config.learning_rate)
#     opt_state = opt.init(params)
#     opt_update = opt.update

#     train_state = TrainState(
#         params    = params, 
#         opt_state = opt_state, 
#         )
#     #---------------------------------------------------------------------------
#     _process_trajectory = vmap(lambda traj : process_trajectory(traj, 
#                                                                 config.gamma, 
#                                                                 config.lambd), 
#                                in_axes = 1,
#                                out_axes = 1)
        
#     def make_batches(traj:ProcessedTrajectory)->t.Iterable:
#         batches = []
        
#         permut = jax.random.permutation(rng, config.T * config.n_actors)
#         shape = (config.T * config.n_actors,)
#         s = traj.s.reshape(shape+(-1,))[permut]
#         a = traj.a.reshape(shape)[permut]
#         lp = traj.lp.reshape(shape)[permut]
#         ret = traj.ret.reshape(shape)[permut]
#         adv = traj.adv.reshape(shape)[permut]
        
#         batch_size = config.batch_size * config.n_actors
#         n_batch = (config.T * config.n_actors) // batch_size
#         for i in range(n_batch):
#             batches.append((
#                 s[i*batch_size:(i+1)*batch_size],
#                 a[i*batch_size:(i+1)*batch_size],
#                 lp[i*batch_size:(i+1)*batch_size],
#                 ret[i*batch_size:(i+1)*batch_size],
#                 adv[i*batch_size:(i+1)*batch_size]
#             ))
#         return batches
#     #---------------------------------------------------------------------------
#     @partial(vmap, in_axes = (None, 0), out_axes = (1, 0))
#     def rollout(params:t.Collection, key)->Trajectory:

#         infos = {'episodes':0}
        
#         ts = jnp.zeros((config.T+1, state_dims))
#         ta = jnp.zeros((config.T+1,))
#         tlp = jnp.zeros((config.T+1,))
#         tv = jnp.zeros((config.T+1,))
#         tr = jnp.zeros((config.T+1,))
#         td = jnp.zeros((config.T+1,))

#         sample_key, reset_key, step_key = jax.random.split(rng, 3)
        
#         s, env_state = env_reset(reset_key)

#         episodes = 0
#         ep_ret = 0
        
#         for step in range(config.T+1):
            
#             logits, v = apply_fn(params, s)
#             v = v[0]
#             dist = Categorical(logits)
#             a, lp = dist.sample_and_log_prob(seed=sample_key)
#             s_, env_state, r, d, _ = env_step(step_key, env_state, a)
#             ep_ret += r

#             episodes += d.astype(int)

#             ts = ts.at[step].set(s)
#             ta = ta.at[step].set(a)
#             tlp = tlp.at[step].set(lp)
#             tv = tv.at[step].set(v)
#             tr = tr.at[step].set(r)
#             td = td.at[step].set(d.astype(float))

#             s = s_
        
#         infos['episodes'] = episodes

#         return Trajectory(s=ts, a=ta, lp=tlp, v=tv, r=tr, d=td), infos
#     #---------------------------------------------------------------------------
    
#     def loss_fn(params, minibatch):
#         return config.loss_fn(params, apply_fn, minibatch, config.c1, 
#                               config.c2, config.epsilon)
    
#     grad_loss = jit(value_and_grad(loss_fn))

#     #---------------------------------------------------------------------------
#     def train_step(train_state:TrainState, batches:t.Iterable):
            
#         infos = {"loss" : 0.}
#         params = train_state.params
#         opt_state = train_state.opt_state
#         for mb in batches:
#             loss, grads = grad_loss(params, mb)
#             infos['loss'] += loss
#             updates, opt_state = opt_update(grads, opt_state) 
#             params = optax.apply_updates(params, updates)
#         train_state.params = params
#         train_state.opt_state = opt_state

#         return train_state, infos

#     #---------------------------------------------------------------------------

#     def train(train_state:TrainState, steps:int):
#         """
#         ppo train function
#         """
#         for step in range(steps):
#             keys = jax.random.split(rng, config.n_actors)
#             traj, roll_infos = rollout(train_state.params, keys)
#             traj = _process_trajectory(traj)
#             for epoch in range(config.epochs):
#                 batches = make_batches(traj)
#                 train_state, infos = train_step(train_state, batches)
#             train_state.training_steps += 1
#             print('='*70)
#             print(f"training step n°{train_state.training_steps}")
#             print(f"n_eps : {np.mean(roll_infos['episodes'])}")
#             print(f"mean loss = {infos['loss']}")

#         return train_state

#     return train_state, train