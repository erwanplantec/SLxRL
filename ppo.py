import jax
from jax import jit, vmap, value_and_grad
import chex
from distrax import Categorical
import rlax
import numpy as np
from utils import *

@chex.dataclass
class Config:
    training_steps:int
    T:int
    epochs:int
    n_actors:int

    loss_fn:t.Callable

    env_name:str
    
    network_config : t.Collection
    learning_rate : float

    seed:int
    
    c1:float
    c2:float
    gamma:float
    lambd:float
    epsilon:float
    
    mem_sz:int
    batch_size:int

@chex.dataclass
class TrainState:
    params : t.Collection
    opt_state : t.Collection
    training_steps : int = 0

def ppo_loss(params, apply_fn, minibatch, c1, c2, eps):
    
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

def init_ppo(config:Config)->t.Tuple[TrainState, t.Callable]:

    """
    Initialize the ppo training procedures

    In :
        config [Config] - collection specifying the training configuration
    Out :
        train_state [TrainState] : initial train state
        train [Callable] : train function
    """
    
    rng = jax.random.PRNGKey(config.seed)
    # Init env
    env_name = config.env_name
    env, env_params = gymnax.make(env_name)
    action_dims = env.num_actions
    state_dims = env.observation_space(env_params).shape[0]
    env_step = jit(env.step)
    env_reset = jit(env.reset)
    # Init network
    net = ActorCritic(state_dims, action_dims, **config.network_config)
    params = net.init(rng, jnp.zeros((state_dims,)))
    apply_fn = jit(net.apply)
    # Init optimizer
    opt = optax.adam(config.learning_rate)
    opt_state = opt.init(params)
    opt_update = opt.update

    train_state = TrainState(
        params    = params, 
        opt_state = opt_state, 
        )
    #---------------------------------------------------------------------------
    _process_trajectory = vmap(lambda traj : process_trajectory(traj, 
                                                                config.gamma, 
                                                                config.lambd), 
                               in_axes = 1,
                               out_axes = 1)
        
    def make_batches(traj:ProcessedTrajectory)->t.Iterable:
        batches = []
        
        permut = jax.random.permutation(rng, config.T * config.n_actors)
        shape = (config.T * config.n_actors,)
        s = traj.s.reshape(shape+(-1,))[permut]
        a = traj.a.reshape(shape)[permut]
        lp = traj.lp.reshape(shape)[permut]
        ret = traj.ret.reshape(shape)[permut]
        adv = traj.adv.reshape(shape)[permut]
        
        batch_size = config.batch_size * config.n_actors
        n_batch = (config.T * config.n_actors) // batch_size
        for i in range(n_batch):
            batches.append((
                s[i*batch_size:(i+1)*batch_size],
                a[i*batch_size:(i+1)*batch_size],
                lp[i*batch_size:(i+1)*batch_size],
                ret[i*batch_size:(i+1)*batch_size],
                adv[i*batch_size:(i+1)*batch_size]
            ))
        return batches
    #---------------------------------------------------------------------------
    @partial(vmap, in_axes = (None, 0), out_axes = (1, 0))
    def rollout(params:t.Collection, key)->Trajectory:

        infos = {'episodes':0}
        
        ts = jnp.zeros((config.T+1, state_dims))
        ta = jnp.zeros((config.T+1,))
        tlp = jnp.zeros((config.T+1,))
        tv = jnp.zeros((config.T+1,))
        tr = jnp.zeros((config.T+1,))
        td = jnp.zeros((config.T+1,))

        sample_key, reset_key, step_key = jax.random.split(rng, 3)
        
        s, env_state = env_reset(reset_key)

        episodes = 0
        ep_ret = 0
        
        for step in range(config.T+1):
            
            logits, v = apply_fn(params, s)
            v = v[0]
            dist = Categorical(logits)
            a, lp = dist.sample_and_log_prob(seed=sample_key)
            s_, env_state, r, d, _ = env_step(step_key, env_state, a)
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
    #---------------------------------------------------------------------------
    
    def loss_fn(params, minibatch):
        return config.loss_fn(params, apply_fn, minibatch, config.c1, 
                              config.c2, config.epsilon)
    
    grad_loss = jit(value_and_grad(loss_fn))

    #---------------------------------------------------------------------------
    def train_step(train_state:TrainState, batches:t.Iterable):
            
        infos = {"loss" : 0.}
        params = train_state.params
        opt_state = train_state.opt_state
        for mb in batches:
            loss, grads = grad_loss(params, mb)
            infos['loss'] += loss
            updates, opt_state = opt_update(grads, opt_state) 
            params = optax.apply_updates(params, updates)
        train_state.params = params
        train_state.opt_state = opt_state

        return train_state, infos

    #---------------------------------------------------------------------------

    def train(train_state:TrainState):
        
        n_steps = config.training_steps - train_state.training_steps
        for step in range(n_steps):
            keys = jax.random.split(rng, config.n_actors)
            traj, roll_infos = rollout(train_state.params, keys)
            traj = _process_trajectory(traj)
            for epoch in range(config.epochs):
                batches = make_batches(traj)
                train_state, infos = train_step(train_state, batches)
            train_state.training_steps += 1
            print('='*70)
            print(f"training step n°{train_state.training_steps}")
            print(f"n_eps : {np.mean(roll_infos['episodes'])}")
            print(f"mean loss = {infos['loss']}")

        return train_state

    return train_state, train