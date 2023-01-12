import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad
import flax.linen as nn
import typing as t

class ActorCritic(nn.Module):

    state_dims : int
    action_dims : int
    act_fn : t.Callable = nn.relu
    hidden_dims : t.Iterable[int] = (128,)

    def setup(self):
        self.layers = tuple([nn.Dense(dim) for dim in self.hidden_dims])
        self.action_head = nn.Dense(self.action_dims)
        self.critic_head = nn.Dense(1)
    
    def __call__(self, s):
        x = s
        for layer in self.layers:
            x = self.act_fn(layer(s))
        logits = nn.relu(self.action_head(x))
        v = self.critic_head(x)
        return logits, v