import jax.numpy as jnp
import typing as t

def build_network(I : jnp.ndarray):
	""""""
	
	n = I.shape[0]
	idxs = jnp.arange(n).astype(int)
	neighs = [idxs[i] for i in I]
	
	def neighbors(i:int)->t.Iterable[int]:
		return neighs[i]

	return neighbors