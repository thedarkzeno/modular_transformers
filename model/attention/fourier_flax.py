import flax.linen as nn
import jax
import jax.numpy as jnp

class Fourier_transform_flax(nn.Module): 
  @nn.compact
  def __call__(self , x):
    return jax.vmap(jnp.fft.fftn)(x).real