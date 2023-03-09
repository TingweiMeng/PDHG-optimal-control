from jax import lax
import jax.numpy as jnp
from functools import partial
from einshape import jax_einshape as einshape
import jax
jax.config.update("jax_enable_x64", True)


def tridiagonal_solve(dl, d, du, b): 
  """Pure JAX implementation of `tridiagonal_solve`.""" 
  prepend_zero = lambda x: jnp.append(jnp.zeros([1], dtype=x.dtype), x[:-1]) 
  fwd1 = lambda tu_, x: x[1] / (x[0] - x[2] * tu_) 
  fwd2 = lambda b_, x: (x[0] - x[3] * b_) / (x[1] - x[3] * x[2]) 
  bwd1 = lambda x_, x: x[0] - x[1] * x_ 
  double = lambda f, args: (f(*args), f(*args)) 

  # Forward pass. 
  _, tu_ = lax.scan(lambda tu_, x: double(fwd1, (tu_, x)), 
                    du[0] / d[0], 
                    (d, du, dl), 
                    unroll=32) 

  _, b_ = lax.scan(lambda b_, x: double(fwd2, (b_, x)), 
                  b[0] / d[0], 
                  (b, d, prepend_zero(tu_), dl), 
                  unroll=32) 

  # Backsubstitution. 
  _, x_ = lax.scan(lambda x_, x: double(bwd1, (x_, x)), 
                  b_[-1], 
                  (b_[::-1], tu_[::-1]), 
                  unroll=32) 

  return x_[::-1] 

# batch in axis 1
tridiagonal_solve_batch = jax.vmap(tridiagonal_solve, in_axes=(None, 1, None, 1), out_axes=1)

if __name__ == "__main__":

  n = 20
  dl = jnp.array([0.0] + [0.1] * (n-1)).astype(jnp.complex128)
  du = jnp.array([0.1] * (n-1) + [0.0]).astype(jnp.complex128)
  d = jnp.ones((n,)).astype(jnp.complex128)
  b = 0.1 * jnp.arange(n).astype(jnp.complex128) 
  out = tridiagonal_solve(dl, d, du, b)
  print(out.shape, out)

  print('=======')
  bs = 3
  d = jnp.ones((n, bs)).astype(jnp.complex128)
  b = jax.random.uniform(jax.random.PRNGKey(1), shape = (n, bs)).astype(jnp.complex128)
  out = tridiagonal_solve_batch(dl, d, du, b)
  print(out.shape, out)
  
