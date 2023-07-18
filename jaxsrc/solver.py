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
tridiagonal_solve_batch_2d = jax.vmap(jax.vmap(tridiagonal_solve, in_axes=(None, -1, None, -1), out_axes=(-1)), 
                            in_axes=(None, -1, None, -1), out_axes=(-1))

def interpolation_t(fn_left, fn_right, scaling_num):
  '''
  Note: the interpolated matrix does not include the last element of fn_right
  @ parameters:
    fn_left, fn_right: [n, ...]
    scaling_num: int
  @ return:
    fn_dense: [n * scaling_num, ...]
  '''
  fn_dense = einshape('k...->(kj)...', fn_left, j = scaling_num)
  incremental = (fn_right - fn_left) / scaling_num
  for i in range(scaling_num):
    fn_dense = fn_dense.at[i::scaling_num, ...].set(fn_left + i * incremental)
    # fn_dense[i::scaling_num, ...] = fn_left + i * incremental
  return fn_dense

def interpolation_x(fn_left, fn_right, scaling_num):
  '''
  Note: the interpolated matrix does not include the last element of fn_right
  @ parameters:
    fn_left, fn_right: [nt, n, ...]
    scaling_num: int
  @ return:
    fn_dense: [nt, n * scaling_num, ...]
  '''
  fn_dense = einshape('ik->i(jk)', fn_left, j = scaling_num)
  incremental = (fn_right - fn_left) / scaling_num
  for i in range(scaling_num):
    fn_dense = fn_dense.at[:, i::scaling_num, ...].set(fn_left + i * incremental)
    # fn_dense[:, i::scaling_num, ...] = fn_left + i * incremental
  return fn_dense

def interpolation_y(fn_left, fn_right, scaling_num):
  '''
  Note: the interpolated matrix does not include the last element of fn_right
        this function is only used for 2d cases
  @ parameters:
    fn_left, fn_right: [nt, nx, n]
    scaling_num: int
  @ return:
    fn_dense: [nt, nx, n * scaling_num]
  '''
  fn_dense = einshape('ilk->il(kj)', fn_left, j = scaling_num)
  incremental = (fn_right - fn_left) / scaling_num
  for i in range(scaling_num):
    fn_dense = fn_dense.at[..., i::scaling_num].set(fn_left + i * incremental)
    # fn_dense[..., i::scaling_num] = fn_left + i * incremental
  return fn_dense

def set_up_example(egno, ndim, nt, nx, ny, x_period, y_period):
  '''
  @ parameters:
    egno, ndim, nt, nx, ny: int
    x_period, y_period: scalars
  @ return:
    J, f_in_H_fn, c_in_H_fn: functions
    filename: string
  '''
  if ndim == 1:
    alpha = 2 * jnp.pi / x_period
    filename = './jaxsrc/eg{}_1d/nt{}_nx{}'.format(egno, nt, nx)
  else:
    alpha = jnp.array([2 * jnp.pi / x_period, 2 * jnp.pi / y_period])
    filename = './jaxsrc/eg{}_2d/nt{}_nx{}_ny{}'.format(egno, nt, nx, ny)
  
  J = lambda x: jnp.sum(jnp.sin(alpha * x), axis = -1)  # input [...,ndim] output [...]

  if egno == 1:
    # example 1
    f_in_H_fn = lambda x: jnp.zeros_like(x[...,0])
    c_in_H_fn = lambda x: 1 + 3* jnp.exp(-4 * jnp.sum((x-1) * (x-1), axis = -1))
  elif egno == 2:
    # example 2
    f_in_H_fn = lambda x: 1 + 3* jnp.exp(-4 * jnp.sum((x-1) * (x-1), axis = -1))
    c_in_H_fn = lambda x: jnp.zeros_like(x[...,0]) + 1
  elif egno == 0:  # x-indep case
    f_in_H_fn = lambda x: jnp.zeros_like(x[...,0])
    c_in_H_fn = lambda x: jnp.zeros_like(x[...,0]) + 1
  else:
    raise ValueError("egno {} not implemented".format(egno))
  return J, f_in_H_fn, c_in_H_fn, filename

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
  
