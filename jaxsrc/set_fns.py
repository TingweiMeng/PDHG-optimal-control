import jax.numpy as jnp
from functools import partial
from einshape import jax_einshape as einshape
import jax
from collections import namedtuple

jax.config.update("jax_enable_x64", True)


def set_up_J(egno, ndim, period_spatial):
  if egno == 1:  # sin
    if ndim == 1:
      x_period = period_spatial[0]
      alpha = 2 * jnp.pi / x_period
    elif ndim == 2:
      x_period, y_period = period_spatial
      alpha = jnp.array([2 * jnp.pi / x_period, 2 * jnp.pi / y_period])
    else:
      raise ValueError("ndim {} not implemented".format(ndim))
    J = lambda x: jnp.sum(jnp.sin(alpha * x), axis = -1)  # [...,ndim] -> [...]
  else:
    raise ValueError("egno {} not implemented".format(egno))
  return J


def set_up_example_fns(egno, ndim):
  '''
  @ parameters:
    egno: int
    ndim: int
  @ return:
    J: initial condition, function
    fns_dict: named tuple of functions
  '''
  print('egno: ', egno, flush=True)
  # omit the indicator function
  # note: dim of p is [nt-1, nx]
  # H_plus_fn, H_minus_fn, H_fn are only used in this function and compute_HJ_residual_EO_1d_general, compute_HJ_residual_EO_2d_general, compute_EO_forward_solution_1d_general, compute_EO_forward_solution_2d_general
  if egno == 1:  # f = -alp, L = |alp|^2/2, dim_ctrl = dim_state = ndim
    # TODO: H_plus and H_minus are used in EO scheme to measure the performance. Any better measure?
    H_plus_fn = lambda p, x_arr, t_arr: jnp.maximum(p,0) **2/2  # [...] -> [...]
    H_minus_fn = lambda p, x_arr, t_arr: jnp.minimum(p,0) **2/2
    f_fn = lambda alp, x_arr, t_arr: -alp  # [..., ndim] -> [..., ndim]
    L_fn = lambda alp, x_arr, t_arr: jnp.sum(alp**2, axis = -1)/2  # [..., ndim] -> [...]
    if ndim == 1:
      def alp_update_fn(alp_prev, Dx_right_phi, Dx_left_phi, rho, sigma, x_arr, t_arr):
        alp1_prev, alp2_prev = alp_prev  # [nt-1, nx, 1]
        eps = 1e-4
        param_inv = (rho + eps) / sigma  # [nt-1, nx]
        alp1_next = (Dx_right_phi + param_inv * alp1_prev[...,0]) / (1 + param_inv)
        alp1_next = jnp.minimum(alp1_next, 0.0)[...,None]
        alp2_next = (Dx_left_phi + param_inv * alp2_prev[...,0]) / (1 + param_inv)
        alp2_next = jnp.maximum(alp2_next, 0.0)[...,None]
        return (alp1_next, alp2_next)
  else:
    raise ValueError("egno {} not implemented".format(egno))
  
  if ndim == 1:
    f_plus_fn = lambda alp, x_arr, t_arr: jnp.maximum(f_fn(alp, x_arr, t_arr)[...,0], 0.0)  # [...,ndim] -> [...]
    f_minus_fn = lambda alp, x_arr, t_arr: jnp.minimum(f_fn(alp, x_arr, t_arr)[...,0], 0.0)
    L1_fn = lambda alp, x_arr, t_arr: L_fn(alp, x_arr, t_arr) * (f_fn(alp, x_arr, t_arr)[...,0] >= 0.0)
    L2_fn = lambda alp, x_arr, t_arr: L_fn(alp, x_arr, t_arr) * (f_fn(alp, x_arr, t_arr)[...,0] < 0.0)
  else: # TODO
    f_plus_x_fn = lambda alp, x_arr, t_arr: jnp.maximum(f_fn(alp, x_arr, t_arr)[...,0], 0.0)  # [..., dim_ctrl] -> [...]
    f_minus_fn = lambda alp, x_arr, t_arr: jnp.minimum(f_fn(alp, x_arr, t_arr), 0.0)
    L1_fn = lambda alp, x_arr, t_arr: L_fn(alp, x_arr, t_arr) * (f_fn(alp, x_arr, t_arr) >= 0.0)
    L2_fn = lambda alp, x_arr, t_arr: L_fn(alp, x_arr, t_arr) * (f_fn(alp, x_arr, t_arr) < 0.0)
  
  if ndim == 1:
    Functions = namedtuple('Functions', ['f_plus_fn', 'f_minus_fn', 'L1_fn', 'L2_fn', 'alp_update_fn',
                                        'H_plus_fn', 'H_minus_fn'])
    fns_dict = Functions(H_plus_fn=H_plus_fn, H_minus_fn=H_minus_fn,
                        f_plus_fn=f_plus_fn, f_minus_fn=f_minus_fn, L1_fn=L1_fn, L2_fn=L2_fn,
                        alp_update_fn = alp_update_fn)
  else:
    raise ValueError("ndim {} not implemented".format(ndim))
  return fns_dict
