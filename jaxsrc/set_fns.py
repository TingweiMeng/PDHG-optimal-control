import jax.numpy as jnp
from functools import partial
from einshape import jax_einshape as einshape
import jax
from collections import namedtuple

jax.config.update("jax_enable_x64", True)


def set_up_J(egno, ndim, period_spatial):
  if egno > 0:  # sin
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
    f_fn = lambda alp, x_arr, t_arr: -alp  # [..., dim_ctrl] -> [..., dim_state]
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
    elif ndim == 2:
      pass
    else:
      raise ValueError("ndim {} not implemented".format(ndim))
  elif egno > 1 and egno <= 5 and ndim == 1: # f = -a(x) * alp, L = |alp|^2/2, dim_ctrl = dim_state = ndim = 1
    if egno == 2:  # a(x) = |x-1| + 0.1
      coeff_fn = lambda x_arr, t_arr: jnp.abs(x_arr[...,0] - 1.0) + 0.1  # [..., ndim] -> [..., 1]
    elif egno == 3:  # a(x) = |x-1| - 0.1
      coeff_fn = lambda x_arr, t_arr: jnp.abs(x_arr - 1.0) - 0.1  # [..., ndim] -> [..., 1]
    elif egno == 4:  # a(x) = |x-1| - 0.5
      coeff_fn = lambda x_arr, t_arr: jnp.abs(x_arr - 1.0) - 0.5  # [..., ndim] -> [..., 1]
    elif egno == 5:  # a(x) = |x-1| - 1.0
      coeff_fn = lambda x_arr, t_arr: jnp.abs(x_arr - 1.0) - 1.0  # [..., ndim] -> [..., 1]
    H_plus_fn = lambda p, x_arr, t_arr: (jnp.maximum(p,0) * coeff_fn(x_arr, t_arr)[...,0]) **2/2
    H_minus_fn = lambda p, x_arr, t_arr: (jnp.minimum(p,0) * coeff_fn(x_arr, t_arr)[...,0]) **2/2
    f_fn = lambda alp, x_arr, t_arr: -alp * coeff_fn(x_arr, t_arr)  # [..., dim_ctrl] -> [..., dim_state]
    L_fn = lambda alp, x_arr, t_arr: jnp.sum(alp**2, axis = -1)/2
    def alp_update_fn(alp_prev, Dx_right_phi, Dx_left_phi, rho, sigma, x_arr, t_arr):
      alp1_prev, alp2_prev = alp_prev  # [nt-1, nx, 1]
      eps = 1e-4
      param_inv = (rho + eps) / sigma
      param_inv = param_inv[...,None]  # [nt-1, nx, 1]
      Dx_right_phi = Dx_right_phi[...,None]  # [nt-1, nx, 1]
      Dx_left_phi = Dx_left_phi[...,None]  # [nt-1, nx, 1]
      alp1_next = (Dx_right_phi * coeff_fn(x_arr, t_arr) + param_inv * alp1_prev) / (1 + param_inv)
      alp1_next = (alp1_next * (f_fn(alp1_next, x_arr, t_arr) >= 0.0))
      alp2_next = (Dx_left_phi * coeff_fn(x_arr, t_arr) + param_inv * alp2_prev) / (1 + param_inv)
      alp2_next = (alp2_next * (f_fn(alp2_next, x_arr, t_arr) < 0.0))
      return (alp1_next, alp2_next)
  else:
    raise ValueError("egno {} not implemented".format(egno))
  
  if ndim == 1:
    f_plus_fn = lambda alp, x_arr, t_arr: jnp.maximum(f_fn(alp, x_arr, t_arr)[...,0], 0.0)  # [..., dim_ctrl] -> [...]
    f_minus_fn = lambda alp, x_arr, t_arr: jnp.minimum(f_fn(alp, x_arr, t_arr)[...,0], 0.0)
    L1_fn = lambda alp, x_arr, t_arr: L_fn(alp, x_arr, t_arr) * (f_fn(alp, x_arr, t_arr)[...,0] >= 0.0)
    L2_fn = lambda alp, x_arr, t_arr: L_fn(alp, x_arr, t_arr) * (f_fn(alp, x_arr, t_arr)[...,0] < 0.0)
  else: # 1 stands for the corresponding coordinate positive, 2 stands for negative
    indicator_11_fn = lambda alp, x_arr, t_arr: (f_fn(alp, x_arr, t_arr)[...,0] >= 0.0) & (f_fn(alp, x_arr, t_arr)[...,1] >= 0.0)  # [..., dim_ctrl] -> [...]
    indicator_12_fn = lambda alp, x_arr, t_arr: (f_fn(alp, x_arr, t_arr)[...,0] >= 0.0) & (f_fn(alp, x_arr, t_arr)[...,1] < 0.0)
    indicator_21_fn = lambda alp, x_arr, t_arr: (f_fn(alp, x_arr, t_arr)[...,0] < 0.0) & (f_fn(alp, x_arr, t_arr)[...,1] >= 0.0)
    indicator_22_fn = lambda alp, x_arr, t_arr: (f_fn(alp, x_arr, t_arr)[...,0] < 0.0) & (f_fn(alp, x_arr, t_arr)[...,1] < 0.0)
    def alp_update_base_fn(alp_prev, Dphi, param_inv, x_arr, t_arr):
      # solves min_alp param_inv * |alp - alp_prev|^2/2 + <alp, Dphi> + |alp|^2/2
      alp_next = (param_inv * alp_prev + Dphi) / (1 + param_inv)
      return alp_next
    def alp_update_fn(alp_prev, Dphi, rho, sigma, x_arr, t_arr):  # Dphi is a tuple including D11_phi, D12_phi, D21_phi, D22_phi
      alp11_prev, alp12_prev, alp21_prev, alp22_prev = alp_prev  # [nt-1, nx, ny, 2]
      D11_phi, D12_phi, D21_phi, D22_phi = Dphi
      eps = 1e-4
      param_inv = (rho[...,None] + eps) / sigma  # [nt-1, nx, ny, 1]
      alp11_next = alp_update_base_fn(alp11_prev, D11_phi, param_inv, x_arr, t_arr)  # [nt-1, nx, ny, 2]
      alp11_next *= indicator_11_fn(alp11_next, x_arr, t_arr)[...,None]  # [nt-1, nx, ny, 2]
      alp12_next = alp_update_base_fn(alp12_prev, D12_phi, param_inv, x_arr, t_arr)
      alp12_next *= indicator_12_fn(alp12_next, x_arr, t_arr)[...,None]
      alp21_next = alp_update_base_fn(alp21_prev, D21_phi, param_inv, x_arr, t_arr)
      alp21_next *= indicator_21_fn(alp21_next, x_arr, t_arr)[...,None]
      alp22_next = alp_update_base_fn(alp22_prev, D22_phi, param_inv, x_arr, t_arr)
      alp22_next *= indicator_22_fn(alp22_next, x_arr, t_arr)[...,None]
      return (alp11_next, alp12_next, alp21_next, alp22_next)
  
  if ndim == 1:
    Functions = namedtuple('Functions', ['f_plus_fn', 'f_minus_fn', 'L1_fn', 'L2_fn', 'alp_update_fn',
                                        'H_plus_fn', 'H_minus_fn'])
    fns_dict = Functions(H_plus_fn=H_plus_fn, H_minus_fn=H_minus_fn,
                        f_plus_fn=f_plus_fn, f_minus_fn=f_minus_fn, L1_fn=L1_fn, L2_fn=L2_fn,
                        alp_update_fn = alp_update_fn)
  elif ndim == 2:
    Functions = namedtuple('Functions', ['f_fn', 'L_fn', 'alp_update_fn', 
                                          'indicator_11_fn', 'indicator_12_fn', 'indicator_21_fn', 'indicator_22_fn',
                                          'Hx_plus_fn', 'Hx_minus_fn', 'Hy_plus_fn', 'Hy_minus_fn'])
    fns_dict = Functions(f_fn = f_fn, L_fn = L_fn, alp_update_fn = alp_update_fn,
                        indicator_11_fn = indicator_11_fn, indicator_12_fn = indicator_12_fn,
                        indicator_21_fn = indicator_21_fn, indicator_22_fn = indicator_22_fn,
                        Hx_plus_fn=H_plus_fn, Hx_minus_fn=H_minus_fn, Hy_plus_fn=H_plus_fn, Hy_minus_fn=H_minus_fn)
                        
  else:
    raise ValueError("ndim {} not implemented".format(ndim))
  return fns_dict
