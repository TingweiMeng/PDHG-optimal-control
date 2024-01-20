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

def set_up_numerical_L(egno, ndim, ind):
  # numerical L is a function taking alp, x, t as input, where alp is a tuple containing alp1, alp2 in 1d and alp11, alp12, alp21, alp22 in 2d
  # for now, all L fns are |alp|^2/2
  L_fn_1d = lambda alp, x_arr, t_arr: alp[...,0]**2/2  # [..., 1] -> [...]
  DL_fn_1d = lambda alp, x_arr, t_arr: alp  # [..., 1] -> [..., 1]
  # L_fn_2d = lambda alp_x, alp_y, x_arr, t_arr: L_fn_1d(alp_x, x_arr, t_arr) + L_fn_1d(alp_y, x_arr, t_arr)  # [..., 1], [..., 1] -> [...]
  L_fn_2d = lambda alp, x_arr, t_arr: alp[...,0]**2/2 + alp[...,1]**2/2  # [..., 2] -> [...]
  DL_fn_2d = lambda alp, x_arr, t_arr: alp  # [..., 2] -> [..., 2]
  HL_fn_2d = lambda alp, x_arr, t_arr: jnp.concatenate([jnp.stack([jnp.ones_like(alp[...,0:1]), jnp.zeros_like(alp[...,0:1])], axis = -1),
                                                        jnp.stack([jnp.zeros_like(alp[...,0:1]), jnp.ones_like(alp[...,0:1])], axis = -1)], axis = -2)  # [..., 2] -> [..., 2, 2]
  if ndim == 1:
    L_fn = lambda alp, x_arr, t_arr: (L_fn_1d(alp[0] + alp[1], x_arr, t_arr) + L_fn_1d(alp[0] - alp[1], x_arr, t_arr))/2
    # DL_fn = lambda alp, x_arr, t_arr: (DL_fn_1d(alp[0] + alp[1], x_arr, t_arr) + DL_fn_2d(alp[0] - alp[1], x_arr, t_arr))/2
  elif ndim == 2: # each component in alp is [..., nctrl]
    # if ind == 0:
    #   L_fn = lambda alp, x_arr, t_arr: (L_fn_2d(alp[0] + alp[1], alp[2] + alp[3], x_arr, t_arr) + L_fn_2d(alp[0] - alp[1], alp[2] - alp[3], x_arr, t_arr))/2
    if ind == 1:
      L_fn = lambda alp, x_arr, t_arr: L_fn_2d(alp[0] + alp[1] + alp[2] + alp[3], x_arr, t_arr)
      DL_fn = lambda alp, x_arr, t_arr: DL_fn_2d(alp[0] + alp[1] + alp[2] + alp[3], x_arr, t_arr)
      HL_fn = lambda alp, x_arr, t_arr: HL_fn_2d(alp[0] + alp[1] + alp[2] + alp[3], x_arr, t_arr)
    else:
      raise ValueError("ind {} not implemented".format(ind))
  else:
    raise ValueError("ndim {} not implemented".format(ndim))
  return L_fn, DL_fn, HL_fn
  

def set_up_example_fns(egno, ndim, numerical_L_ind):
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
  if ndim == 2:  # f = -alp, dim_ctrl = dim_state = ndim
    if egno == 1:
      # TODO: H_plus and H_minus are used in EO scheme to measure the performance. Any better measure?
      H_plus_fn = lambda p, x_arr, t_arr: jnp.maximum(p,0) **2/2  # [...] -> [...]
      H_minus_fn = lambda p, x_arr, t_arr: jnp.minimum(p,0) **2/2
      coeff_fn_1 = lambda x_arr, t_arr: jnp.concatenate([jnp.ones_like(x_arr[...,0:1]), jnp.zeros_like(x_arr[...,0:1])], axis = -1)  # [..., ndim] -> [..., ndim]
      coeff_fn_2 = lambda x_arr, t_arr: jnp.concatenate([jnp.zeros_like(x_arr[...,0:1]), jnp.ones_like(x_arr[...,0:1])], axis = -1)  # [..., ndim] -> [..., ndim]
      f_fn = lambda alp, x_arr, t_arr: - jnp.concatenate([jnp.sum(coeff_fn_1(x_arr, t_arr) * alp, axis = -1, keepdims = True), 
                                                          jnp.sum(coeff_fn_2(x_arr, t_arr) * alp, axis = -1, keepdims = True)], axis = -1)  # [..., 2] -> [..., 2]
      Df1_fn = lambda alp, x_arr, t_arr: - coeff_fn_1(x_arr, t_arr)  # [..., 2] -> [..., 2]
      Df2_fn = lambda alp, x_arr, t_arr: - coeff_fn_2(x_arr, t_arr)
      Hf1_fn = lambda alp, x_arr, t_arr: jnp.zeros(alp.shape + alp.shape[-1:])  # [..., 2] -> [..., 2, 2]
      Hf2_fn = lambda alp, x_arr, t_arr: jnp.zeros(alp.shape + alp.shape[-1:])
      def alp_update_base_fn(alp_prev, Dphi, param_inv, x_arr, t_arr):
        # solves min_alp param_inv * |alp - alp_prev|^2/2 + <alp, Dphi> + |alp|^2/2
        ''' @ parameters:
              alp_prev: [nt-1, nx, ny, 1]
              Dphi: [nt-1, nx, ny]
              param_inv: [nt-1, nx, ny, 1]
              x_arr: vec that can be broadcasted to [nt-1, nx, ny]
              t_arr: vec that can be broadcasted to [nt-1, nx, ny]
            @ return:
              alp_next: [nt-1, nx, ny, 1]
        '''
        alp_next = (param_inv * alp_prev + Dphi[...,None]) / (1 + param_inv)
        return alp_next
      def alp_update_fn(alp_prev, Dphi, rho, sigma, x_arr, t_arr):  # Dphi is a tuple including D11_phi, D12_phi, D21_phi, D22_phi
        alp1_x_prev, alp2_x_prev, alp1_y_prev, alp2_y_prev = alp_prev  # [nt-1, nx, ny, 1]
        Dx_right_phi, Dx_left_phi, Dy_right_phi, Dy_left_phi = Dphi  # [nt-1, nx, ny]
        eps = 1e-4
        param_inv = (rho[...,None] + eps) / sigma  # [nt-1, nx, ny, 1]
        alp1_x_next = alp_update_base_fn(alp1_x_prev[...,0:1], Dx_right_phi, param_inv, x_arr, t_arr)
        alp1_x_next *= (f_fn(alp1_x_next, x_arr, t_arr)[...,0:1] >= 0.0)  # [nt-1, nx, ny, 1]
        alp1_x_next = jnp.pad(alp1_x_next, ((0,0),(0,0),(0,0),(0,1)), mode='constant')  # [nt-1, nx, ny, 2]
        alp2_x_next = alp_update_base_fn(alp2_x_prev[...,0:1], Dx_left_phi, param_inv, x_arr, t_arr)
        alp2_x_next *= (f_fn(alp2_x_next, x_arr, t_arr)[...,0:1] < 0.0)  # [nt-1, nx, ny, 1]
        alp2_x_next = jnp.pad(alp2_x_next, ((0,0),(0,0),(0,0),(0,1)), mode='constant')  # [nt-1, nx, ny, 2]
        alp1_y_next = alp_update_base_fn(alp1_y_prev[...,1:2], Dy_right_phi, param_inv, x_arr, t_arr)
        alp1_y_next *= (f_fn(alp1_y_next, x_arr, t_arr)[...,0:1] >= 0.0)  # [nt-1, nx, ny, 1]
        alp1_y_next = jnp.pad(alp1_y_next, ((0,0),(0,0),(0,0),(1,0)), mode='constant')  # [nt-1, nx, ny, 2]
        alp2_y_next = alp_update_base_fn(alp2_y_prev[...,1:2], Dy_left_phi, param_inv, x_arr, t_arr)
        alp2_y_next *= (f_fn(alp2_y_next, x_arr, t_arr)[...,0:1] < 0.0)  # [nt-1, nx, ny, 1]
        alp2_y_next = jnp.pad(alp2_y_next, ((0,0),(0,0),(0,0),(1,0)), mode='constant')  # [nt-1, nx, ny, 2]
        return (alp1_x_next, alp2_x_next, alp1_y_next, alp2_y_next)
    elif egno == 2:  # f = -(|x|^2-1) alp, L = |alp|^2/2
      coeff_fn = lambda x_arr, t_arr: (x_arr - 1.0)**2 + 0.1  # [..., ndim] -> [..., ndim]
      c_fn = lambda x_arr, t_arr: jnp.ones_like(x_arr)  # [..., ndim] -> [..., ndim]
      # Note: H is seperable, each dim has the same fn, so just choose the first dim
      H_plus_fn = lambda p, x_arr, t_arr: (jnp.maximum(p,0) * coeff_fn(x_arr, t_arr)[...,0]) **2/2 * c_fn(x_arr, t_arr)[...,0]
      H_minus_fn = lambda p, x_arr, t_arr: (jnp.minimum(p,0) * coeff_fn(x_arr, t_arr)[...,0]) **2/2 * c_fn(x_arr, t_arr)[...,0]
      f_fn = lambda alp, x_arr, t_arr: -alp * coeff_fn(x_arr, t_arr)  # [..., dim_ctrl] -> [..., dim_state] or [..., 1] -> [..., 1]
      def alp_update_base_fn(alp_prev, Dphi, param_inv, x_arr, t_arr):
        # solves min_alp param_inv * |alp - alp_prev|^2/2 + <alp, Dphi> + |alp|^2/2
        ''' @ parameters:
              alp_prev: [nt-1, nx, ny, 1]
              Dphi: [nt-1, nx, ny]
              param_inv: [nt-1, nx, ny, 1]
              x_arr: vec that can be broadcasted to [nt-1, nx, ny]
              t_arr: vec that can be broadcasted to [nt-1, nx, ny]
            @ return:
              alp_next: [nt-1, nx, ny, 1]
        '''
        alp_next = (Dphi[...,None] * coeff_fn(x_arr, t_arr) + param_inv * alp_prev) / (1/c_fn(x_arr, t_arr) + param_inv)
        return alp_next
      def alp_update_fn(alp_prev, Dphi, rho, sigma, x_arr, t_arr):  # Dphi is a tuple including D11_phi, D12_phi, D21_phi, D22_phi
        alp1_x_prev, alp2_x_prev, alp1_y_prev, alp2_y_prev = alp_prev  # [nt-1, nx, ny, 1]
        Dx_right_phi, Dx_left_phi, Dy_right_phi, Dy_left_phi = Dphi  # [nt-1, nx, ny]
        eps = 1e-4
        param_inv = (rho[...,None] + eps) / sigma  # [nt-1, nx, ny, 1]
        alp1_x_next = alp_update_base_fn(alp1_x_prev[...,0:1], Dx_right_phi, param_inv, x_arr, t_arr)
        alp1_x_next *= (f_fn(alp1_x_next, x_arr, t_arr)[...,0:1] >= 0.0)  # [nt-1, nx, ny, 1]
        alp1_x_next = jnp.pad(alp1_x_next, ((0,0),(0,0),(0,0),(0,1)), mode='constant')  # [nt-1, nx, ny, 2]
        alp2_x_next = alp_update_base_fn(alp2_x_prev[...,0:1], Dx_left_phi, param_inv, x_arr, t_arr)
        alp2_x_next *= (f_fn(alp2_x_next, x_arr, t_arr)[...,0:1] < 0.0)  # [nt-1, nx, ny, 1]
        alp2_x_next = jnp.pad(alp2_x_next, ((0,0),(0,0),(0,0),(0,1)), mode='constant')  # [nt-1, nx, ny, 2]
        alp1_y_next = alp_update_base_fn(alp1_y_prev[...,1:2], Dy_right_phi, param_inv, x_arr, t_arr)
        alp1_y_next *= (f_fn(alp1_y_next, x_arr, t_arr)[...,0:1] >= 0.0)  # [nt-1, nx, ny, 1]
        alp1_y_next = jnp.pad(alp1_y_next, ((0,0),(0,0),(0,0),(1,0)), mode='constant')  # [nt-1, nx, ny, 2]
        alp2_y_next = alp_update_base_fn(alp2_y_prev[...,1:2], Dy_left_phi, param_inv, x_arr, t_arr)
        alp2_y_next *= (f_fn(alp2_y_next, x_arr, t_arr)[...,0:1] < 0.0)  # [nt-1, nx, ny, 1]
        alp2_y_next = jnp.pad(alp2_y_next, ((0,0),(0,0),(0,0),(1,0)), mode='constant')  # [nt-1, nx, ny, 2]
        return (alp1_x_next, alp2_x_next, alp1_y_next, alp2_y_next)
  elif ndim == 1: # f = -a(x) * alp, L = |alp|^2/2/c(x), dim_ctrl = dim_state = ndim = 1
    if egno == 1 or egno == 11:  # a(x) = 1
      coeff_fn = lambda x_arr, t_arr: jnp.ones_like(x_arr)  # [..., ndim] -> [..., 1]
    elif egno == 2 or egno == 12:  # a(x) = |x-1| + 0.1
      coeff_fn = lambda x_arr, t_arr: jnp.abs(x_arr - 1.0) + 0.1  # [..., ndim] -> [..., 1]
    elif egno == 3 or egno == 13:  # a(x) = |x-1| - 0.1
      coeff_fn = lambda x_arr, t_arr: jnp.abs(x_arr - 1.0) - 0.1  # [..., ndim] -> [..., 1]
    elif egno == 4 or egno == 14:  # a(x) = |x-1| - 0.5
      coeff_fn = lambda x_arr, t_arr: jnp.abs(x_arr - 1.0) - 0.5  # [..., ndim] -> [..., 1]
    elif egno == 5 or egno == 15:  # a(x) = |x-1| - 1.0
      coeff_fn = lambda x_arr, t_arr: jnp.abs(x_arr - 1.0) - 1.0  # [..., ndim] -> [..., 1]
    elif egno == 6 or egno == 16:  # a(x) = |x-1|^2 + 0.1
      coeff_fn = lambda x_arr, t_arr: (x_arr - 1.0)**2 + 0.1  # [..., ndim] -> [..., 1]
    if egno < 10:  # c(x) = 1
      c_fn = lambda x_arr, t_arr: jnp.ones_like(x_arr)  # [..., ndim] -> [..., 1]
    else:  # c(x) = a(x) ** 2
      c_fn = lambda x_arr, t_arr: coeff_fn(x_arr, t_arr) ** 2 + 1e-6
    H_plus_fn = lambda p, x_arr, t_arr: (jnp.maximum(p,0) * coeff_fn(x_arr, t_arr)[...,0]) **2/2 * c_fn(x_arr, t_arr)[...,0]
    H_minus_fn = lambda p, x_arr, t_arr: (jnp.minimum(p,0) * coeff_fn(x_arr, t_arr)[...,0]) **2/2 * c_fn(x_arr, t_arr)[...,0]
    f_fn = lambda alp, x_arr, t_arr: -alp * coeff_fn(x_arr, t_arr)  # [..., dim_ctrl] -> [..., dim_state] or [..., 1] -> [..., 1]
    def alp_update_fn(alp_prev, Dx_right_phi, Dx_left_phi, rho, sigma, x_arr, t_arr):
      alp1_prev, alp2_prev = alp_prev  # [nt-1, nx, 1]
      eps = 1e-4
      param_inv = (rho + eps) / sigma
      param_inv = param_inv[...,None]  # [nt-1, nx, 1]
      Dx_right_phi = Dx_right_phi[...,None]  # [nt-1, nx, 1]
      Dx_left_phi = Dx_left_phi[...,None]  # [nt-1, nx, 1]
      alp1_next = (Dx_right_phi * coeff_fn(x_arr, t_arr) + param_inv * alp1_prev) / (1/c_fn(x_arr, t_arr) + param_inv)
      alp1_next = (alp1_next * (f_fn(alp1_next, x_arr, t_arr) >= 0.0))
      alp2_next = (Dx_left_phi * coeff_fn(x_arr, t_arr) + param_inv * alp2_prev) / (1/c_fn(x_arr, t_arr) + param_inv)
      alp2_next = (alp2_next * (f_fn(alp2_next, x_arr, t_arr) < 0.0))
      return (alp1_next, alp2_next)
  else:
    raise ValueError("egno {} not implemented".format(egno))
  
  numerical_L_fn, DL_fn, HL_fn = set_up_numerical_L(egno, ndim, numerical_L_ind)

  if ndim == 1:
    Functions = namedtuple('Functions', ['f_fn', 'numerical_L_fn', 'alp_update_fn',
                                        'H_plus_fn', 'H_minus_fn'])
    fns_dict = Functions(H_plus_fn=H_plus_fn, H_minus_fn=H_minus_fn,
                        f_fn=f_fn, numerical_L_fn=numerical_L_fn,
                        alp_update_fn = alp_update_fn)
  elif ndim == 2:
    Functions = namedtuple('Functions', ['f_fn', 'Df1_fn', 'Df2_fn', 'Hf1_fn', 'Hf2_fn',
                                         'numerical_L_fn', 'DL_fn', 'HL_fn',
                                         'alp_update_fn', 'Hx_plus_fn', 'Hx_minus_fn', 'Hy_plus_fn', 'Hy_minus_fn'])
    fns_dict = Functions(f_fn = f_fn, Df1_fn = Df1_fn, Df2_fn = Df2_fn, Hf1_fn = Hf1_fn, Hf2_fn = Hf2_fn,
                         numerical_L_fn = numerical_L_fn, DL_fn = DL_fn, HL_fn = HL_fn,
                         alp_update_fn = alp_update_fn,
                         Hx_plus_fn=H_plus_fn, Hx_minus_fn=H_minus_fn, Hy_plus_fn=H_plus_fn, Hy_minus_fn=H_minus_fn)
  else:
    raise ValueError("ndim {} not implemented".format(ndim))
  return fns_dict
