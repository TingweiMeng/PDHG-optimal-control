import jax.numpy as jnp
from functools import partial
from einshape import jax_einshape as einshape
import jax
from collections import namedtuple

jax.config.update("jax_enable_x64", True)


def set_up_J(egno, ndim, period_spatial):
  if egno > 0 and egno != 30:  # sin alp1 x + sin alp2 y
    if ndim == 1:
      x_period = period_spatial[0]
      alpha = 2 * jnp.pi / x_period
    elif ndim == 2:
      x_period, y_period = period_spatial
      alpha = jnp.array([2 * jnp.pi / x_period, 2 * jnp.pi / y_period])
    else:
      raise ValueError("ndim {} not implemented".format(ndim))
    J = lambda x: jnp.sum(jnp.sin(alpha * x), axis = -1)  # [...,ndim] -> [...]
  elif egno == 30:  # newton: J(x,y) = (sin alp y) * exp(- |x|^2 / 2)
    x_period, y_period = period_spatial
    J = lambda x: jnp.sin(2 * jnp.pi / y_period * x[...,1]) * jnp.exp(- x[...,0]**2 / 2)
  else:
    raise ValueError("egno {} not implemented".format(egno))
  return J

def set_up_numerical_L(egno, n_ctrl, ind, fn_coeff_H):
  # numerical L is a function taking alp, x, t as input, where alp is a tuple containing alp1, alp2 in 1d and alp11, alp12, alp21, alp22 in 2d
  # L is a quadratic fn with coeff fn_coeff_L / 2
  # fn_coeff_L is from x_arr, t_arr to [..., n_ctrl]
  L_fn_1d = lambda alp, x_arr, t_arr: alp[...,0]**2 / fn_coeff_H(x_arr, t_arr)[...,0] / 2  # [..., 1] -> [...]
  DL_fn_1d = lambda alp, x_arr, t_arr: alp / fn_coeff_H(x_arr, t_arr)  # [..., 1] -> [..., 1]
  HL_fn_1d = lambda alp, x_arr, t_arr: 1/fn_coeff_H(x_arr, t_arr)[...,None]  # [..., 1] -> [..., 1, 1]
  # L_fn_2d = lambda alp_x, alp_y, x_arr, t_arr: L_fn_1d(alp_x, x_arr, t_arr) + L_fn_1d(alp_y, x_arr, t_arr)  # [..., 1], [..., 1] -> [...]
  L_fn_2d = lambda alp, x_arr, t_arr: jnp.sum(alp**2 / fn_coeff_H(x_arr, t_arr), axis = -1) / 2  # [..., 2] -> [...]
  DL_fn_2d = lambda alp, x_arr, t_arr: alp / fn_coeff_H(x_arr, t_arr)  # [..., 2] -> [..., 2]
  HL_fn_2d = lambda alp, x_arr, t_arr: jnp.concatenate([1 / fn_coeff_H(x_arr, t_arr)[...,None], 
              1 / fn_coeff_H(x_arr, t_arr)[...,None]], axis = -1) * jnp.eye(2)  # [..., 2] -> [..., 2, 2]
  if n_ctrl == 1:
    if ind == 0:
      L_fn = lambda alp, x_arr, t_arr: L_fn_1d(alp[0], x_arr, t_arr) + L_fn_1d(alp[1], x_arr, t_arr)
      DL_fn = None
      HL_fn = None
    elif ind == 1:
      L_fn = lambda alp, x_arr, t_arr: L_fn_1d(alp[0] + alp[1], x_arr, t_arr)
      DL_fn = lambda alp, x_arr, t_arr: DL_fn_1d(alp[0] + alp[1], x_arr, t_arr)
      HL_fn = lambda alp, x_arr, t_arr: HL_fn_1d(alp[0] + alp[1], x_arr, t_arr)
    else:
      raise ValueError("ind {} not implemented".format(ind))
  elif n_ctrl == 2: # each component in alp is [..., n_ctrl]
    if ind == 0:
      L_fn = lambda alp, x_arr, t_arr: L_fn_2d(alp[0], x_arr, t_arr) + L_fn_2d(alp[1], x_arr, t_arr) + L_fn_2d(alp[2], x_arr, t_arr) + L_fn_2d(alp[3], x_arr, t_arr)
      DL_fn = None
      HL_fn = None
    elif ind == 1:
      L_fn = lambda alp, x_arr, t_arr: L_fn_2d(alp[0] + alp[1] + alp[2] + alp[3], x_arr, t_arr)
      DL_fn = lambda alp, x_arr, t_arr: DL_fn_2d(alp[0] + alp[1] + alp[2] + alp[3], x_arr, t_arr)
      HL_fn = lambda alp, x_arr, t_arr: HL_fn_2d(alp[0] + alp[1] + alp[2] + alp[3], x_arr, t_arr)
    else:
      raise ValueError("ind {} not implemented".format(ind))
  else:
    raise ValueError("n_ctrl {} not implemented".format(n_ctrl))
  return L_fn, DL_fn, HL_fn
  

def set_up_example_fns(egno, ndim, numerical_L_ind):
  '''
  @ parameters:
    egno: int
    ndim: int
  @ return:
    fns_dict: named tuple of functions
  '''
  print('egno: ', egno, flush=True)
  # omit the indicator function
  # note: dim of p is [nt-1, nx]
  # H_plus_fn, H_minus_fn, H_fn are only used in this function and compute_HJ_residual_EO_1d_general, compute_HJ_residual_EO_2d_general, compute_EO_forward_solution_1d_general, compute_EO_forward_solution_2d_general
  if egno == 30:  # newton: ndim=2, n_ctrl=1, x=(vel, pos), f = [a,x_1], L = |alp|^2/2/c(x), c(x) = 1
    n_ctrl = 1
    f_fn = lambda alp, x_arr, t_arr: jnp.concatenate([alp, x_arr[...,0:1]], axis = -1)  # [..., 1] -> [..., 2]
    coeff_fn_H = lambda x_arr, t_arr: jnp.ones_like(x_arr[...,0:1])  # [..., 2], t -> [..., 1]
    numerical_L_fn, DL_fn, HL_fn = set_up_numerical_L(egno, n_ctrl, numerical_L_ind, coeff_fn_H)
    def alp_update_fn(alp_prev, Dphi, rho, sigma, x_arr, t_arr):  # Dphi is a tuple including four components
      alp1_x_prev, alp2_x_prev, alp1_y_prev, alp2_y_prev = alp_prev  # [nt-1, nx, ny, 1]
      Dx_right_phi, Dx_left_phi, _, _ = Dphi  # [nt-1, nx, ny]
      eps = 1e-4
      param_inv = (rho[...,None] + eps) / sigma  # [nt-1, nx, ny, 1]
      coeff_L = 1 / coeff_fn_H(x_arr, t_arr)  # [nt-1, nx, ny, 2]
      alp1_x_next = (-Dx_right_phi[...,None] + param_inv * alp1_x_prev) / (coeff_L + param_inv)  # [nt-1, nx, ny, 1]
      alp1_x_next *= (f_fn(alp1_x_next, x_arr, t_arr)[...,0:1] >= 0.0)  # [nt-1, nx, ny, 1]
      alp2_x_next = (-Dx_left_phi[...,None] + param_inv * alp2_x_prev) / (coeff_L + param_inv)  # [nt-1, nx, ny, 1]
      alp2_x_next *= (f_fn(alp2_x_next, x_arr, t_arr)[...,0:1] < 0.0)  # [nt-1, nx, ny, 1]
      return (alp1_x_next, alp2_x_next, alp1_y_prev, alp2_y_prev)
  elif ndim == 2:  # f_i = -cf_i(x,t)^T alp, dim_ctrl = dim_state = ndim, L = |alp|^2/2/cH(x,t), cH(x,t) = 1
    n_ctrl = ndim
    if egno == 1:
      coeff_fn_f1_neg = lambda x_arr, t_arr: jnp.concatenate([jnp.ones_like(x_arr[...,0:1]), jnp.zeros_like(x_arr[...,0:1])], axis = -1)  # [..., ndim] -> [..., ndim]
      coeff_fn_f2_neg = lambda x_arr, t_arr: jnp.concatenate([jnp.zeros_like(x_arr[...,0:1]), jnp.ones_like(x_arr[...,0:1])], axis = -1)  # [..., ndim] -> [..., ndim]
    elif egno == 2:
      coeff_fn_f1_neg = lambda x_arr, t_arr: jnp.concatenate([(x_arr[...,0:1] - 1.0)**2 + 0.1, jnp.zeros_like(x_arr[...,0:1])], axis = -1)
      coeff_fn_f2_neg = lambda x_arr, t_arr: jnp.concatenate([jnp.zeros_like(x_arr[...,0:1]), (x_arr[...,1:2] - 1.0)**2 + 0.1], axis = -1)
    elif egno == 3:
      coeff_fn_f1_neg = lambda x_arr, t_arr: jnp.concatenate([(x_arr[...,0:1]/2 + x_arr[...,1:2]/2 - 1.0)**2 + 0.1, jnp.zeros_like(x_arr[...,0:1])], axis = -1)
      coeff_fn_f2_neg = lambda x_arr, t_arr: jnp.concatenate([jnp.zeros_like(x_arr[...,0:1]), (x_arr[...,0:1]/2 + x_arr[...,1:2]/2 - 1.0)**2 + 0.1], axis = -1)
    coeff_fn_H = lambda x_arr, t_arr: jnp.ones_like(x_arr)  # [..., ndim] -> [..., n_ctrl]  (n_ctrl = ndim)
    f_fn = lambda alp, x_arr, t_arr: - jnp.concatenate([jnp.sum(coeff_fn_f1_neg(x_arr, t_arr) * alp, axis = -1, keepdims = True), 
                                                        jnp.sum(coeff_fn_f2_neg(x_arr, t_arr) * alp, axis = -1, keepdims = True)], axis = -1)  # [..., 2] -> [..., 2]
    Df1_fn = lambda alp, x_arr, t_arr: - coeff_fn_f1_neg(x_arr, t_arr)  # [..., 2] -> [..., 2]
    Df2_fn = lambda alp, x_arr, t_arr: - coeff_fn_f2_neg(x_arr, t_arr)
    Hf1_fn = lambda alp, x_arr, t_arr: jnp.zeros(alp.shape + alp.shape[-1:])  # [..., 2] -> [..., 2, 2]
    Hf2_fn = lambda alp, x_arr, t_arr: jnp.zeros(alp.shape + alp.shape[-1:])
    numerical_L_fn, DL_fn, HL_fn = set_up_numerical_L(egno, n_ctrl, numerical_L_ind, coeff_fn_H)
    def alp_update_base_fn(alp_prev, Dphi, param_inv, coeff_f_neg, coeff_L):
      # solves min_alp param_inv * |alp - alp_prev|^2/2 + <alp, Dphi> + L(alp)
      # assume L is a quad fn with diag coeff fn_coeff_L / 2
      # f is a linear fn with coeff -coeff_f_neg (corres to either x or y component of f)
      ''' @ parameters:
            alp_prev: [nt-1, nx, ny, 2]
            Dphi: [nt-1, nx, ny]
            param_inv: [nt-1, nx, ny, 1]
            coeff_f_neg: [nt-1, nx, ny, 2]
            coeff_L: [nt-1, nx, ny, 2]
          @ return:
            alp_next: [nt-1, nx, ny, 2]
      '''
      alp_next = (Dphi[...,None] * coeff_f_neg + param_inv * alp_prev) / (coeff_L + param_inv)
      return alp_next
    def alp_update_fn(alp_prev, Dphi, rho, sigma, x_arr, t_arr):  # Dphi is a tuple including four components
      alp1_x_prev, alp2_x_prev, alp1_y_prev, alp2_y_prev = alp_prev  # [nt-1, nx, ny, 2]
      Dx_right_phi, Dx_left_phi, Dy_right_phi, Dy_left_phi = Dphi  # [nt-1, nx, ny]
      eps = 1e-4
      param_inv = (rho[...,None] + eps) / sigma  # [nt-1, nx, ny, 1]
      coeff_f1_neg = coeff_fn_f1_neg(x_arr, t_arr)  # [nt-1, nx, ny, 2]
      coeff_f2_neg = coeff_fn_f2_neg(x_arr, t_arr)  # [nt-1, nx, ny, 2]
      coeff_L = 1 / coeff_fn_H(x_arr, t_arr)  # [nt-1, nx, ny, 2]
      alp1_x_next = alp_update_base_fn(alp1_x_prev, Dx_right_phi, param_inv, coeff_f1_neg, coeff_L)  # [nt-1, nx, ny, 2]
      alp1_x_next *= (f_fn(alp1_x_next, x_arr, t_arr)[...,0:1] >= 0.0)  # [nt-1, nx, ny, 2]
      alp2_x_next = alp_update_base_fn(alp2_x_prev, Dx_left_phi, param_inv, coeff_f1_neg, coeff_L)
      alp2_x_next *= (f_fn(alp2_x_next, x_arr, t_arr)[...,0:1] < 0.0)  # [nt-1, nx, ny, 2]
      alp1_y_next = alp_update_base_fn(alp1_y_prev, Dy_right_phi, param_inv, coeff_f2_neg, coeff_L)
      alp1_y_next *= (f_fn(alp1_y_next, x_arr, t_arr)[...,1:2] >= 0.0)  # [nt-1, nx, ny, 2]
      alp2_y_next = alp_update_base_fn(alp2_y_prev, Dy_left_phi, param_inv, coeff_f2_neg, coeff_L)
      alp2_y_next *= (f_fn(alp2_y_next, x_arr, t_arr)[...,1:2] < 0.0)  # [nt-1, nx, ny, 2]
      return (alp1_x_next, alp2_x_next, alp1_y_next, alp2_y_next)
  elif ndim == 1: # f = -a(x) * alp, L = |alp|^2/2/c(x), dim_ctrl = dim_state = ndim = 1
    n_ctrl = ndim
    if egno == 1 or egno == 11:  # a(x) = 1
      coeff_fn_f_neg = lambda x_arr, t_arr: jnp.ones_like(x_arr)  # [..., 1] -> [..., 1]
    elif egno == 2 or egno == 12:  # a(x) = |x-1| + 0.1
      coeff_fn_f_neg = lambda x_arr, t_arr: jnp.abs(x_arr - 1.0) + 0.1  # [..., 1] -> [..., 1]
    elif egno == 3 or egno == 13:  # a(x) = |x-1| - 0.1
      coeff_fn_f_neg = lambda x_arr, t_arr: jnp.abs(x_arr - 1.0) - 0.1  # [..., 1] -> [..., 1]
    elif egno == 4 or egno == 14:  # a(x) = |x-1| - 0.5
      coeff_fn_f_neg = lambda x_arr, t_arr: jnp.abs(x_arr - 1.0) - 0.5  # [..., 1] -> [..., 1]
    elif egno == 5 or egno == 15:  # a(x) = |x-1| - 1.0
      coeff_fn_f_neg = lambda x_arr, t_arr: jnp.abs(x_arr - 1.0) - 1.0  # [..., 1] -> [..., 1]
    elif egno == 6 or egno == 16:  # a(x) = |x-1|^2 + 0.1
      coeff_fn_f_neg = lambda x_arr, t_arr: (x_arr - 1.0)**2 + 0.1  # [..., 1] -> [..., 1]
    if egno < 10:  # c(x) = 1
      coeff_fn_H = lambda x_arr, t_arr: jnp.ones_like(x_arr)  # [..., 1] -> [..., 1]
    else:  # c(x) = a(x) ** 2
      coeff_fn_H = lambda x_arr, t_arr: coeff_fn_f_neg(x_arr, t_arr) ** 2 + 1e-6
    f_fn = lambda alp, x_arr, t_arr: -alp * coeff_fn_f_neg(x_arr, t_arr)  # [..., 1] -> [..., 1]
    Df_fn = lambda alp, x_arr, t_arr: - coeff_fn_f_neg(x_arr, t_arr)  # [..., 1] -> [..., 1]
    Hf_fn = lambda alp, x_arr, t_arr: jnp.zeros(alp.shape + alp.shape[-1:])  # [..., 1] -> [..., 1, 1]
    numerical_L_fn, DL_fn, HL_fn = set_up_numerical_L(egno, n_ctrl, numerical_L_ind, coeff_fn_H)
    def alp_update_fn(alp_prev, Dx_right_phi, Dx_left_phi, rho, sigma, x_arr, t_arr):
      alp1_prev, alp2_prev = alp_prev  # [nt-1, nx, 1]
      eps = 1e-4
      param_inv = (rho + eps) / sigma
      param_inv = param_inv[...,None]  # [nt-1, nx, 1]
      Dx_right_phi = Dx_right_phi[...,None]  # [nt-1, nx, 1]
      Dx_left_phi = Dx_left_phi[...,None]  # [nt-1, nx, 1]
      c_f_neg = coeff_fn_f_neg(x_arr, t_arr)  # [nt-1, nx, 1]
      c_L = 1 / coeff_fn_H(x_arr, t_arr)  # [nt-1, nx, 1]
      # def fn_compute_obj_1(alp):
      #   return param_inv * (alp - alp1_prev)**2/2 + jnp.maximum(-c_f_neg * alp, 0) * Dx_right_phi + numerical_L_fn((alp, alp2_prev), x_arr, t_arr)
      # def fn_compute_obj_2(alp):
      #   return param_inv * (alp - alp2_prev)**2/2 + jnp.minimum(-c_f_neg * alp, 0) * Dx_left_phi + numerical_L_fn((alp1_prev, alp), x_arr, t_arr)
      alp1_next = (Dx_right_phi * c_f_neg + param_inv * alp1_prev) / (c_L + param_inv)
      # alp1_next = (Dx_right_phi * c_f_neg + param_inv * alp1_prev - alp2_prev * c_L) / (c_L + param_inv)
      # alp1_next_2 = 0 * alp1_next_1  # the boundary point for the piece of f
      # alp1_next_3 = (param_inv * alp1_prev - alp2_prev/c_fn(x_arr, t_arr)) / (1/c_fn(x_arr, t_arr) + param_inv)
      # alp1_val1 = fn_compute_obj_1(alp1_next_1)
      # alp1_val2 = fn_compute_obj_1(alp1_next_2)
      # alp1_val3 = fn_compute_obj_1(alp1_next_3)
      # alp1_next = jnp.where(alp1_val1 < alp1_val2, alp1_next_1, alp1_next_2)
      # alp1_next = jnp.where(alp1_val3 < jnp.minimum(alp1_val1, alp1_val2), alp1_next_3, alp1_next)
      alp1_next = (alp1_next * (f_fn(alp1_next, x_arr, t_arr) >= 0.0))
      alp2_next = (Dx_left_phi * c_f_neg + param_inv * alp2_prev) / (c_L + param_inv)
      # alp2_next = (Dx_left_phi * c_f_neg + param_inv * alp2_prev  - alp1_next * c_L) / (c_L + param_inv)
      # alp2_next2 = 0 * alp2_next1  # the boundary point for the piece of f
      # alp2_next3 = (param_inv * alp2_prev - alp1_prev/c_fn(x_arr, t_arr)) / (1/c_fn(x_arr, t_arr) + param_inv)
      # alp2_val1 = fn_compute_obj_2(alp2_next1)
      # alp2_val2 = fn_compute_obj_2(alp2_next2)
      # alp2_val3 = fn_compute_obj_2(alp2_next3)
      # alp2_next = jnp.where(alp2_val1 < alp2_val2, alp2_next1, alp2_next2)
      # alp2_next = jnp.where(alp2_val3 < jnp.minimum(alp2_val1, alp2_val2), alp2_next3, alp2_next)
      alp2_next = (alp2_next * (f_fn(alp2_next, x_arr, t_arr) < 0.0))
      return (alp1_next, alp2_next)
  else:
    raise ValueError("egno {} not implemented".format(egno))
  
  Functions = namedtuple('Functions', ['f_fn', 'numerical_L_fn', 'alp_update_fn'])
  fns_dict = Functions(f_fn=f_fn, numerical_L_fn=numerical_L_fn, alp_update_fn = alp_update_fn)
  return fns_dict
