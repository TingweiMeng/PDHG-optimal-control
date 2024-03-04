import jax.numpy as jnp
from functools import partial
from einshape import jax_einshape as einshape
import jax
from collections import namedtuple

jax.config.update("jax_enable_x64", True)


def set_up_J(egno, ndim, period_spatial):
  if egno != 3:  # sin alp1 x + sin alp2 y
    if ndim == 1:
      x_period = period_spatial[0]
      alpha = 2 * jnp.pi / x_period
    elif ndim == 2:
      x_period, y_period = period_spatial
      alpha = jnp.array([2 * jnp.pi / x_period, 2 * jnp.pi / y_period])
    else:
      raise ValueError("ndim {} not implemented".format(ndim))
    J = lambda x: jnp.sum(jnp.sin(alpha * x), axis = -1)  # [...,ndim] -> [...]
  else:  # newton: J(x,y) = (sin alp y) * exp(- |x|^2 / 2)
    x_period, y_period = period_spatial
    J = lambda x: jnp.sin(2 * jnp.pi / y_period * x[...,1]) * jnp.exp(- x[...,0]**2 / 2)
  return J

def set_up_numerical_L(egno, n_ctrl, ind, fn_coeff_H):
  # numerical L is a function taking alp, x, t as input, where alp is a tuple containing alp1, alp2 in 1d and alp11, alp12, alp21, alp22 in 2d
  # if egno == 1 or 3, L is a quadratic fn with coeff 1/fn_coeff_H / 2
  # if egno == 2, L is the indicator function of the set {alp: |alp| <= fn_coeff_H}
  # fn_coeff_H is from x_arr, t_arr to [..., n_ctrl]
  if egno != 2:
    L_fn_1d = lambda alp, x_arr, t_arr: alp[...,0]**2 / fn_coeff_H(x_arr, t_arr)[...,0] / 2  # [..., 1] -> [...]
    L_fn_2d = lambda alp, x_arr, t_arr: jnp.sum(alp**2 / fn_coeff_H(x_arr, t_arr), axis = -1) / 2  # [..., 2] -> [...]
  else:
    L_fn_1d = lambda alp, x_arr, t_arr: 0.0 * alp[...,0]  # [..., 1] -> [...]
    L_fn_2d = lambda alp, x_arr, t_arr: 0.0 * alp[...,0]  # [..., 2] -> [...]
  if n_ctrl == 1:
    if ind == 0:
      L_fn = lambda alp, x_arr, t_arr: L_fn_1d(alp[0], x_arr, t_arr) + L_fn_1d(alp[1], x_arr, t_arr)
    else:
      raise ValueError("ind {} not implemented".format(ind))
  elif n_ctrl == 2: # each component in alp is [..., n_ctrl]
    if ind == 0:
      L_fn = lambda alp, x_arr, t_arr: L_fn_2d(alp[0], x_arr, t_arr) + L_fn_2d(alp[1], x_arr, t_arr) + L_fn_2d(alp[2], x_arr, t_arr) + L_fn_2d(alp[3], x_arr, t_arr)
    else:
      raise ValueError("ind {} not implemented".format(ind))
  else:
    raise ValueError("n_ctrl {} not implemented".format(n_ctrl))
  return L_fn
  

def set_up_example_fns(egno, ndim, numerical_L_ind):
  '''
  @ parameters:
    egno: int
    ndim: int
    numerical_L_ind: int, we only implemented 0
  @ return:
    fns_dict: named tuple of functions
  '''
  print('egno: ', egno, flush=True)
  if egno == 1:  # L = |alp|^2/2/cH(x,t)
    def alp_update_base_fn(alp_prev, Dphi, param_inv, coeff_f_neg, coeff_H):
      # solves min_alp param_inv * |alp - alp_prev|^2/2 - <coeff_f_neg * alp, Dphi> + L(alp)
      # assume L is a quad fn with diag coeff 1/fn_coeff_H / 2
      # f is a linear fn with coeff -coeff_f_neg (corres to either x or y component of f)
      ''' @ parameters:
            alp_prev: [nt-1, nx, ny, 2] or [nt-1, nx, 1]
            Dphi: [nt-1, nx, ny] or [nt-1, nx]
            param_inv: [nt-1, nx, ny, 1] or [nt-1, nx, 1]
            coeff_f_neg: [nt-1, nx, ny, 2] or [nt-1, nx, 1]
            coeff_H: [nt-1, nx, ny, 2] or [nt-1, nx, 1]
          @ return:
            alp_next: [nt-1, nx, ny, 2] or [nt-1, nx, 1]
      '''
      alp_next = (Dphi[...,None] * coeff_f_neg + param_inv * alp_prev) / (1/coeff_H + param_inv)
      return alp_next
  elif egno == 2:  # L = ind_{|alp| <= coeff_H}
    def alp_update_base_fn(alp_prev, Dphi, param_inv, coeff_f_neg, coeff_H):
      # solves min_alp param_inv * |alp - alp_prev|^2/2 - <coeff_f_neg * alp, Dphi> + L(alp)
      # assume L is an indicator function of the set {alp: |alp| <= coeff_H}
      # f is a linear fn with coeff -coeff_f_neg (corres to either x or y component of f)
      ''' @ parameters:
            alp_prev: [nt-1, nx, ny, 2] or [nt-1, nx, 1]
            Dphi: [nt-1, nx, ny] or [nt-1, nx]
            param_inv: [nt-1, nx, ny, 1] or [nt-1, nx, 1]
            coeff_f_neg: [nt-1, nx, ny, 2] or [nt-1, nx, 1]
            coeff_H: [nt-1, nx, ny, 2] or [nt-1, nx, 1]
          @ return:
            alp_next: [nt-1, nx, ny, 2] or [nt-1, nx, 1]
      '''
      alp_next = Dphi[...,None] * coeff_f_neg / param_inv + alp_prev  # [nt-1, nx, ny, 2] or [nt-1, nx, 1]
      # project to the set {alp: |alp| <= coeff_H}
      alp_next = jnp.minimum(coeff_H, jnp.maximum(-coeff_H, alp_next))
      return alp_next
  if egno == 3:  # newton: ndim=2, n_ctrl=1, x=(vel, pos), f = [a,x_1], L = |alp|^2/2/c(x), c(x) = 1
    n_ctrl = 1
    f_fn = lambda alp, x_arr, t_arr: jnp.concatenate([alp, x_arr[...,0:1]], axis = -1)  # [..., 1] -> [..., 2]
    coeff_fn_H = lambda x_arr, t_arr: jnp.ones_like(x_arr[...,0:1])  # [..., 2], t -> [..., 1]
    numerical_L_fn = set_up_numerical_L(egno, n_ctrl, numerical_L_ind, coeff_fn_H)
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
  elif ndim == 2:  # f_i = -cf_i(x,t)^T alp, dim_ctrl = dim_state = ndim
    # if egno == 1 or 3, L = |alp|^2/2/cH(x,t), cH(x,t) = 1
    # if egno == 2, L(alp) = ind_{|alp| <= cH(x,t)} (pointwise <=), cH(x,t) = 1
    n_ctrl = ndim
    # f1 = ((x-1)^2 + 0.1)*alp1, f2 = ((y-1)^2 + 0.1)*alp2
    coeff_fn_f1_neg = lambda x_arr, t_arr: jnp.concatenate([(x_arr[...,0:1] - 1.0)**2 + 0.1, jnp.zeros_like(x_arr[...,0:1])], axis = -1)
    coeff_fn_f2_neg = lambda x_arr, t_arr: jnp.concatenate([jnp.zeros_like(x_arr[...,0:1]), (x_arr[...,1:2] - 1.0)**2 + 0.1], axis = -1)
    coeff_fn_H = lambda x_arr, t_arr: jnp.ones_like(x_arr)  # [..., ndim] -> [..., n_ctrl]  (n_ctrl = ndim)
    f_fn = lambda alp, x_arr, t_arr: - jnp.concatenate([jnp.sum(coeff_fn_f1_neg(x_arr, t_arr) * alp, axis = -1, keepdims = True), 
                                                        jnp.sum(coeff_fn_f2_neg(x_arr, t_arr) * alp, axis = -1, keepdims = True)], axis = -1)  # [..., 2] -> [..., 2]
    numerical_L_fn = set_up_numerical_L(egno, n_ctrl, numerical_L_ind, coeff_fn_H)
    def alp_update_fn(alp_prev, Dphi, rho, sigma, x_arr, t_arr):  # Dphi is a tuple including four components
      alp1_x_prev, alp2_x_prev, alp1_y_prev, alp2_y_prev = alp_prev  # [nt-1, nx, ny, 2]
      Dx_right_phi, Dx_left_phi, Dy_right_phi, Dy_left_phi = Dphi  # [nt-1, nx, ny]
      eps = 1e-4
      param_inv = (rho[...,None] + eps) / sigma  # [nt-1, nx, ny, 1]
      coeff_f1_neg = coeff_fn_f1_neg(x_arr, t_arr)  # [nt-1, nx, ny, 2]
      coeff_f2_neg = coeff_fn_f2_neg(x_arr, t_arr)  # [nt-1, nx, ny, 2]
      coeff_H = coeff_fn_H(x_arr, t_arr)  # [nt-1, nx, ny, 2]
      alp1_x_next = alp_update_base_fn(alp1_x_prev, Dx_right_phi, param_inv, coeff_f1_neg, coeff_H)  # [nt-1, nx, ny, 2]
      alp1_x_next *= (f_fn(alp1_x_next, x_arr, t_arr)[...,0:1] >= 0.0)  # [nt-1, nx, ny, 2]
      alp2_x_next = alp_update_base_fn(alp2_x_prev, Dx_left_phi, param_inv, coeff_f1_neg, coeff_H)
      alp2_x_next *= (f_fn(alp2_x_next, x_arr, t_arr)[...,0:1] < 0.0)  # [nt-1, nx, ny, 2]
      alp1_y_next = alp_update_base_fn(alp1_y_prev, Dy_right_phi, param_inv, coeff_f2_neg, coeff_H)
      alp1_y_next *= (f_fn(alp1_y_next, x_arr, t_arr)[...,1:2] >= 0.0)  # [nt-1, nx, ny, 2]
      alp2_y_next = alp_update_base_fn(alp2_y_prev, Dy_left_phi, param_inv, coeff_f2_neg, coeff_H)
      alp2_y_next *= (f_fn(alp2_y_next, x_arr, t_arr)[...,1:2] < 0.0)  # [nt-1, nx, ny, 2]
      return (alp1_x_next, alp2_x_next, alp1_y_next, alp2_y_next)
  elif ndim == 1: # f = -a(x) * alp, dim_ctrl = dim_state = ndim = 1
    # if egno == 1 or 3, L = |alp|^2/2/c(x), c(x) = 1
    # if egno == 2, L = ind_{|alp| <= c(x)}, c(x) = 1
    n_ctrl = ndim
    # a(x) = |x-1|^2 + 0.1
    coeff_fn_f_neg = lambda x_arr, t_arr: (x_arr - 1.0)**2 + 0.1  # [..., 1] -> [..., 1]
    coeff_fn_H = lambda x_arr, t_arr: jnp.ones_like(x_arr)  # [..., 1] -> [..., 1]
    f_fn = lambda alp, x_arr, t_arr: -alp * coeff_fn_f_neg(x_arr, t_arr)  # [..., 1] -> [..., 1]
    numerical_L_fn = set_up_numerical_L(egno, n_ctrl, numerical_L_ind, coeff_fn_H)
    def alp_update_fn(alp_prev, Dx_right_phi, Dx_left_phi, rho, sigma, x_arr, t_arr):
      alp1_prev, alp2_prev = alp_prev  # [nt-1, nx, 1]
      eps = 1e-4
      param_inv = (rho + eps) / sigma
      param_inv = param_inv[...,None]  # [nt-1, nx, 1]
      c_f_neg = coeff_fn_f_neg(x_arr, t_arr)  # [nt-1, nx, 1]
      c_H = coeff_fn_H(x_arr, t_arr)  # [nt-1, nx, 1]
      alp1_next = alp_update_base_fn(alp1_prev, Dx_right_phi, param_inv, c_f_neg, c_H)  # [nt-1, nx, 1]
      alp1_next = (alp1_next * (f_fn(alp1_next, x_arr, t_arr) >= 0.0))
      alp2_next = alp_update_base_fn(alp2_prev, Dx_left_phi, param_inv, c_f_neg, c_H)  # [nt-1, nx, 1]
      alp2_next = (alp2_next * (f_fn(alp2_next, x_arr, t_arr) < 0.0))
      return (alp1_next, alp2_next)
  else:
    raise ValueError("egno {} not implemented".format(egno))
  
  Functions = namedtuple('Functions', ['f_fn', 'numerical_L_fn', 'alp_update_fn'])
  fns_dict = Functions(f_fn=f_fn, numerical_L_fn=numerical_L_fn, alp_update_fn = alp_update_fn)
  return fns_dict
