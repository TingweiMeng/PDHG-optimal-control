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
  # if egno < 20 or egno == 30, L is a quadratic fn with coeff 1/fn_coeff_H / 2
  # if 20 <= egno < 30, L is the indicator function of the set {alp: |alp| <= fn_coeff_H}
  # fn_coeff_H is from x_arr, t_arr to [..., n_ctrl]
  if egno < 20 or egno == 30:
    L_fn_1d = lambda alp, x_arr, t_arr: alp[...,0]**2 / fn_coeff_H(x_arr, t_arr)[...,0] / 2  # [..., 1] -> [...]
    DL_fn_1d = lambda alp, x_arr, t_arr: alp / fn_coeff_H(x_arr, t_arr)  # [..., 1] -> [..., 1]
    HL_fn_1d = lambda alp, x_arr, t_arr: 1/fn_coeff_H(x_arr, t_arr)[...,None]  # [..., 1] -> [..., 1, 1]
    # L_fn_2d = lambda alp_x, alp_y, x_arr, t_arr: L_fn_1d(alp_x, x_arr, t_arr) + L_fn_1d(alp_y, x_arr, t_arr)  # [..., 1], [..., 1] -> [...]
    L_fn_2d = lambda alp, x_arr, t_arr: jnp.sum(alp**2 / fn_coeff_H(x_arr, t_arr), axis = -1) / 2  # [..., 2] -> [...]
    DL_fn_2d = lambda alp, x_arr, t_arr: alp / fn_coeff_H(x_arr, t_arr)  # [..., 2] -> [..., 2]
    HL_fn_2d = lambda alp, x_arr, t_arr: jnp.concatenate([1 / fn_coeff_H(x_arr, t_arr)[...,None], 
                1 / fn_coeff_H(x_arr, t_arr)[...,None]], axis = -1) * jnp.eye(2)  # [..., 2] -> [..., 2, 2]
  elif egno >= 20 and egno < 30:
    L_fn_1d = lambda alp, x_arr, t_arr: 0.0 * alp[...,0]  # [..., 1] -> [...]
    DL_fn_1d = lambda alp, x_arr, t_arr: 0.0 * alp
    HL_fn_1d = lambda alp, x_arr, t_arr: 0.0 * alp[...,None]
    L_fn_2d = lambda alp, x_arr, t_arr: 0.0 * alp[...,0]  # [..., 2] -> [...]
    DL_fn_2d = lambda alp, x_arr, t_arr: 0.0 * alp
    HL_fn_2d = lambda alp, x_arr, t_arr: jnp.zeros(alp.shape + alp.shape[-1:])  # [..., 2] -> [..., 2, 2]
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
  if egno < 20:  # L = |alp|^2/2/cH(x,t)
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
  elif egno >= 20 and egno < 30:  # L = ind_{|alp| <= coeff_H}
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
      prod = Dphi[...,None] * coeff_f_neg
      alp_next = prod / param_inv + alp_prev  # [nt-1, nx, ny, 2] or [nt-1, nx, 1]
      # project to the set {alp: |alp| <= coeff_H}
      alp_next = jnp.minimum(coeff_H, jnp.maximum(-coeff_H, alp_next))
      bool_flag = (param_inv > 0)
      prod_sign = 2 * (prod >= 0) - 1
      alp_next = alp_next * bool_flag + prod_sign * coeff_H * (1 - bool_flag)
      return alp_next
  if egno == 30:  # newton: ndim=2, n_ctrl=1, x=(vel, pos), f = [a,x_1], L = |alp|^2/2/c(x), c(x) = 1
    n_ctrl = 1
    f_fn = lambda alp, x_arr, t_arr: jnp.concatenate([alp, x_arr[...,0:1]], axis = -1)  # [..., 1] -> [..., 2]
    coeff_fn_H = lambda x_arr, t_arr: jnp.ones_like(x_arr[...,0:1])  # [..., 2], t -> [..., 1]
    numerical_L_fn, DL_fn, HL_fn = set_up_numerical_L(egno, n_ctrl, numerical_L_ind, coeff_fn_H)
    coeff_fn_f_neg = None
    def alp_update_fn(alp_prev, Dphi, rho, sigma, x_arr, t_arr):  # Dphi is a tuple including four components
      alp1_x_prev, alp2_x_prev, alp1_y_prev, alp2_y_prev = alp_prev  # [nt-1, nx, ny, 1]
      Dx_right_phi, Dx_left_phi, _, _ = Dphi  # [nt-1, nx, ny]
      param_inv = rho[...,None] / sigma  # [nt-1, nx, ny, 1]
      coeff_L = 1 / coeff_fn_H(x_arr, t_arr)  # [nt-1, nx, ny, 2]
      alp1_x_next = (-Dx_right_phi[...,None] + param_inv * alp1_x_prev) / (coeff_L + param_inv)  # [nt-1, nx, ny, 1]
      alp1_x_next *= (f_fn(alp1_x_next, x_arr, t_arr)[...,0:1] >= 0.0)  # [nt-1, nx, ny, 1]
      alp2_x_next = (-Dx_left_phi[...,None] + param_inv * alp2_x_prev) / (coeff_L + param_inv)  # [nt-1, nx, ny, 1]
      alp2_x_next *= (f_fn(alp2_x_next, x_arr, t_arr)[...,0:1] < 0.0)  # [nt-1, nx, ny, 1]
      return (alp1_x_next, alp2_x_next, alp1_y_prev, alp2_y_prev)
  elif ndim == 2:  # f_i = -cf_i(x,t)^T alp, dim_ctrl = dim_state = ndim
    # if egno < 20 or egno == 30, L = |alp|^2/2/cH(x,t), cH(x,t) = 1
    # if 20 <= egno < 30, L(alp) = ind_{|alp| <= cH(x,t)} (pointwise <=), cH(x,t) = 1
    n_ctrl = ndim
    if egno == 1 or egno == 21:  # f1 = -alp1, f2 = -alp2
      coeff_fn_f1_neg = lambda x_arr, t_arr: jnp.concatenate([jnp.ones_like(x_arr[...,0:1]), jnp.zeros_like(x_arr[...,0:1])], axis = -1)  # [..., ndim] -> [..., ndim]
      coeff_fn_f2_neg = lambda x_arr, t_arr: jnp.concatenate([jnp.zeros_like(x_arr[...,0:1]), jnp.ones_like(x_arr[...,0:1])], axis = -1)  # [..., ndim] -> [..., ndim]
    elif egno == 2 or egno == 22:  # f1 = -((x-1)^2 + 0.1)*alp1, f2 = -((y-1)^2 + 0.1)*alp2
      coeff_fn_f1_neg = lambda x_arr, t_arr: jnp.concatenate([(x_arr[...,0:1] - 1.0)**2 + 0.1, jnp.zeros_like(x_arr[...,0:1])], axis = -1)
      coeff_fn_f2_neg = lambda x_arr, t_arr: jnp.concatenate([jnp.zeros_like(x_arr[...,0:1]), (x_arr[...,1:2] - 1.0)**2 + 0.1], axis = -1)
    elif egno == 3 or egno == 23:  # f1 = -((x/2+y/2-1)^2 - 0.1)*alp1, f2 = -((x/2+y/2-1)^2 - 0.1)*alp2
      coeff_fn_f1_neg = lambda x_arr, t_arr: jnp.concatenate([(x_arr[...,0:1]/2 + x_arr[...,1:2]/2 - 1.0)**2 + 0.1, jnp.zeros_like(x_arr[...,0:1])], axis = -1)
      coeff_fn_f2_neg = lambda x_arr, t_arr: jnp.concatenate([jnp.zeros_like(x_arr[...,0:1]), (x_arr[...,0:1]/2 + x_arr[...,1:2]/2 - 1.0)**2 + 0.1], axis = -1)
    # coeff_fn_f_neg = lambda x_arr, t_arr: (coeff_fn_f1_neg(x_arr, t_arr), coeff_fn_f2_neg(x_arr, t_arr))
    coeff_fn_H = lambda x_arr, t_arr: jnp.ones_like(x_arr)  # [..., ndim] -> [..., n_ctrl]  (n_ctrl = ndim)
    f_fn = lambda alp, x_arr, t_arr: - jnp.concatenate([jnp.sum(coeff_fn_f1_neg(x_arr, t_arr) * alp, axis = -1, keepdims = True), 
                                                        jnp.sum(coeff_fn_f2_neg(x_arr, t_arr) * alp, axis = -1, keepdims = True)], axis = -1)  # [..., 2] -> [..., 2]
    Df1_fn = lambda alp, x_arr, t_arr: - coeff_fn_f1_neg(x_arr, t_arr)  # [..., 2] -> [..., 2]
    Df2_fn = lambda alp, x_arr, t_arr: - coeff_fn_f2_neg(x_arr, t_arr)
    Hf1_fn = lambda alp, x_arr, t_arr: jnp.zeros(alp.shape + alp.shape[-1:])  # [..., 2] -> [..., 2, 2]
    Hf2_fn = lambda alp, x_arr, t_arr: jnp.zeros(alp.shape + alp.shape[-1:])
    numerical_L_fn, DL_fn, HL_fn = set_up_numerical_L(egno, n_ctrl, numerical_L_ind, coeff_fn_H)
    def alp_update_fn(alp_prev, Dphi, rho, sigma, x_arr, t_arr):  # Dphi is a tuple including four components
      alp1_x_prev, alp2_x_prev, alp1_y_prev, alp2_y_prev = alp_prev  # [nt-1, nx, ny, 2]
      Dx_right_phi, Dx_left_phi, Dy_right_phi, Dy_left_phi = Dphi  # [nt-1, nx, ny]
      param_inv = rho[...,None] / sigma  # [nt-1, nx, ny, 1]
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
    # if egno < 20 or egno == 30, L = |alp|^2/2/c(x), c(x) = 1
    # if 20 <= egno < 30, L = ind_{|alp| <= c(x)}, c(x) = 1
    n_ctrl = ndim
    if egno == 1 or egno == 11 or egno == 21:  # a(x) = 1
      coeff_fn_f_neg = lambda x_arr, t_arr: jnp.ones_like(x_arr)  # [..., 1] -> [..., 1]
    elif egno == 2 or egno == 12 or egno == 22:  # a(x) = |x-1| + 0.1
      coeff_fn_f_neg = lambda x_arr, t_arr: jnp.abs(x_arr - 1.0) + 0.1  # [..., 1] -> [..., 1]
    elif egno == 3 or egno == 13 or egno == 23:  # a(x) = |x-1| - 0.1
      coeff_fn_f_neg = lambda x_arr, t_arr: jnp.abs(x_arr - 1.0) - 0.1  # [..., 1] -> [..., 1]
    elif egno == 4 or egno == 14 or egno == 24:  # a(x) = |x-1| - 0.5
      coeff_fn_f_neg = lambda x_arr, t_arr: jnp.abs(x_arr - 1.0) - 0.5  # [..., 1] -> [..., 1]
    elif egno == 5 or egno == 15 or egno == 25:  # a(x) = |x-1| - 1.0
      coeff_fn_f_neg = lambda x_arr, t_arr: jnp.abs(x_arr - 1.0) - 1.0  # [..., 1] -> [..., 1]
    elif egno == 6 or egno == 16 or egno == 26:  # a(x) = |x-1|^2 + 0.1
      coeff_fn_f_neg = lambda x_arr, t_arr: (x_arr - 1.0)**2 + 0.1  # [..., 1] -> [..., 1]
    if egno < 10 or egno >= 20:  # c(x) = 1
      coeff_fn_H = lambda x_arr, t_arr: jnp.ones_like(x_arr)  # [..., 1] -> [..., 1]
    else:  # c(x) = a(x) ** 2
      coeff_fn_H = lambda x_arr, t_arr: coeff_fn_f_neg(x_arr, t_arr) ** 2 + 1e-6
    f_fn = lambda alp, x_arr, t_arr: -alp * coeff_fn_f_neg(x_arr, t_arr)  # [..., 1] -> [..., 1]
    Df_fn = lambda alp, x_arr, t_arr: - coeff_fn_f_neg(x_arr, t_arr)  # [..., 1] -> [..., 1]
    Hf_fn = lambda alp, x_arr, t_arr: jnp.zeros(alp.shape + alp.shape[-1:])  # [..., 1] -> [..., 1, 1]
    numerical_L_fn, DL_fn, HL_fn = set_up_numerical_L(egno, n_ctrl, numerical_L_ind, coeff_fn_H)
    def alp_update_fn(alp_prev, Dx_phi, rho, sigma, x_arr, t_arr):
      Dx_right_phi, Dx_left_phi = Dx_phi  # [nt-1, nx]
      alp1_prev, alp2_prev = alp_prev  # [nt-1, nx, 1]
      param_inv = rho[...,None] / sigma # [nt-1, nx, 1]
      # Dx_right_phi = Dx_right_phi[...,None]  # [nt-1, nx, 1]
      # Dx_left_phi = Dx_left_phi[...,None]  # [nt-1, nx, 1]
      c_f_neg = coeff_fn_f_neg(x_arr, t_arr)  # [nt-1, nx, 1]
      c_H = coeff_fn_H(x_arr, t_arr)  # [nt-1, nx, 1]
      alp1_next = alp_update_base_fn(alp1_prev, Dx_right_phi, param_inv, c_f_neg, c_H)  # [nt-1, nx, 1]
      # alp1_next = (Dx_right_phi * c_f_neg + param_inv * alp1_prev) / (c_L + param_inv)
      alp1_next = (alp1_next * (f_fn(alp1_next, x_arr, t_arr) >= 0.0))
      alp2_next = alp_update_base_fn(alp2_prev, Dx_left_phi, param_inv, c_f_neg, c_H)  # [nt-1, nx, 1]
      # alp2_next = (Dx_left_phi * c_f_neg + param_inv * alp2_prev) / (c_L + param_inv)
      alp2_next = (alp2_next * (f_fn(alp2_next, x_arr, t_arr) < 0.0))
      return (alp1_next, alp2_next)
  else:
    raise ValueError("egno {} not implemented".format(egno))
  
  Functions = namedtuple('Functions', ['f_fn', 'numerical_L_fn', 'alp_update_fn'])
  fns_dict = Functions(f_fn=f_fn, numerical_L_fn=numerical_L_fn, alp_update_fn = alp_update_fn)
  return fns_dict
