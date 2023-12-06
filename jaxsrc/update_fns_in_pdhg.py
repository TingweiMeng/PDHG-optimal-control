import jax
import jax.numpy as jnp
from functools import partial
import os
from utils.utils_precond import H1_precond_1d, H1_precond_2d
from utils.utils_diff_op import Dx_right_decreasedim, Dx_left_decreasedim, Dxx_decreasedim, Dt_decreasedim, \
  Dx_left_increasedim, Dx_right_increasedim, Dxx_increasedim, Dt_increasedim, Dy_left_decreasedim, \
  Dy_right_decreasedim, Dy_left_increasedim, Dy_right_increasedim, Dyy_increasedim, Dyy_decreasedim

jax.config.update("jax_enable_x64", True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

def get_f_vals_1d(fns_dict, alp, x_arr, t_arr):
  ''' @ parameters:
      fns_dict: named tuple of functions, containing f_fn
      alp: tuple of alp1, alp2, each term is [nt-1, nx, 1]
      x_arr: vec that can be broadcasted to [nt-1, nx]
      t_arr: vec that can be broadcasted to [nt-1, nx]
    @ return:
      f_val: tuple of f1, f2, each term is [nt-1, nx]
  '''
  alp1, alp2 = alp  # [nt-1, nx, 1]
  f1 = fns_dict.f_fn(alp1, x_arr, t_arr)[...,0]  # [nt-1, nx]
  f1 = f1 * (f1 >= 0.0)  # [nt-1, nx]
  f2 = fns_dict.f_fn(alp2, x_arr, t_arr)[...,0]  # [nt-1, nx]
  f2 = f2 * (f2 < 0.0)  # [nt-1, nx]
  return f1, f2

def get_f_vals_2d(fns_dict, alp, x_arr, t_arr):
  ''' @ parameters:
      fns_dict: named tuple of functions, containing f_fn
      alp: tuple of alp1_x, alp2_x, alp1_y, alp2_y, each term is [nt-1, nx, ny, 1]
      x_arr: vec that can be broadcasted to [nt-1, nx, ny]
      t_arr: vec that can be broadcasted to [nt-1, nx, ny]
    @ return:
      f_val: tuple of f11, f12, f21, f22, each term is [nt-1, nx, ny]
  '''
  alp1_x, alp2_x, alp1_y, alp2_y = alp  # [nt-1, nx, ny, 1]
  f1_x = fns_dict.f_fn(alp1_x, x_arr, t_arr)[...,0]  # [nt-1, nx, ny]
  f1_x = f1_x * (f1_x >= 0.0)  # [nt-1, nx, ny]
  f2_x = fns_dict.f_fn(alp2_x, x_arr, t_arr)[...,0]  # [nt-1, nx, ny]
  f2_x = f2_x * (f2_x < 0.0)  # [nt-1, nx, ny]
  f1_y = fns_dict.f_fn(alp1_y, x_arr, t_arr)[...,0]  # [nt-1, nx, ny]
  f1_y = f1_y * (f1_y >= 0.0)  # [nt-1, nx, ny]
  f2_y = fns_dict.f_fn(alp2_y, x_arr, t_arr)[...,0]  # [nt-1, nx, ny]
  f2_y = f2_y * (f2_y < 0.0)  # [nt-1, nx, ny]
  return f1_x, f2_x, f1_y, f2_y

def compute_HJ_residual_1d(phi, alp, dt, dspatial, fns_dict, epsl, x_arr, t_arr):
  dx = dspatial[0]
  L_val = fns_dict.numerical_L_fn(alp, x_arr, t_arr)
  f1, f2 = get_f_vals_1d(fns_dict, alp, x_arr, t_arr)
  vec = Dt_decreasedim(phi, dt) - epsl * Dxx_decreasedim(phi, dx)  # [nt-1, nx]
  vec -= Dx_right_decreasedim(phi, dx) * f1 + Dx_left_decreasedim(phi, dx) * f2
  vec -= L_val
  return vec

def compute_HJ_residual_2d(phi, alp, dt, dspatial, fns_dict, epsl, x_arr, t_arr):
  # alp_11, alp_12, alp_21, alp_22 = alp
  # dx, dy = dspatial
  # ind_11 = fns_dict.indicator_11_fn(alp_11, x_arr, t_arr)
  # ind_12 = fns_dict.indicator_12_fn(alp_12, x_arr, t_arr)
  # ind_21 = fns_dict.indicator_21_fn(alp_21, x_arr, t_arr)
  # ind_22 = fns_dict.indicator_22_fn(alp_22, x_arr, t_arr)
  # L_val = fns_dict.L_fn(alp_11, x_arr, t_arr) * ind_11[...,0] + fns_dict.L_fn(alp_12, x_arr, t_arr) * ind_12[...,0] \
  #         + fns_dict.L_fn(alp_21, x_arr, t_arr) * ind_21[...,0] + fns_dict.L_fn(alp_22, x_arr, t_arr) * ind_22[...,0]  # [nt-1, nx, ny]
  # Dx_right_phi = Dx_right_decreasedim(phi, dx)  # [nt-1, nx, ny]
  # Dx_left_phi = Dx_left_decreasedim(phi, dx)  # [nt-1, nx, ny]
  # Dy_right_phi = Dy_right_decreasedim(phi, dy)  # [nt-1, nx, ny]
  # Dy_left_phi = Dy_left_decreasedim(phi, dy)  # [nt-1, nx, ny]
  # # in Dphi, 1 for right, 2 for left
  # D11_phi = jnp.stack([Dx_right_phi, Dy_right_phi], axis = -1) # [nt-1, nx, ny, 2]
  # D12_phi = jnp.stack([Dx_right_phi, Dy_left_phi], axis = -1)
  # D21_phi = jnp.stack([Dx_left_phi, Dy_right_phi], axis = -1)
  # D22_phi = jnp.stack([Dx_left_phi, Dy_left_phi], axis = -1)
  # f11 = fns_dict.f_fn(alp_11, x_arr, t_arr) * ind_11  # [nt-1, nx, ny, 2]
  # f12 = fns_dict.f_fn(alp_12, x_arr, t_arr) * ind_12
  # f21 = fns_dict.f_fn(alp_21, x_arr, t_arr) * ind_21
  # f22 = fns_dict.f_fn(alp_22, x_arr, t_arr) * ind_22
  # vec = Dt_decreasedim(phi, dt) - epsl * Dxx_decreasedim(phi, dx) - epsl * Dyy_decreasedim(phi, dy)  # [nt-1, nx, ny]
  # TODO: this is for debug only !!!!!!!!!!!!!!!!!!
  # vx_neg = alp_11[...,0] + alp_12[...,0]
  # vx_pos = alp_21[...,0] + alp_22[...,0]
  # vy_neg = alp_11[...,1] + alp_21[...,1]
  # vy_pos = alp_12[...,1] + alp_22[...,1]
  # vec -= Dx_right_decreasedim(phi, dx) * vx_neg + Dx_left_decreasedim(phi, dx) * vx_pos
  # vec -= Dy_right_decreasedim(phi, dy) * vy_neg + Dy_left_decreasedim(phi, dy) * vy_pos
  vxp, vxm, vyp, vym = alp
  dx, dy = dspatial[0], dspatial[1]
  # Hstar_plus_fn = lambda p, x_arr, t_arr: jnp.maximum(p, 0.0) **2/ 2
  # Hstar_minus_fn = lambda p, x_arr, t_arr: jnp.minimum(p, 0.0) **2/ 2
  # Hstar_val = Hstar_plus_fn(vxm, x_arr, t_arr) + Hstar_minus_fn(vxp, x_arr, t_arr) \
  #             + Hstar_plus_fn(vym, x_arr, t_arr) + Hstar_minus_fn(vyp, x_arr, t_arr)
  vec = Dx_right_decreasedim(phi, dx) * vxp + Dx_left_decreasedim(phi, dx) * vxm
  vec = vec + Dy_right_decreasedim(phi, dy) * vyp + Dy_left_decreasedim(phi, dy) * vym
  vec = vec + Dt_decreasedim(phi,dt) - epsl * Dxx_decreasedim(phi, dx) - epsl * Dyy_decreasedim(phi, dy)  # [nt-1, nx, ny]
  # vec = vec - Hstar_val
  # vec -= jnp.sum(D11_phi * f11 + D12_phi * f12 + D21_phi * f21 + D22_phi * f22, axis = -1)
  # vec = vec - L_val
  vec -= vxp **2 /2 + vxm **2 /2 + vyp **2 /2 + vym **2 /2
  return vec

def compute_cont_residual_1d(rho, alp, dt, dspatial, fns_dict, c_on_rho, epsl, x_arr, t_arr):
  dx = dspatial[0]
  eps = 1e-4
  f1, f2 = get_f_vals_1d(fns_dict, alp, x_arr, t_arr)
  m1 = (rho + eps) * f1  # [nt-1, nx]
  m2 = (rho + eps) * f2  # [nt-1, nx]
  delta_phi = -Dx_left_increasedim(m1, dx) + -Dx_right_increasedim(m2, dx) \
              + Dt_increasedim(rho,dt) + epsl * Dxx_increasedim(rho,dx) # [nt, nx]
  delta_phi = jnp.concatenate([delta_phi[:-1,...], delta_phi[-1:,...] + c_on_rho/dt], axis = 0)
  return delta_phi

def compute_cont_residual_2d(rho, alp, dt, dspatial, fns_dict, c_on_rho, epsl, x_arr, t_arr):
  alp_11, alp_12, alp_21, alp_22 = alp
  dx, dy = dspatial
  eps = 1e-4
  ind_11 = fns_dict.indicator_11_fn(alp_11, x_arr, t_arr)
  ind_12 = fns_dict.indicator_12_fn(alp_12, x_arr, t_arr)
  ind_21 = fns_dict.indicator_21_fn(alp_21, x_arr, t_arr)
  ind_22 = fns_dict.indicator_22_fn(alp_22, x_arr, t_arr)
  f11 = fns_dict.f_fn(alp_11, x_arr, t_arr) * ind_11  # [nt-1, nx, ny, 2]
  f12 = fns_dict.f_fn(alp_12, x_arr, t_arr) * ind_12
  f21 = fns_dict.f_fn(alp_21, x_arr, t_arr) * ind_21
  f22 = fns_dict.f_fn(alp_22, x_arr, t_arr) * ind_22
  Dx_left_coeff = f11[...,0] + f12[...,0]  # [nt-1, nx, ny], velocity for Dx_left(v*rho)
  Dx_right_coeff = f21[...,0] + f22[...,0]
  Dy_left_coeff = f11[...,1] + f21[...,1]
  Dy_right_coeff = f12[...,1] + f22[...,1]
  delta_phi = Dt_increasedim(rho,dt) + epsl * Dxx_increasedim(rho,dx) + epsl * Dyy_increasedim(rho,dy)  # [nt, nx, ny]
  delta_phi -= Dx_left_increasedim(Dx_left_coeff * (rho + eps), dx) + Dx_right_increasedim(Dx_right_coeff * (rho + eps), dx) \
              + Dy_left_increasedim(Dy_left_coeff * (rho + eps), dy) + Dy_right_increasedim(Dy_right_coeff * (rho + eps), dy)
  delta_phi = jnp.concatenate([delta_phi[:-1,...], delta_phi[-1:,...] + c_on_rho/dt], axis = 0)
  return delta_phi


def update_rho_1d(rho_prev, phi, alp, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr):
  vec = compute_HJ_residual_1d(phi, alp, dt, dspatial, fns_dict, epsl, x_arr, t_arr)
  rho_next = rho_prev + sigma * vec
  rho_next = jnp.maximum(rho_next, 0.0)  # [nt-1, nx]
  return rho_next

def update_alp_1d(alp_prev, phi, rho, sigma, dspatial, fns_dict, x_arr, t_arr, eps=1e-4):
  dx = dspatial[0]
  Dx_right_phi = Dx_right_decreasedim(phi, dx)  # [nt-1, nx]
  Dx_left_phi = Dx_left_decreasedim(phi, dx)  # [nt-1, nx]
  if 'alp_update_fn' in fns_dict._fields:
    alp_next = fns_dict.alp_update_fn(alp_prev, Dx_right_phi, Dx_left_phi, rho, sigma, x_arr, t_arr)
  else:
    raise NotImplementedError
  return alp_next

def update_rho_2d(rho_prev, phi, alp, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr):
  vec = compute_HJ_residual_2d(phi, alp, dt, dspatial, fns_dict, epsl, x_arr, t_arr)
  rho_next = rho_prev + sigma * vec
  rho_next = jnp.maximum(rho_next, 0.0)  # [nt-1, nx, ny]
  return rho_next

def update_alp_2d(alp_prev, phi, rho, sigma, dspatial, fns_dict, x_arr, t_arr, eps=1e-4):
  dx, dy = dspatial
  Dx_right_phi = Dx_right_decreasedim(phi, dx)  # [nt-1, nx, ny]
  Dx_left_phi = Dx_left_decreasedim(phi, dx)  # [nt-1, nx, ny]
  Dy_right_phi = Dy_right_decreasedim(phi, dy)  # [nt-1, nx, ny]
  Dy_left_phi = Dy_left_decreasedim(phi, dy)  # [nt-1, nx, ny]
  # in Dphi, 1 for right, 2 for left
  D11_phi = jnp.stack([Dx_right_phi, Dy_right_phi], axis = -1) # [nt-1, nx, ny, 2]
  D12_phi = jnp.stack([Dx_right_phi, Dy_left_phi], axis = -1)
  D21_phi = jnp.stack([Dx_left_phi, Dy_right_phi], axis = -1)
  D22_phi = jnp.stack([Dx_left_phi, Dy_left_phi], axis = -1)
  Dphi = (D11_phi, D12_phi, D21_phi, D22_phi)
  # # TODO: for debug only !!!!!!!
  # alp_11, alp_12, alp_21, alp_22 = alp_prev
  # vx_neg = alp_11[...,0] + alp_12[...,0]
  # vx_pos = alp_21[...,0] + alp_22[...,0]
  # vy_neg = alp_11[...,1] + alp_21[...,1]
  # vy_pos = alp_12[...,1] + alp_22[...,1]
  # param = sigma / (rho + eps)
  # vxp_next_raw = vx_neg + param * Dx_right_decreasedim(phi, dx)  # [nt-1, nx, ny]
  # vxm_next_raw = vx_pos + param * Dx_left_decreasedim(phi, dx)  # [nt-1, nx, ny]
  # vyp_next_raw = vy_neg + param * Dy_right_decreasedim(phi, dy)  # [nt-1, nx, ny]
  # vym_next_raw = vy_pos + param * Dy_left_decreasedim(phi, dy)  # [nt-1, nx, ny]
  # vxp_next = Hxstar_minus_prox_fn(vxp_next_raw, param, x_arr, t_arr)  # [nt-1, nx, ny]
  # vxm_next = Hxstar_plus_prox_fn(vxm_next_raw, param, x_arr, t_arr)  # [nt-1, nx, ny]  
  # vyp_next = Hystar_minus_prox_fn(vyp_next_raw, param, x_arr, t_arr)  # [nt-1, nx, ny]
  # vym_next = Hystar_plus_prox_fn(vym_next_raw, param, x_arr, t_arr)  # [nt-1, nx, ny]  
  # if 'alp_update_fn' in fns_dict._fields:
  #   alp_next = fns_dict.alp_update_fn(alp_prev, Dphi, rho, sigma, x_arr, t_arr)
  # else:
  #   raise NotImplementedError
  # alp11_prev, alp12_prev, alp21_prev, alp22_prev = alp_prev  # [nt-1, nx, ny, 2]
  # D11_phi, D12_phi, D21_phi, D22_phi = Dphi  # [nt-1, nx, ny, 2]
  # eps = 1e-4
  # param_inv = (rho[...,None] + eps) / sigma  # [nt-1, nx, ny, 1]
  # alp11_next = (param_inv * alp11_prev + D11_phi) / (1 + param_inv)
  # alp11_next *= indicator_11_fn(alp11_next, x_arr, t_arr)  # [nt-1, nx, ny, 2]
  # alp12_next = (param_inv * alp12_prev + D12_phi) / (1 + param_inv)
  # alp12_next *= indicator_12_fn(alp12_next, x_arr, t_arr)
  # alp21_next = (param_inv * alp21_prev + D21_phi) / (1 + param_inv)
  # alp21_next *= indicator_21_fn(alp21_next, x_arr, t_arr)
  # alp22_next = (param_inv * alp22_prev + D22_phi) / (1 + param_inv)
  # alp22_next *= indicator_22_fn(alp22_next, x_arr, t_arr)
  # return (alp11_next, alp12_next, alp21_next, alp22_next)

  vxp_prev, vxm_prev, vyp_prev, vym_prev = alp_prev
  dx, dy = dspatial[0], dspatial[1]
  Hstar_plus_prox_fn = lambda p, param, x_arr, t_arr: jnp.maximum(p / (1+ param), 0.0)
  Hstar_minus_prox_fn = lambda p, param, x_arr, t_arr: jnp.minimum(p / (1+ param), 0.0)
  Hxstar_plus_prox_fn = Hstar_plus_prox_fn
  Hxstar_minus_prox_fn = Hstar_minus_prox_fn
  Hystar_plus_prox_fn = Hstar_plus_prox_fn
  Hystar_minus_prox_fn = Hstar_minus_prox_fn
  param = sigma / (rho + eps)
  vxp_next_raw = vxp_prev + param * Dx_right_decreasedim(phi, dx)  # [nt-1, nx, ny]
  vxm_next_raw = vxm_prev + param * Dx_left_decreasedim(phi, dx)  # [nt-1, nx, ny]
  vyp_next_raw = vyp_prev + param * Dy_right_decreasedim(phi, dy)  # [nt-1, nx, ny]
  vym_next_raw = vym_prev + param * Dy_left_decreasedim(phi, dy)  # [nt-1, nx, ny]
  indvxp = (vxp_next_raw < 0.0)
  indvxm = (vxm_next_raw >= 0.0)
  indvyp = (vyp_next_raw < 0.0)
  indvym = (vym_next_raw >= 0.0)
  vxp_next = Hxstar_minus_prox_fn(vxp_next_raw, param, x_arr, t_arr)  # [nt-1, nx, ny]
  vxm_next = Hxstar_plus_prox_fn(vxm_next_raw, param, x_arr, t_arr)  # [nt-1, nx, ny]  
  vyp_next = Hystar_minus_prox_fn(vyp_next_raw, param, x_arr, t_arr)  # [nt-1, nx, ny]
  vym_next = Hystar_plus_prox_fn(vym_next_raw, param, x_arr, t_arr)  # [nt-1, nx, ny]  
  vxp_next = vxp_next * (indvyp + indvym)  # [nt-1, nx, ny]
  vxm_next = vxm_next * (indvyp + indvym)
  vyp_next = vyp_next * (indvxp + indvxm)
  vym_next = vym_next * (indvxp + indvxm)
  return (vxp_next, vxm_next, vyp_next, vym_next)
  return alp_next

@partial(jax.jit, static_argnames=("fns_dict", "Ct"))
def update_primal_1d(phi_prev, rho_prev, c_on_rho, alp_prev, tau, dt, dspatial, fns_dict, fv, epsl, x_arr, t_arr, 
                     C = 1.0, pow = 1, Ct = 1):
  delta_phi = compute_cont_residual_1d(rho_prev, alp_prev, dt, dspatial, fns_dict, c_on_rho, epsl, x_arr, t_arr)
  phi_next = phi_prev + tau * H1_precond_1d(delta_phi, fv, dt, C = C, pow = pow, Ct = Ct)
  return phi_next

@partial(jax.jit, static_argnames=("fns_dict",))
def update_primal_2d(phi_prev, rho_prev, c_on_rho, alp_prev, tau, dt, dspatial, fns_dict, fv, epsl, x_arr, t_arr):
  # delta_phi = compute_cont_residual_2d(rho_prev, alp_prev, dt, dspatial, fns_dict, c_on_rho, epsl, x_arr, t_arr)
  C = 1.0
  # # phi_next = phi_prev + tau * H1_precond_2d(delta_phi, fv, dt, C = C)
  # # TODO: for debug only!!!!!!!!
  # eps = 1e-4
  # alp_11, alp_12, alp_21, alp_22 = alp_prev
  # vx_neg = alp_11[...,0] + alp_12[...,0]
  # vx_pos = alp_21[...,0] + alp_22[...,0]
  # vy_neg = alp_11[...,1] + alp_21[...,1]
  # vy_pos = alp_12[...,1] + alp_22[...,1]
  # mxp_prev = (rho_prev + eps) * vx_neg  # [nt-1, nx]
  # mxm_prev = (rho_prev + eps) * vx_pos  # [nt-1, nx]
  # myp_prev = (rho_prev + eps) * vy_neg  # [nt-1, nx]
  # mym_prev = (rho_prev + eps) * vy_pos  # [nt-1, nx]
  # dx, dy = dspatial
  # delta_phi = Dx_left_increasedim(mxp_prev, dx) + Dx_right_increasedim(mxm_prev, dx) \
  #             + Dy_left_increasedim(myp_prev, dy) + Dy_right_increasedim(mym_prev, dy) \
  #             + Dt_increasedim(rho_prev,dt) + epsl * Dxx_increasedim(rho_prev,dx) \
  #             + epsl * Dyy_increasedim(rho_prev,dy)  # [nt, nx]
  # delta_phi = jnp.concatenate([delta_phi[:-1,...], delta_phi[-1:,...] + c_on_rho/dt], axis = 0)
  # phi_next = phi_prev + tau * H1_precond_2d(delta_phi, fv, dt, C = C)
  vxp_prev, vxm_prev, vyp_prev, vym_prev = alp_prev
  dx, dy = dspatial
  eps = 1e-4
  mxp_prev = (rho_prev + eps) * vxp_prev  # [nt-1, nx, ny]
  mxm_prev = (rho_prev + eps) * vxm_prev  # [nt-1, nx, ny]
  myp_prev = (rho_prev + eps) * vyp_prev  # [nt-1, nx, ny]
  mym_prev = (rho_prev + eps) * vym_prev  # [nt-1, nx, ny]
  delta_phi = Dx_left_increasedim(mxp_prev, dx) + Dx_right_increasedim(mxm_prev, dx) \
              + Dy_left_increasedim(myp_prev, dy) + Dy_right_increasedim(mym_prev, dy) \
              + Dt_increasedim(rho_prev,dt) + epsl * Dxx_increasedim(rho_prev,dx) \
              + epsl * Dyy_increasedim(rho_prev,dy)  # [nt, nx]
  delta_phi = jnp.concatenate([delta_phi[:-1,...], delta_phi[-1:,...] + c_on_rho/dt], axis = 0)
  phi_next = phi_prev + tau * H1_precond_2d(delta_phi, fv, dt, C = C)
  return phi_next
  return phi_next


@partial(jax.jit, static_argnames=("fns_dict", "ndim"))
def update_dual_oneiter(phi_bar, rho_prev, c_on_rho, alp_prev, sigma, dt, dspatial, epsl, x_arr, t_arr, fns_dict, ndim):
  if ndim == 1:
    update_alp = update_alp_1d
    update_rho = update_rho_1d
  elif ndim == 2:
    update_alp = update_alp_2d
    update_rho = update_rho_2d
  else:
    raise NotImplementedError
  alp_next = update_alp(alp_prev, phi_bar, rho_prev, sigma, dspatial, fns_dict, x_arr, t_arr)
  rho_next = update_rho(rho_prev, phi_bar, alp_next, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr)
  err = jnp.linalg.norm(rho_next - rho_prev) / jnp.maximum(jnp.linalg.norm(rho_prev), 1.0)
  for alp_p, alp_n in zip(alp_prev, alp_next):
    err = jnp.maximum(err, jnp.linalg.norm(alp_n - alp_p) / jnp.maximum(jnp.linalg.norm(alp_p), 1.0))
  return rho_next, alp_next, err


def update_dual(phi_bar, rho_prev, c_on_rho, alp_prev, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr, ndim,
                   rho_alp_iters=10, eps=1e-7):
  '''
  @ parameters:
  fns_dict: dict of functions, see the function set_up_example_fns in set_fns.py
  '''
  for j in range(rho_alp_iters):
    rho_next, alp_next, err = update_dual_oneiter(phi_bar, rho_prev, c_on_rho, alp_prev, sigma, dt, dspatial, epsl,
                                                              x_arr, t_arr, fns_dict, ndim)
    if err < eps:
      break
    rho_prev = rho_next
    alp_prev = alp_next
  return rho_next, alp_next
  