import jax
import jax.numpy as jnp
from functools import partial
import os
from utils.utils_precond import H1_precond_1d, H1_precond_2d
from utils.utils_diff_op import Dx_right_decreasedim, Dx_left_decreasedim, Dxx_decreasedim, Dt_decreasedim, \
  Dx_left_increasedim, Dx_right_increasedim, Dxx_increasedim, Dt_increasedim, Dy_left_decreasedim, \
  Dy_right_decreasedim, Dy_left_increasedim, Dy_right_increasedim

jax.config.update("jax_enable_x64", True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


def compute_HJ_residual_1d(phi, alp, dt, dspatial, fns_dict, epsl, x_arr, t_arr):
  alp1, alp2 = alp
  dx = dspatial[0]
  L_val = fns_dict.L1_fn(alp1, x_arr, t_arr) + fns_dict.L2_fn(alp2, x_arr, t_arr)
  f_plus = fns_dict.f_plus_fn(alp1, x_arr, t_arr)
  f_minus = fns_dict.f_minus_fn(alp2, x_arr, t_arr)
  vec = Dx_right_decreasedim(phi, dx) * f_plus + Dx_left_decreasedim(phi, dx) * f_minus
  vec = - vec + Dt_decreasedim(phi, dt) - epsl * Dxx_decreasedim(phi, dx)  # [nt-1, nx]
  vec = vec - L_val
  return vec

def compute_HJ_residual_2d(phi, alp, dt, dspatial, fns_dict, epsl, x_arr, t_arr):
  alp_11, alp_12, alp_21, alp_22 = alp
  dx, dy = dspatial
  ind_11 = fns_dict.indicator_11_fn(alp_11, x_arr, t_arr)
  ind_12 = fns_dict.indicator_12_fn(alp_12, x_arr, t_arr)
  ind_21 = fns_dict.indicator_21_fn(alp_21, x_arr, t_arr)
  ind_22 = fns_dict.indicator_22_fn(alp_22, x_arr, t_arr)
  L_val = fns_dict.L_fn(alp_11, x_arr, t_arr) * ind_11 + fns_dict.L_fn(alp_12, x_arr, t_arr) * ind_12 \
          + fns_dict.L_fn(alp_21, x_arr, t_arr) * ind_21 + fns_dict.L_fn(alp_22, x_arr, t_arr) * ind_22  # [nt-1, nx, ny]
  Dx_right_phi = Dx_right_decreasedim(phi, dx)  # [nt-1, nx, ny]
  Dx_left_phi = Dx_left_decreasedim(phi, dx)  # [nt-1, nx, ny]
  Dy_right_phi = Dy_right_decreasedim(phi, dy)  # [nt-1, nx, ny]
  Dy_left_phi = Dy_left_decreasedim(phi, dy)  # [nt-1, nx, ny]
  # in Dphi, 1 for right, 2 for left
  D11_phi = jnp.stack([Dx_right_phi, Dy_right_phi], axis = -1) # [nt-1, nx, ny, 2]
  D12_phi = jnp.stack([Dx_right_phi, Dy_left_phi], axis = -1)
  D21_phi = jnp.stack([Dx_left_phi, Dy_right_phi], axis = -1)
  D22_phi = jnp.stack([Dx_left_phi, Dy_left_phi], axis = -1)
  f11 = fns_dict.f_fn(alp_11, x_arr, t_arr) * ind_11[...,None]  # [nt-1, nx, ny, 2]
  f12 = fns_dict.f_fn(alp_12, x_arr, t_arr) * ind_12[...,None]
  f21 = fns_dict.f_fn(alp_21, x_arr, t_arr) * ind_21[...,None]
  f22 = fns_dict.f_fn(alp_22, x_arr, t_arr) * ind_22[...,None]
  vec = Dt_decreasedim(phi, dt) - epsl * Dxx_decreasedim(phi, dx)  # [nt-1, nx, ny]
  vec -= jnp.sum(D11_phi * f11 + D12_phi * f12 + D21_phi * f21 + D22_phi * f22, axis = -1)
  vec = vec - L_val
  return vec

def compute_cont_residual_1d(rho, alp, dt, dspatial, fns_dict, c_on_rho, epsl, x_arr, t_arr):
  alp1, alp2 = alp
  dx = dspatial[0]
  eps = 1e-4
  f_plus = fns_dict.f_plus_fn(alp1, x_arr, t_arr)
  f_minus = fns_dict.f_minus_fn(alp2, x_arr, t_arr)
  m1 = (rho + eps) * f_plus  # [nt-1, nx]
  m2 = (rho + eps) * f_minus  # [nt-1, nx]
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
  f11 = fns_dict.f_fn(alp_11, x_arr, t_arr) * ind_11[...,None]  # [nt-1, nx, ny, 2]
  f12 = fns_dict.f_fn(alp_12, x_arr, t_arr) * ind_12[...,None]
  f21 = fns_dict.f_fn(alp_21, x_arr, t_arr) * ind_21[...,None]
  f22 = fns_dict.f_fn(alp_22, x_arr, t_arr) * ind_22[...,None]
  Dx_left_coeff = f11[...,0] + f12[...,0]  # [nt-1, nx, ny], velocity for Dx_left(v*rho)
  Dx_right_coeff = f21[...,0] + f22[...,0]
  Dy_left_coeff = f11[...,1] + f21[...,1]
  Dy_right_coeff = f12[...,1] + f22[...,1]
  delta_phi = Dt_increasedim(rho,dt) + epsl * Dxx_increasedim(rho,dx) # [nt, nx, ny]
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
  rho_next = jnp.maximum(rho_next, 0.0)  # [nt-1, nx]
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
  if 'alp_update_fn' in fns_dict._fields:
    alp_next = fns_dict.alp_update_fn(alp_prev, Dphi, rho, sigma, x_arr, t_arr)
  else:
    raise NotImplementedError
  return alp_next

@partial(jax.jit, static_argnames=("fns_dict",))
def update_primal_1d(phi_prev, rho_prev, c_on_rho, alp_prev, tau, dt, dspatial, fns_dict, fv, epsl, x_arr, t_arr):
  delta_phi = compute_cont_residual_1d(rho_prev, alp_prev, dt, dspatial, fns_dict, c_on_rho, epsl, x_arr, t_arr)
  C = 1.0
  pow = 0.5
  phi_next = phi_prev + tau * H1_precond_1d(delta_phi, fv, dt, C = C, pow = pow)
  return phi_next

@partial(jax.jit, static_argnames=("fns_dict",))
def update_primal_2d(phi_prev, rho_prev, c_on_rho, alp_prev, tau, dt, dspatial, fns_dict, fv, epsl, x_arr, t_arr):
  delta_phi = compute_cont_residual_2d(rho_prev, alp_prev, dt, dspatial, fns_dict, c_on_rho, epsl, x_arr, t_arr)
  C = 1.0
  phi_next = phi_prev + tau * H1_precond_2d(delta_phi, fv, dt, C = C)
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
  