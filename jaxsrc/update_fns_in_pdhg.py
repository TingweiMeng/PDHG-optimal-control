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

def get_f_vals_1d(f_fn, alp, x_arr, t_arr):
  ''' @ parameters:
      f_fn: a function taking in alp, x_arr, t_arr, returning [nt-1, nx, ndim]
      alp: tuple of alp1, alp2, each term is [nt-1, nx, n_ctrl]
      x_arr: vec that can be broadcasted to [nt-1, nx]
      t_arr: vec that can be broadcasted to [nt-1, nx]
    @ return:
      f_val: tuple of f1, f2, each term is [nt-1, nx]
  '''
  alp1, alp2 = alp  # [nt-1, nx, n_ctrl]
  f1 = f_fn(alp1, x_arr, t_arr)[...,0]  # [nt-1, nx]
  f1 = f1 * (f1 >= 0.0)  # [nt-1, nx]
  f2 = f_fn(alp2, x_arr, t_arr)[...,0]  # [nt-1, nx]
  f2 = f2 * (f2 < 0.0)  # [nt-1, nx]
  return f1, f2

def get_f_vals_2d(f_fn, alp, x_arr, t_arr):
  ''' @ parameters:
      f_fn: a function taking in alp, x_arr, t_arr, returning [nt-1, nx, ny, ndim]
      alp: tuple of alp1_x, alp2_x, alp1_y, alp2_y, each term is [nt-1, nx, ny, n_ctrl]
      x_arr: vec that can be broadcasted to [nt-1, nx, ny]
      t_arr: vec that can be broadcasted to [nt-1, nx, ny]
    @ return:
      f_val: tuple of f1_x, f2_x, f1_y, f2_y, each term is [nt-1, nx, ny]
  '''
  alp1_x, alp2_x, alp1_y, alp2_y = alp  # [nt-1, nx, ny, n_ctrl]
  f1_x = f_fn(alp1_x, x_arr, t_arr)[...,0]  # [nt-1, nx, ny]
  f1_x = f1_x * (f1_x >= 0.0)  # [nt-1, nx, ny]
  f2_x = f_fn(alp2_x, x_arr, t_arr)[...,0]  # [nt-1, nx, ny]
  f2_x = f2_x * (f2_x < 0.0)  # [nt-1, nx, ny]
  f1_y = f_fn(alp1_y, x_arr, t_arr)[...,1]  # [nt-1, nx, ny]
  f1_y = f1_y * (f1_y >= 0.0)  # [nt-1, nx, ny]
  f2_y = f_fn(alp2_y, x_arr, t_arr)[...,1]  # [nt-1, nx, ny]
  f2_y = f2_y * (f2_y < 0.0)  # [nt-1, nx, ny]
  return f1_x, f2_x, f1_y, f2_y

def compute_HJ_residual_1d(phi, alp, dt, dspatial, fns_dict, epsl, x_arr, t_arr, bc):
  dx = dspatial[0]
  L_val = fns_dict.numerical_L_fn(alp, x_arr, t_arr)
  f1, f2 = get_f_vals_1d(fns_dict.f_fn, alp, x_arr, t_arr)
  vec = Dt_decreasedim(phi, dt) - epsl * Dxx_decreasedim(phi, dx, bc)  # [nt-1, nx]
  vec -= Dx_right_decreasedim(phi, dx, bc) * f1 + Dx_left_decreasedim(phi, dx, bc) * f2
  vec -= L_val
  return vec

def compute_HJ_residual_2d(phi, alp, dt, dspatial, fns_dict, epsl, x_arr, t_arr, bc):
  dx, dy = dspatial
  bc_x, bc_y = bc
  L_val = fns_dict.numerical_L_fn(alp, x_arr, t_arr)  # [nt-1, nx, ny]
  Dx_right_phi = Dx_right_decreasedim(phi, dx, bc_x)  # [nt-1, nx, ny]
  Dx_left_phi = Dx_left_decreasedim(phi, dx, bc_x)  # [nt-1, nx, ny]
  Dy_right_phi = Dy_right_decreasedim(phi, dy, bc_y)  # [nt-1, nx, ny]
  Dy_left_phi = Dy_left_decreasedim(phi, dy, bc_y)  # [nt-1, nx, ny]
  f1_x, f2_x, f1_y, f2_y = get_f_vals_2d(fns_dict.f_fn, alp, x_arr, t_arr)
  vec = Dt_decreasedim(phi, dt) - epsl * Dxx_decreasedim(phi, dx, bc_x) - epsl * Dyy_decreasedim(phi, dy, bc_y)  # [nt-1, nx, ny]
  vec -= Dx_right_phi * f1_x + Dx_left_phi * f2_x + Dy_right_phi * f1_y + Dy_left_phi * f2_y
  vec -= L_val
  return vec

def compute_cont_residual_1d(rho, alp, dt, dspatial, fns_dict, c_on_rho, epsl, x_arr, t_arr, bc):
  dx = dspatial[0]
  eps = 1e-4
  f1, f2 = get_f_vals_1d(fns_dict.f_fn, alp, x_arr, t_arr)
  m1 = (rho + eps) * f1  # [nt-1, nx]
  m2 = (rho + eps) * f2  # [nt-1, nx]
  res = Dt_increasedim(rho,dt) + epsl * Dxx_increasedim(rho,dx, bc) # [nt, nx]
  res -= Dx_left_increasedim(m1, dx, bc) + Dx_right_increasedim(m2, dx, bc)
  res = jnp.concatenate([res[:-1,...], res[-1:,...] + c_on_rho/dt], axis = 0)
  return res

def compute_cont_residual_2d(rho, alp, dt, dspatial, fns_dict, c_on_rho, epsl, x_arr, t_arr, bc):
  dx, dy = dspatial
  bc_x, bc_y = bc
  eps = 1e-4
  f1_x, f2_x, f1_y, f2_y = get_f_vals_2d(fns_dict.f_fn, alp, x_arr, t_arr)
  m1_x = (rho + eps) * f1_x  # [nt-1, nx, ny]
  m2_x = (rho + eps) * f2_x  # [nt-1, nx, ny]
  m1_y = (rho + eps) * f1_y  # [nt-1, nx, ny]
  m2_y = (rho + eps) * f2_y  # [nt-1, nx, ny]
  res = Dt_increasedim(rho,dt) + epsl * Dxx_increasedim(rho,dx,bc_x) + epsl * Dyy_increasedim(rho,dy,bc_y)  # [nt, nx, ny]
  res -= Dx_left_increasedim(m1_x, dx, bc_x) + Dx_right_increasedim(m2_x, dx, bc_x) \
          + Dy_left_increasedim(m1_y, dy, bc_y) + Dy_right_increasedim(m2_y, dy, bc_y)
  res = jnp.concatenate([res[:-1,...], res[-1:,...] + c_on_rho/dt], axis = 0)
  return res


def update_rho_1d(rho_prev, phi, alp, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr, bc):
  vec = compute_HJ_residual_1d(phi, alp, dt, dspatial, fns_dict, epsl, x_arr, t_arr, bc)
  rho_next = rho_prev + sigma * vec
  rho_next = jnp.maximum(rho_next, 0.0)  # [nt-1, nx]
  return rho_next

def update_alp_1d(alp_prev, phi, rho, sigma, dspatial, fns_dict, x_arr, t_arr, bc, eps=1e-4):
  dx = dspatial[0]
  Dx_right_phi = Dx_right_decreasedim(phi, dx, bc)  # [nt-1, nx]
  Dx_left_phi = Dx_left_decreasedim(phi, dx, bc)  # [nt-1, nx]
  if 'alp_update_fn' in fns_dict._fields:
    alp_next = fns_dict.alp_update_fn(alp_prev, Dx_right_phi, Dx_left_phi, rho, sigma, x_arr, t_arr)
  else:
    raise NotImplementedError
  return alp_next

def update_rho_2d(rho_prev, phi, alp, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr, bc):
  vec = compute_HJ_residual_2d(phi, alp, dt, dspatial, fns_dict, epsl, x_arr, t_arr, bc)
  rho_next = rho_prev + sigma * vec
  rho_next = jnp.maximum(rho_next, 0.0)  # [nt-1, nx, ny]
  return rho_next

def update_alp_2d(alp_prev, phi, rho, sigma, dspatial, fns_dict, x_arr, t_arr, bc, eps=1e-4):
  dx, dy = dspatial
  bc_x, bc_y = bc
  Dx_right_phi = Dx_right_decreasedim(phi, dx, bc_x)  # [nt-1, nx, ny]
  Dx_left_phi = Dx_left_decreasedim(phi, dx, bc_x)  # [nt-1, nx, ny]
  Dy_right_phi = Dy_right_decreasedim(phi, dy, bc_y)  # [nt-1, nx, ny]
  Dy_left_phi = Dy_left_decreasedim(phi, dy, bc_y)  # [nt-1, nx, ny]
  Dphi = (Dx_right_phi, Dx_left_phi, Dy_right_phi, Dy_left_phi)
  if 'alp_update_fn' in fns_dict._fields:
    alp_next = fns_dict.alp_update_fn(alp_prev, Dphi, rho, sigma, x_arr, t_arr)
  else:
    raise NotImplementedError
  return alp_next

@partial(jax.jit, static_argnames=("fns_dict", "Ct", "bc"))
def update_primal_1d(phi_prev, rho_prev, c_on_rho, alp_prev, tau, dt, dspatial, fns_dict, fv, epsl, x_arr, t_arr, bc,
                     C = 1.0, pow = 1, Ct = 1):
  delta_phi = compute_cont_residual_1d(rho_prev, alp_prev, dt, dspatial, fns_dict, c_on_rho, epsl, x_arr, t_arr, bc)
  phi_next = phi_prev + tau * H1_precond_1d(delta_phi, fv, dt, bc, C = C, pow = pow, Ct = Ct)
  return phi_next

@partial(jax.jit, static_argnames=("fns_dict", "bc"))
def update_primal_2d(phi_prev, rho_prev, c_on_rho, alp_prev, tau, dt, dspatial, fns_dict, fv, epsl, x_arr, t_arr, bc, 
                     C = 1.0, pow = 1, Ct = 1):
  delta_phi = compute_cont_residual_2d(rho_prev, alp_prev, dt, dspatial, fns_dict, c_on_rho, epsl, x_arr, t_arr, bc)
  phi_next = phi_prev + tau * H1_precond_2d(delta_phi, fv, dt, bc, C = C)  # NOTE: pow and Ct are not implemented
  return phi_next


@partial(jax.jit, static_argnames=("fns_dict", "ndim", "bc"))
def update_dual_oneiter(phi_bar, rho_prev, c_on_rho, alp_prev, sigma, dt, dspatial, epsl, x_arr, t_arr, bc, fns_dict, ndim):
  if ndim == 1:
    update_alp = update_alp_1d
    update_rho = update_rho_1d
  elif ndim == 2:
    update_alp = update_alp_2d
    update_rho = update_rho_2d
  else:
    raise NotImplementedError
  alp_next = update_alp(alp_prev, phi_bar, rho_prev, sigma, dspatial, fns_dict, x_arr, t_arr, bc)
  rho_next = update_rho(rho_prev, phi_bar, alp_next, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr, bc)
  err = jnp.sum((rho_next - rho_prev) ** 2) / jnp.sum(rho_next ** 2)  # scalar
  for alp_p, alp_n in zip(alp_prev, alp_next):
    err += jnp.sum((alp_n - alp_p) ** 2) / jnp.sum(alp_n ** 2)  # scalar
  return rho_next, alp_next, err

def update_dual_alternative(phi_bar, rho_prev, c_on_rho, alp_prev, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr, ndim, bc,
                   rho_alp_iters=10, eps=1e-7):
  '''
  @ parameters:
  fns_dict: dict of functions, see the function set_up_example_fns in set_fns.py
  '''
  for j in range(rho_alp_iters):
    rho_next, alp_next, err = update_dual_oneiter(phi_bar, rho_prev, c_on_rho, alp_prev, sigma, dt, dspatial, epsl,
                                                              x_arr, t_arr, bc, fns_dict, ndim)
    if err < eps:
      break
    rho_prev = rho_next
    alp_prev = alp_next
  return rho_next, alp_next