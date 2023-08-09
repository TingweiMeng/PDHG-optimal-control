import jax
import jax.numpy as jnp
from functools import partial
import os
import solver
from pdhg_solver import Dx_left_decreasedim, Dx_right_decreasedim, Dx_left_increasedim, Dx_right_increasedim
from pdhg_solver import Dy_left_decreasedim, Dy_right_decreasedim, Dy_left_increasedim, Dy_right_increasedim
from pdhg_solver import Dt_decreasedim, Dt_increasedim, Dxx_decreasedim, Dxx_increasedim
from pdhg_solver import Dyy_decreasedim, Dyy_increasedim

jax.config.update("jax_enable_x64", True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

def update_rho_1d(rho_prev, phi, v, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr):
  vp, vm = v[0], v[1]
  dx = dspatial[0]
  Hstar_plus_fn = fns_dict.Hstar_plus_fn
  Hstar_minus_fn = fns_dict.Hstar_minus_fn
  vec = Dx_right_decreasedim(phi, dx) * vp + Dx_left_decreasedim(phi, dx) * vm
  vec = vec + Dt_decreasedim(phi,dt) - epsl * Dxx_decreasedim(phi, dx)  # [nt-1, nx]
  vec = vec - Hstar_plus_fn(vm, x_arr, t_arr) - Hstar_minus_fn(vp, x_arr, t_arr)
  rho_next = rho_prev + sigma * vec
  rho_next = jnp.maximum(rho_next, 0.0)  # [nt-1, nx]
  return rho_next

def update_v_1d(v_prev, phi, rho, sigma, dspatial, fns_dict, x_arr, t_arr, eps=1e-4):
  '''
  @ parameters:
    Hstar_plus_prox_fn and Hstar_minus_prox_fn are prox point operator taking (x,t) as input
    and output argmin_u H(u) + |x-u|^2/(2t)
  '''
  vp_prev, vm_prev = v_prev[0], v_prev[1]
  dx = dspatial[0]
  Hstar_plus_prox_fn = fns_dict.Hstar_plus_prox_fn
  Hstar_minus_prox_fn = fns_dict.Hstar_minus_prox_fn
  param = sigma / (rho + eps)
  vp_next_raw = vp_prev + param * Dx_right_decreasedim(phi, dx)  # [nt-1, nx]
  vm_next_raw = vm_prev + param * Dx_left_decreasedim(phi, dx)  # [nt-1, nx]
  vp_next = Hstar_minus_prox_fn(vp_next_raw, param, x_arr, t_arr)  # [nt-1, nx]
  vm_next = Hstar_plus_prox_fn(vm_next_raw, param, x_arr, t_arr)  # [nt-1, nx]  
  return (vp_next, vm_next)

def update_rho_2d(rho_prev, phi, v, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr):
  vxp, vxm = v[0], v[1]
  vyp, vym = v[2], v[3]
  dx, dy = dspatial[0], dspatial[1]
  Hxstar_plus_fn = fns_dict.Hxstar_plus_fn
  Hxstar_minus_fn = fns_dict.Hxstar_minus_fn
  Hystar_plus_fn = fns_dict.Hystar_plus_fn
  Hystar_minus_fn = fns_dict.Hystar_minus_fn
  vec = Dx_right_decreasedim(phi, dx) * vxp + Dx_left_decreasedim(phi, dx) * vxm
  vec = vec + Dy_right_decreasedim(phi, dy) * vyp + Dy_left_decreasedim(phi, dy) * vym
  vec = vec + Dt_decreasedim(phi,dt) - epsl * Dxx_decreasedim(phi, dx) - epsl * Dyy_decreasedim(phi, dy)  # [nt-1, nx, ny]
  vec = vec - Hxstar_plus_fn(vxm, x_arr, t_arr) - Hxstar_minus_fn(vxp, x_arr, t_arr)
  vec = vec - Hystar_plus_fn(vym, x_arr, t_arr) - Hystar_minus_fn(vyp, x_arr, t_arr)
  rho_next = rho_prev + sigma * vec
  rho_next = jnp.maximum(rho_next, 0.0)  # [nt-1, nx, ny]
  return rho_next

def update_v_2d(v_prev, phi, rho, sigma, dspatial, fns_dict, x_arr, t_arr, eps=1e-4):
  '''
  @ parameters:
    Hstar_plus_prox_fn and Hstar_minus_prox_fn are prox point operator taking (x,t) as input
    and output argmin_u H(u) + |x-u|^2/(2t)
  '''
  vxp_prev, vxm_prev, vyp_prev, vym_prev = v_prev[0], v_prev[1], v_prev[2], v_prev[3]
  dx, dy = dspatial[0], dspatial[1]
  Hxstar_plus_prox_fn = fns_dict.Hxstar_plus_prox_fn
  Hxstar_minus_prox_fn = fns_dict.Hxstar_minus_prox_fn
  Hystar_plus_prox_fn = fns_dict.Hystar_plus_prox_fn
  Hystar_minus_prox_fn = fns_dict.Hystar_minus_prox_fn
  param = sigma / (rho + eps)
  vxp_next_raw = vxp_prev + param * Dx_right_decreasedim(phi, dx)  # [nt-1, nx]
  vxm_next_raw = vxm_prev + param * Dx_left_decreasedim(phi, dx)  # [nt-1, nx]
  vyp_next_raw = vyp_prev + param * Dy_right_decreasedim(phi, dy)  # [nt-1, nx]
  vym_next_raw = vym_prev + param * Dy_left_decreasedim(phi, dy)  # [nt-1, nx]
  vxp_next = Hxstar_minus_prox_fn(vxp_next_raw, param, x_arr, t_arr)  # [nt-1, nx]
  vxm_next = Hxstar_plus_prox_fn(vxm_next_raw, param, x_arr, t_arr)  # [nt-1, nx]  
  vyp_next = Hystar_minus_prox_fn(vyp_next_raw, param, x_arr, t_arr)  # [nt-1, nx]
  vym_next = Hystar_plus_prox_fn(vym_next_raw, param, x_arr, t_arr)  # [nt-1, nx]  
  return (vxp_next, vxm_next, vyp_next, vym_next)




@jax.jit
def update_primal_1d(phi_prev, rho_prev, c_on_rho, v_prev, tau, dt, dspatial, fv, epsl):
  vp_prev, vm_prev = v_prev[0], v_prev[1]
  dx = dspatial[0]
  eps = 1e-4
  mp_prev = (rho_prev + eps) * vp_prev  # [nt-1, nx]
  mm_prev = (rho_prev + eps) * vm_prev  # [nt-1, nx]
  delta_phi = Dx_left_increasedim(mp_prev, dx) + Dx_right_increasedim(mm_prev, dx) \
              + Dt_increasedim(rho_prev,dt) + epsl * Dxx_increasedim(rho_prev,dx) # [nt, nx]
  delta_phi = jnp.concatenate([delta_phi[:-1,...], delta_phi[-1:,...] + c_on_rho/dt], axis = 0)
  phi_next = phi_prev - tau * solver.Poisson_eqt_solver(delta_phi, fv, dt)
  return phi_next

@jax.jit
def update_primal_2d(phi_prev, rho_prev, c_on_rho, v_prev, tau, dt, dspatial, fv, epsl):
  vxp_prev, vxm_prev, vyp_prev, vym_prev = v_prev[0], v_prev[1], v_prev[2], v_prev[3]
  dx, dy = dspatial[0], dspatial[1]
  eps = 1e-4
  mxp_prev = (rho_prev + eps) * vxp_prev  # [nt-1, nx]
  mxm_prev = (rho_prev + eps) * vxm_prev  # [nt-1, nx]
  myp_prev = (rho_prev + eps) * vyp_prev  # [nt-1, nx]
  mym_prev = (rho_prev + eps) * vym_prev  # [nt-1, nx]
  delta_phi = Dx_left_increasedim(mxp_prev, dx) + Dx_right_increasedim(mxm_prev, dx) \
              + Dy_left_increasedim(myp_prev, dy) + Dy_right_increasedim(mym_prev, dy) \
              + Dt_increasedim(rho_prev,dt) + epsl * Dxx_increasedim(rho_prev,dx) \
              + epsl * Dyy_increasedim(rho_prev,dy)  # [nt, nx]
  delta_phi = jnp.concatenate([delta_phi[:-1,...], delta_phi[-1:,...] + c_on_rho/dt], axis = 0)
  phi_next = phi_prev - tau * solver.Poisson_eqt_solver_2d(delta_phi, fv, dt)
  return phi_next


@partial(jax.jit, static_argnames=("fns_dict", "ndim"))
def update_dual_oneiter(phi_bar, rho_prev, c_on_rho, v_prev, sigma, dt, dspatial, epsl, x_arr, t_arr, fns_dict, ndim):
  if ndim == 1:
    update_v = update_v_1d
    update_rho = update_rho_1d
  elif ndim == 2:
    update_v = update_v_2d
    update_rho = update_rho_2d
  else:
    raise NotImplementedError
  v_next = update_v(v_prev, phi_bar, rho_prev, sigma, dspatial, fns_dict, x_arr, t_arr)
  rho_next = update_rho(rho_prev, phi_bar, v_next, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr)
  err = jnp.linalg.norm(rho_next - rho_prev) / jnp.maximum(jnp.linalg.norm(rho_prev), 1.0)
  for v0, v1 in zip(v_prev, v_next):
    err = jnp.maximum(err, jnp.linalg.norm(v1 - v0) / jnp.maximum(jnp.linalg.norm(v0), 1.0))
  return rho_next, v_next, err


def update_dual(phi_bar, rho_prev, c_on_rho, v_prev, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr, ndim,
                   rho_v_iters=10, eps=1e-7):
  '''
  @ parameters:
  fns_dict: dict of functions, should contain Hstar_plus_prox_fn, Hstar_minus_prox_fn, Hstar_plus_fn, Hstar_minus_fn
  '''
  for j in range(rho_v_iters):
    rho_next, v_next, err = update_dual_oneiter(phi_bar, rho_prev, c_on_rho, v_prev, sigma, dt, dspatial, epsl,
                                                              x_arr, t_arr, fns_dict, ndim)
    if err < eps:
      break
    rho_prev = rho_next
    v_prev = v_next
  return rho_next, v_next
  