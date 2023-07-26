import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import utils
from einshape import jax_einshape as einshape
import os
import solver
import pickle
import matplotlib.pyplot as plt
from pdhg_solver import Dx_left_decreasedim, Dx_right_decreasedim, Dx_left_increasedim, Dx_right_increasedim
from pdhg_solver import Dt_decreasedim, Dt_increasedim, Dxx_decreasedim, Dxx_increasedim

jax.config.update("jax_enable_x64", True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

def update_rho_1d(rho_prev, phi, vp, vm, sigma, dt, dx, epsl, c_on_rho, Hstar_plus_fn, Hstar_minus_fn, x_arr, t_arr):
  vec = -Dx_right_decreasedim(phi, dx) * vp - Dx_left_decreasedim(phi, dx) * vm
  vec = vec - Dt_decreasedim(phi,dt) + epsl * Dxx_decreasedim(phi, dx)  # [nt-1, nx]
  vec = vec + Hstar_plus_fn(vm, x_arr, t_arr) + Hstar_minus_fn(vp, x_arr, t_arr)
  rho_next = jnp.maximum(rho_prev - sigma * vec, -c_on_rho)  # [nt-1, nx]
  return rho_next

def update_v_1d(vp_prev, vm_prev, phi, rho, sigma, dx, c_on_rho, 
               Hstar_plus_prox_fn, Hstar_minus_prox_fn, x_arr, t_arr, eps=1e-4):
  '''
  @ parameters:
    Hstar_plus_prox_fn and Hstar_minus_prox_fn are prox point operator taking (x,t) as input
    and output argmin_u H(u) + |x-u|^2/(2t)
  '''
  param = sigma / (rho + c_on_rho + eps)
  vp_next_raw = vp_prev + param * Dx_right_decreasedim(phi, dx)  # [nt-1, nx]
  vm_next_raw = vm_prev + param * Dx_left_decreasedim(phi, dx)  # [nt-1, nx]
  vp_next = Hstar_minus_prox_fn(vp_next_raw, param, x_arr, t_arr)  # [nt-1, nx]
  vm_next = Hstar_plus_prox_fn(vm_next_raw, param, x_arr, t_arr)  # [nt-1, nx]  
  return vp_next, vm_next

@jax.jit
def update_primal_1d(phi_prev, rho_prev, c_on_rho, vp_prev, vm_prev, tau, dt, dx, fv, epsl):
  eps = 1e-4
  mp_prev = (rho_prev + c_on_rho + eps) * vp_prev  # [nt-1, nx]
  mm_prev = (rho_prev + c_on_rho + eps) * vm_prev  # [nt-1, nx]
  delta_phi = - tau * (Dx_left_increasedim(mp_prev, dx) + Dx_right_increasedim(mm_prev, dx) \
                      + Dt_increasedim(rho_prev,dt) + epsl * Dxx_increasedim(rho_prev,dx)) # [nt, nx]
  phi_next = phi_prev + solver.Poisson_eqt_solver(delta_phi, fv, dt)
  # reg_param = 10
  # reg_param2 = 1
  # f = -2*reg_param *phi_prev[0:1,:]
  # # phi_next = phi_prev + solver.pdhg_phi_update(delta_phi, phi_prev, fv, dt, Neumann_cond = True, reg_param = reg_param)
  # phi_next_1 = solver.pdhg_precondition_update(delta_phi[1:,:], phi_prev[1:,:], fv, dt, 
  #                                   reg_param = reg_param, reg_param2=reg_param2, f=f)
  # phi_next = jnp.concatenate([phi_prev[0:1,:], phi_next_1], axis = 0)
  return phi_next

# @partial(jax.jit, static_argnames=("fns_dict",))
def update_dual_1d(phi_bar, rho_prev, c_on_rho, vp_prev, vm_prev, sigma, dt, dx, epsl, fns_dict,
                  x_arr, t_arr, rho_v_iters=10, eps=1e-7):
  '''
  @ parameters:
  fns_dict: dict of functions, should contain Hstar_plus_prox_fn, Hstar_minus_prox_fn, Hstar_plus_fn, Hstar_minus_fn
  '''
  Hstar_plus_prox_fn = fns_dict['Hstar_plus_prox_fn']
  Hstar_minus_prox_fn = fns_dict['Hstar_minus_prox_fn']
  Hstar_plus_fn = fns_dict['Hstar_plus_fn']
  Hstar_minus_fn = fns_dict['Hstar_minus_fn']
  for j in range(rho_v_iters):
    vp_next, vm_next = update_v_1d(vp_prev, vm_prev, phi_bar, rho_prev, sigma, dx, c_on_rho, 
                                Hstar_plus_prox_fn, Hstar_minus_prox_fn, x_arr, t_arr)
    rho_next = update_rho_1d(rho_prev, phi_bar, vp_next, vm_next, sigma, dt, dx, epsl, c_on_rho,
                          Hstar_plus_fn, Hstar_minus_fn, x_arr, t_arr)
    err1 = jnp.linalg.norm(vp_next - vp_prev) / jnp.maximum(jnp.linalg.norm(vp_prev), 1.0)
    err2 = jnp.linalg.norm(vm_next - vm_prev) / jnp.maximum(jnp.linalg.norm(vm_prev), 1.0)
    err3 = jnp.linalg.norm(rho_next - rho_prev) / jnp.maximum(jnp.linalg.norm(rho_prev), 1.0)
    if err1 < eps and err2 < eps and err3 < eps:
      break
    rho_prev = rho_next
    vp_prev = vp_next
    vm_prev = vm_next
  return rho_next, vp_next, vm_next
  