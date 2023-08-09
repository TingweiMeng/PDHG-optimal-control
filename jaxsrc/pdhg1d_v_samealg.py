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
from pdhg_solver import Dy_left_decreasedim, Dy_right_decreasedim, Dy_left_increasedim, Dy_right_increasedim
from pdhg_solver import Dt_decreasedim, Dt_increasedim, Dxx_decreasedim, Dxx_increasedim
from pdhg_solver import Dyy_decreasedim, Dyy_increasedim

jax.config.update("jax_enable_x64", True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

def update_rho_1d(rho_prev, phi, v, sigma, dt, dspatial, epsl, c_on_rho, fns_dict, x_arr, t_arr):
  vp, vm = v[0], v[1]
  dx = dspatial[0]
  vec = -Dx_right_decreasedim(phi, dx) * vp - Dx_left_decreasedim(phi, dx) * vm
  vec = vec - Dt_decreasedim(phi,dt) + epsl * Dxx_decreasedim(phi, dx)  # [nt-1, nx]
  rho_next = jnp.maximum(rho_prev - sigma * vec, -c_on_rho)  # [nt-1, nx]
  # print('rho max {}, rho min {}'.format(jnp.max(rho_next), jnp.min(rho_next)))
  return rho_next

def update_v_1d(v_prev, phi, rho, sigma, dspatial, c_on_rho, fns_dict, x_arr, t_arr, eps=1e-4):
  '''
  @ parameters:
    Hstar_plus_prox_fn and Hstar_minus_prox_fn are prox point operator taking (x,t) as input
    and output argmin_u H(u) + |x-u|^2/(2t)
  '''
  vp_next = 0*rho
  vm_next = 0*rho + 1
  return [vp_next, vm_next]



@jax.jit
def update_primal_1d(phi_prev, rho_prev, c_on_rho, v_prev, tau, dt, dspatial, fv, epsl):
  vp_prev, vm_prev = v_prev[0], v_prev[1]
  dx = dspatial[0]
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

@partial(jax.jit, static_argnames=("fns_dict", "ndim"))
def update_dual_oneiter(phi_bar, rho_prev, c_on_rho, v_prev, sigma, dt, dspatial, epsl, x_arr, t_arr, fns_dict, ndim):
  v_next = update_v_1d(v_prev, phi_bar, rho_prev, sigma, dspatial, c_on_rho, fns_dict, x_arr, t_arr)
  rho_next = update_rho_1d(rho_prev, phi_bar, v_next, sigma, dt, dspatial, epsl, c_on_rho, fns_dict, x_arr, t_arr)
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
  


