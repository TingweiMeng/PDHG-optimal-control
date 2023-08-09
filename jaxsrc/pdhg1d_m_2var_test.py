import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import utils
from einshape import jax_einshape as einshape
import os
import solver
from solver import interpolation_x, interpolation_t
import pickle
import matplotlib.pyplot as plt
from pdhg_solver import Dx_left_increasedim, Dx_right_decreasedim, Dt_decreasedim, Dt_increasedim, Dxx_decreasedim, Dxx_increasedim

jax.config.update("jax_enable_x64", True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


def get_Gsq_from_rho(rho_plus_c_mul_cinH, z):
  '''
  @parameters:
    rho_plus_c_mul_cinH: [7, nt-1, nx]
    z: [nt-1, nx]
  @return 
    fn_val: [7, nt-1, nx]
  '''
  n_can = jnp.shape(rho_plus_c_mul_cinH)[0]
  z_left = jnp.roll(z, 1, axis = 1)
  z_rep = einshape("ij->kij", z, k=n_can)
  z_left_rep = einshape("ij->kij", z_left, k=n_can)
  G1 = jnp.minimum(rho_plus_c_mul_cinH + z_left_rep, 0) # when z_left < 0
  G2 = jnp.minimum(rho_plus_c_mul_cinH - z_rep, 0) # when z >=0
  G = jnp.zeros_like(rho_plus_c_mul_cinH)
  G = jnp.where(z_left_rep < 0, G + G1 ** 2, G)
  G = jnp.where(z_rep >= 0, G + G2 ** 2, G)
  return G # [n_can, nt-1, nx]


def get_minimizer_ind(rho_candidates, shift_term, c, z, c_in_H):
  '''
  A2_mul_phi is of size ((nt-1)*nx, 1)
  for each (k,i) index, find min_r (r - shift_term)^2 + G(rho)_{k,i}^2 in candidates
  @ parameters:
    rho_candidates: [7, nt-1, nx]
    shift_term: [nt-1, nx]
    c: scalar
    z: [nt-1, nx]
    c_in_H: [1, nx]
  @ return: 
    rho_min: [nt-1, nx]
  '''
  fn_val = (rho_candidates - shift_term[None,:,:])**2 # [7, nt-1, nx]
  fn_val_p = fn_val + get_Gsq_from_rho((rho_candidates + c) * c_in_H[None,:,:], z)
  minindex = jnp.argmin(fn_val_p, axis=0, keepdims=True)
  rho_min = jnp.take_along_axis(rho_candidates, minindex, axis = 0)
  return rho_min[0,:,:]

@jax.jit
def update_primal_1d(phi_prev, rho_prev, c_on_rho, m_prev_ls, tau, dt, dspatial, fv, epsl):
  m_prev = m_prev_ls[0]
  dx = dspatial[0]
  delta_phi = - tau * (Dx_left_increasedim(m_prev, dx) + Dt_increasedim(rho_prev, dt) + epsl * Dxx_increasedim(rho_prev, dx)) # [nt, nx]
  # phi_next = phi_prev + solver.Poisson_eqt_solver(delta_phi, fv, dt)
  reg_param = 10
  reg_param2 = 1
  f = -2*reg_param *phi_prev[0:1,:]
  # phi_next = phi_prev + solver.pdhg_phi_update(delta_phi, phi_prev, fv, dt, Neumann_cond = True, reg_param = reg_param)
  phi_next_1 = solver.pdhg_precondition_update(delta_phi[1:,:], phi_prev[1:,:], fv, dt, 
                                    reg_param = reg_param, reg_param2=reg_param2, f=f)
  phi_next = jnp.concatenate([phi_prev[0:1,:], phi_next_1], axis = 0)
  return phi_next

@partial(jax.jit, static_argnames=("fns_dict",))
def update_dual_1d(phi_bar, rho_prev, c_on_rho, m_prev_ls, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr, ndim):
  m_prev = m_prev_ls[0]
  dx = dspatial[0]
  rho_lb = -c_on_rho + 1e-4
  f_in_H = fns_dict.f_in_H_fn(x_arr, t_arr)
  c_in_H = fns_dict.c_in_H_fn(x_arr, t_arr)
  rho_candidates = []
  z = m_prev + sigma * Dx_right_decreasedim(phi_bar, dx)  # [nt-1, nx]
  z_left = jnp.roll(z, 1, axis = 1) # [vec1(:,end), vec1(:,1:end-1)]
  alp = rho_prev + sigma * (Dt_decreasedim(phi_bar, dt) - epsl * Dxx_decreasedim(phi_bar, dx) + f_in_H) # [nt-1, nx]

  rho_candidates.append(rho_lb * jnp.ones_like(rho_prev))  # left bound
  # two possible quadratic terms on G, 4 combinations
  vec3 = -c_in_H * c_in_H * c_on_rho + z * c_in_H
  vec4 = -c_in_H * c_in_H * c_on_rho - z_left * c_in_H
  rho_candidates.append(jnp.maximum(alp, rho_lb))  # for rho large, G = 0
  rho_candidates.append(jnp.maximum((alp + vec3)/(1+ c_in_H*c_in_H), rho_lb))#  % if G_i = (rho_i + c)c(xi) - z_i
  rho_candidates.append(jnp.maximum((alp + vec4)/(1+ c_in_H*c_in_H), rho_lb))#  % if G_i = (rho_i + c)c(xi) + z_{i-1}
  rho_candidates.append(jnp.maximum((alp + vec3 + vec4)/(1+ 2*c_in_H*c_in_H), rho_lb)) # we have both terms above
  rho_candidates.append(jnp.maximum(-c_on_rho + z / c_in_H, rho_lb)) # boundary term 1
  rho_candidates.append(jnp.maximum(-c_on_rho - z_left / c_in_H, rho_lb)) # boundary term 2
  
  rho_candidates = jnp.array(rho_candidates) # [7, nt-1, nx]
  rho_next = get_minimizer_ind(rho_candidates, alp, c_on_rho, z, c_in_H)
  # m is truncation of vec1 into [-(rho_{i+1}+c)c(x{i+1}), (rho_{i}+c)c(x_{i})]
  m_next = jnp.minimum(jnp.maximum(z, -(jnp.roll(rho_next, -1, axis = 1) + c_on_rho) * jnp.roll(c_in_H, -1, axis = 1)), 
                        (rho_next + c_on_rho) * c_in_H)
  return rho_next, (m_next,)
