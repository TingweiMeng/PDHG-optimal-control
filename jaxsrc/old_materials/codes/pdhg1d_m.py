import jax
import jax.numpy as jnp
from functools import partial
from einshape import jax_einshape as einshape
import os
import solver
from pdhg_solver import Dx_left_increasedim, Dx_right_decreasedim, Dt_decreasedim, Dt_increasedim, Dxx_decreasedim, Dxx_increasedim

jax.config.update("jax_enable_x64", True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


def get_Gsq_from_rho(rho, c_in_H, z):
  '''
  @parameters:
    rho: [7, nt-1, nx]
    c_in_H: [1, 1, nx] or [1, nt-1, nx]
    z: [nt-1, nx]
  @return 
    fn_val: [7, nt-1, nx]
  '''
  n_can = jnp.shape(rho)[0]
  z_left = jnp.roll(z, 1, axis = -1)
  z_rep = einshape("ij->kij", z, k=n_can)
  z_left_rep = einshape("ij->kij", z_left, k=n_can)
  c_in_H_left = jnp.roll(c_in_H, 1, axis = -1)  # [1, 1, nx] or [1, nt-1, nx]
  G1 = jnp.minimum(rho * c_in_H + z_rep, 0) # when z <= 0
  G2 = jnp.minimum(rho * c_in_H_left - z_left_rep, 0) # when z_left > 0
  G = jnp.zeros_like(rho)
  G = jnp.where(z_rep <= 0, G + G1 ** 2, G)
  G = jnp.where(z_left_rep > 0, G + G2 ** 2, G)
  return G # [n_can, nt-1, nx]


def get_minimizer_ind(rho_candidates, alp, c, z, c_in_H):
  '''
  A2_mul_phi is of size ((nt-1)*nx, 1)
  for each (k,i) index, find min_r (r - alp)^2 + G(rho)_{k,i}^2 in candidates
  @ parameters:
    rho_candidates: [7, nt-1, nx]
    alp: [nt-1, nx]
    c: scalar
    z: [nt-1, nx]
    c_in_H: [1, nx] or [nt-1, nx]
  @ return: 
    rho_min: [nt-1, nx]
  '''
  fn_val = (rho_candidates - alp[None,...])**2 # [7, nt-1, nx]
  fn_val_p = fn_val + get_Gsq_from_rho(rho_candidates, c_in_H[None,...], z)
  minindex = jnp.argmin(fn_val_p, axis=0, keepdims=True)
  rho_min = jnp.take_along_axis(rho_candidates, minindex, axis = 0)
  return rho_min[0,:,:]

@jax.jit
def update_primal_1d(phi_prev, rho_prev, c_on_rho, m_prev_ls, tau, dt, dspatial, fv, epsl):
  m_prev = m_prev_ls[0]
  dx = dspatial[0]
  delta_phi = Dx_left_increasedim(m_prev, dx) + Dt_increasedim(rho_prev, dt) + epsl * Dxx_increasedim(rho_prev, dx) # [nt, nx]
  delta_phi = jnp.concatenate([delta_phi[:-1,...], delta_phi[-1:,...] + c_on_rho/dt], axis = 0)
  phi_next = phi_prev - tau * solver.Poisson_eqt_solver(delta_phi, fv, dt)
  return phi_next

@partial(jax.jit, static_argnames=("fns_dict",))
def update_dual_1d(phi_bar, rho_prev, c_on_rho, m_prev_ls, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr, ndim):
  m_prev = m_prev_ls[0]
  dx = dspatial[0]
  f_in_H = fns_dict.f_in_H_fn(x_arr, t_arr)
  c_in_H = fns_dict.c_in_H_fn(x_arr, t_arr)
  rho_candidates = []
  z = m_prev + sigma * Dx_right_decreasedim(phi_bar, dx)  # [nt-1, nx]
  z_left = jnp.roll(z, 1, axis = 1) # [vec1(:,end), vec1(:,1:end-1)]
  alp = rho_prev + sigma * (Dt_decreasedim(phi_bar, dt) - epsl * Dxx_decreasedim(phi_bar, dx) + f_in_H) # [nt-1, nx]
  c_in_H_left = jnp.roll(c_in_H, 1, axis = 1)

  rho_candidates.append(jnp.zeros_like(rho_prev))  # left bound
  # two possible quadratic terms on G, 4 combinations
  vec3 = - z * c_in_H
  vec4 = z_left * c_in_H_left
  rho_candidates.append(jnp.maximum(alp, 0))  # for rho large, G = 0
  rho_candidates.append(jnp.maximum((alp + vec3)/(1+ c_in_H*c_in_H), 0))
  rho_candidates.append(jnp.maximum((alp + vec4)/(1+ c_in_H_left*c_in_H_left), 0))
  rho_candidates.append(jnp.maximum((alp + vec3 + vec4)/(1+ c_in_H*c_in_H + c_in_H_left*c_in_H_left), 0))
  rho_candidates.append(jnp.maximum(- z / c_in_H, 0)) # boundary term 1
  rho_candidates.append(jnp.maximum(z_left / c_in_H_left, 0)) # boundary term 2
  
  rho_candidates = jnp.array(rho_candidates) # [7, nt-1, nx]
  rho_next = get_minimizer_ind(rho_candidates, alp, c_on_rho, z, c_in_H)
  # m is truncation of vec1 into [-rho_i c(xi), rho_{i+1} c(x_{i+1})]
  m_next = jnp.minimum(jnp.maximum(z, -rho_next * c_in_H), 
                        jnp.roll(rho_next * c_in_H, -1, axis = 1))
  return rho_next, (m_next,)
