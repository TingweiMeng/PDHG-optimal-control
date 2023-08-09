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
from pdhg_solver import Dx_left_increasedim, Dx_right_decreasedim, Dt_decreasedim, Dt_increasedim, Dxx_decreasedim, Dxx_increasedim
from pdhg_solver import Dy_left_increasedim, Dy_right_decreasedim, Dyy_decreasedim, Dyy_increasedim

jax.config.update("jax_enable_x64", True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


def get_Gsq_from_rho(rho_plus_c_mul_cinH, z1, z2):
  '''
  @parameters:
    rho_plus_c_mul_cinH: [21, nt-1, nx, ny]
    z1, z2: [nt-1, nx, ny]
  @return 
    fn_val: [21, nt-1, nx, ny]
  '''
  n_can = jnp.shape(rho_plus_c_mul_cinH)[0]
  z1_left = jnp.roll(z1, 1, axis = 1)
  z1_rep = einshape("ijl->kijl", z1, k=n_can)
  z1_left_rep = einshape("ijl->kijl", z1_left, k=n_can)
  z2_left = jnp.roll(z2, 1, axis = 2)
  z2_rep = einshape("ijl->kijl", z2, k=n_can)
  z2_left_rep = einshape("ijl->kijl", z2_left, k=n_can)
  G1_1 = jnp.minimum(rho_plus_c_mul_cinH + z1_rep, 0) # when z1 < 0
  G2_1 = jnp.minimum(rho_plus_c_mul_cinH - z1_left_rep, 0) # when z1_left >=0
  G1_2 = jnp.minimum(rho_plus_c_mul_cinH + z2_rep, 0) # when z2 < 0
  G2_2 = jnp.minimum(rho_plus_c_mul_cinH - z2_left_rep, 0) # when z2_left >=0
  G = jnp.zeros_like(rho_plus_c_mul_cinH)
  G = jnp.where(z1_rep < 0, G + G1_1 ** 2, G)
  G = jnp.where(z1_left_rep >= 0, G + G2_1 ** 2, G)
  G = jnp.where(z2_rep < 0, G + G1_2 ** 2, G)
  G = jnp.where(z2_left_rep >= 0, G + G2_2 ** 2, G)
  return G  # [n_can, nt-1, nx, ny]

def get_minimizer_ind(rho_candidates, alp, c, z1, z2, c_in_H):
  '''
  for each (k,i,j) index, find min_r (r - alp)^2 + G(rho)_{k,i,j}^2 in candidates
  @ parameters:
    rho_candidates: [21, nt-1, nx, ny]
    alp: [nt-1, nx, ny]
    c: scalar
    z1, z2: [nt-1, nx, ny]
    c_in_H: [1, nx, ny]
  @ return: 
    rho_min: [nt-1, nx, ny]
  '''
  fn_val = (rho_candidates - alp[None,...])**2 # [21, nt-1, nx, ny]
  fn_val_p = fn_val + get_Gsq_from_rho((rho_candidates + c) * c_in_H[None,...], z1, z2)
  minindex = jnp.argmin(fn_val_p, axis=0, keepdims=True)
  rho_min = jnp.take_along_axis(rho_candidates, minindex, axis = 0)
  return rho_min[0,...]

def update_phi_preconditioning(delta_phi, phi_prev, fv, dt):
  '''
  @parameters:
    delta_phi: [nt, nx, ny]
    phi_prev: [nt, nx, ny]
    fv: [nx, ny], complex
    dt: scalar
  @return:
    phi_next: [nt, nx, ny]
  '''
  nt, nx, ny = jnp.shape(delta_phi)
  v_Fourier =  jnp.fft.fft2(delta_phi, axes = (1,2)) # [nt, nx, ny]
  dl = jnp.pad(1/(dt*dt)*jnp.ones((nt-1,)), (1,0), mode = 'constant', constant_values=0.0).astype(jnp.complex128)
  du = jnp.pad(1/(dt*dt)*jnp.ones((nt-1,)), (0,1), mode = 'constant', constant_values=0.0).astype(jnp.complex128)
  thomas_b = einshape('nk->mnk', fv - 2/(dt*dt), m = nt) # [nt, nx, ny]
  phi_fouir_part = solver.tridiagonal_solve_batch_2d(dl, thomas_b, du, v_Fourier)  # [nt, nx, ny]
  F_phi_updates = jnp.fft.ifft2(phi_fouir_part, axes = (1,2)).real  # [nt, nx, ny]
  phi_next = phi_prev + F_phi_updates
  return phi_next

@jax.jit
def update_primal(phi_prev, rho_prev, c_on_rho, m_prev_ls, tau, dt, dspatial, fv, epsl):
  m_prev = m_prev_ls[0]
  dx = dspatial[0]
  delta_phi = - tau * (Dx_left_increasedim(m_prev, dx) + Dt_increasedim(rho_prev, dt) + epsl * Dxx_increasedim(rho_prev, dx)) # [nt, nx]
  # phi_next = phi_prev + solver.Poisson_eqt_solver(delta_phi, fv, dt)
  reg_param = 10
  reg_param2 = 1
  f = -2*reg_param *phi_prev[0:1,...]
  # phi_next = phi_prev + solver.pdhg_phi_update(delta_phi, phi_prev, fv, dt, Neumann_cond = True, reg_param = reg_param)
  phi_next_1 = solver.pdhg_precondition_update_2d(delta_phi[1:,...], phi_prev[1:,...], fv, dt, 
                                    reg_param = reg_param, reg_param2=reg_param2, f=f)
  phi_next = jnp.concatenate([phi_prev[0:1,...], phi_next_1], axis = 0)
  return phi_next

@partial(jax.jit, static_argnames=("fns_dict",))
def update_dual(phi_bar, rho_prev, c_on_rho, m_prev_ls, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr, ndim):
  eps = 1e-4
  m1_prev, m2_prev = m_prev_ls[0], m_prev_ls[1]
  dx, dy = dspatial[0], dspatial[1]
  f_in_H = fns_dict.f_in_H_fn(x_arr, t_arr)
  c_in_H = fns_dict.c_in_H_fn(x_arr, t_arr)

  rho_candidates = []
  z1 = m1_prev + sigma * Dx_right_decreasedim(phi_bar, dx)  # [nt-1, nx, ny]
  z2 = m2_prev + sigma * Dy_right_decreasedim(phi_bar, dy)  # [nt-1, nx, ny]
  z1_left = jnp.roll(z1, 1, axis = 1)
  z2_left = jnp.roll(z2, 1, axis = 2)
  alp = rho_prev + sigma * (Dt_decreasedim(phi_bar, dt) - epsl * Dxx_decreasedim(phi_bar, dx)\
                            - epsl * Dyy_decreasedim(phi_bar, dy) + f_in_H) # [nt-1, nx, ny]
  rho_candidates_1 = -c_on_rho * jnp.ones_like(rho_prev)[None,...]  # left bound, [1,nt-1, nx,ny]
  # 16 candidates using mask
  mask = jnp.array([[0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1], [0,1,0,0], [0,1,0,1], [0,1,1,0], [0,1,1,1], 
                    [1,0,0,0], [1,0,0,1], [1,0,1,0], [1,0,1,1], [1,1,0,0], [1,1,0,1], [1,1,1,0], [1,1,1,1]])
  num_vec_in_C = jnp.sum(mask, axis = -1)[:,None,None,None]   # [16, 1, 1,1]
  sum_vec_in_C = -z1_left[None,...] * mask[:,0,None,None,None] \
              + z1[None,...] * mask[:,1,None,None,None] + z2[None,...] * mask[:,2,None,None,None] \
              - z2_left[None,...] * mask[:,3,None,None,None]  # [16, nt-1, nx, ny]
  rho_candidates_16 = jnp.maximum((alp[None,...] - num_vec_in_C * c_in_H[None,...] **2 * c_on_rho - c_in_H[None,...] *\
              sum_vec_in_C) / (1 + num_vec_in_C * c_in_H[None,...]**2), -c_on_rho)
  rho_candidates_17 = jnp.maximum(-c_on_rho - z1/ c_in_H, -c_on_rho)[None,...]  # [1, nt-1, nx, ny]
  rho_candidates_18 = jnp.maximum(-c_on_rho - z2/ c_in_H, -c_on_rho)[None,...]  # [1, nt-1, nx, ny]
  rho_candidates_19 = jnp.maximum(-c_on_rho + z1_left/ c_in_H, -c_on_rho)[None,...]  # [1, nt-1, nx, ny]
  rho_candidates_20 = jnp.maximum(-c_on_rho + z2_left/ c_in_H, -c_on_rho)[None,...]  # [1, nt-1, nx, ny]
  rho_candidates = jnp.concatenate([rho_candidates_1, rho_candidates_16, rho_candidates_17, rho_candidates_18,
                                    rho_candidates_19, rho_candidates_20], axis = 0) # [21, nt-1, nx, ny]  (16 candidates, lower bound, and boundary candidates)
  rho_next = get_minimizer_ind(rho_candidates, alp, c_on_rho, z1, z2, c_in_H)

  m1_next = jnp.minimum(jnp.maximum(z1, -(rho_next + c_on_rho) * c_in_H), 
                        (jnp.roll(rho_next, -1, axis = 1) + c_on_rho) * jnp.roll(c_in_H, -1, axis = 1))
  # m2 is truncation of z2 into [-(rho_{i,j}+c)c(xi,yj), (rho_{i,j+1}+c)c(xi, y_{j+1})]
  m2_next = jnp.minimum(jnp.maximum(z2, -(rho_next + c_on_rho) * c_in_H), 
                        (jnp.roll(rho_next, -1, axis = 2) + c_on_rho) * jnp.roll(c_in_H, -1, axis = 2))
  return rho_next, (m1_next, m2_next)
  
  