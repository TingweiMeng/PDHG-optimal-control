import jax
import jax.numpy as jnp
from functools import partial
from einshape import jax_einshape as einshape
import os
import solver
from pdhg_solver import Dx_left_increasedim, Dx_right_decreasedim, Dt_decreasedim, Dt_increasedim, Dxx_decreasedim, Dxx_increasedim
from pdhg_solver import Dy_left_increasedim, Dy_right_decreasedim, Dyy_decreasedim, Dyy_increasedim

jax.config.update("jax_enable_x64", True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


def get_Gsq_from_rho(rho, c_in_H, z1, z2):
  '''
  @parameters:
    rho: [21, nt-1, nx, ny]
    c_in_H: [1, 1, nx, ny] or [1, nt-1, nx, ny]
    z1, z2: [nt-1, nx, ny]
  @return 
    fn_val: [21, nt-1, nx, ny]
  '''
  n_can = jnp.shape(rho)[0]
  z1_xleft = jnp.roll(z1, 1, axis = -2)
  z1_rep = einshape("ijl->kijl", z1, k=n_can)
  z1_xleft_rep = einshape("ijl->kijl", z1_xleft, k=n_can)
  z2_yleft = jnp.roll(z2, 1, axis = -1)
  z2_rep = einshape("ijl->kijl", z2, k=n_can)
  z2_yleft_rep = einshape("ijl->kijl", z2_yleft, k=n_can)
  c_in_H_xleft = jnp.roll(c_in_H, 1, axis = -2)  # [1, 1, nx, ny] or [1, nt-1, nx, ny]
  c_in_H_yleft = jnp.roll(c_in_H, 1, axis = -1)  # [1, 1, nx, ny] or [1, nt-1, nx, ny]
  G1_1 = jnp.minimum(rho * c_in_H + z1_rep, 0) # when z1 <= 0
  G2_1 = jnp.minimum(rho * c_in_H_xleft - z1_xleft_rep, 0) # when z1_xleft >0
  G1_2 = jnp.minimum(rho * c_in_H + z2_rep, 0) # when z2 <= 0
  G2_2 = jnp.minimum(rho * c_in_H_yleft - z2_yleft_rep, 0) # when z2_yleft >0
  G = jnp.zeros_like(rho)
  G = jnp.where(z1_rep <= 0, G + G1_1 ** 2, G)
  G = jnp.where(z1_xleft_rep > 0, G + G2_1 ** 2, G)
  G = jnp.where(z2_rep <= 0, G + G1_2 ** 2, G)
  G = jnp.where(z2_yleft_rep > 0, G + G2_2 ** 2, G)
  return G  # [n_can, nt-1, nx, ny]

def get_minimizer_ind(rho_candidates, alp, c, z1, z2, c_in_H):
  '''
  for each (k,i,j) index, find min_r (r - alp)^2 + G(rho)_{k,i,j}^2 in candidates
  @ parameters:
    rho_candidates: [21, nt-1, nx, ny]
    alp: [nt-1, nx, ny]
    c: scalar
    z1, z2: [nt-1, nx, ny]
    c_in_H: [1, nx, ny] or [nt-1, nx, ny]
  @ return: 
    rho_min: [nt-1, nx, ny]
  '''
  fn_val = (rho_candidates - alp[None,...])**2 # [21, nt-1, nx, ny]
  fn_val_p = fn_val + get_Gsq_from_rho(rho_candidates, c_in_H[None,...], z1, z2)
  minindex = jnp.argmin(fn_val_p, axis=0, keepdims=True)
  rho_min = jnp.take_along_axis(rho_candidates, minindex, axis = 0)
  return rho_min[0,...]


@jax.jit
def update_primal(phi_prev, rho_prev, c_on_rho, m_prev_ls, tau, dt, dspatial, fv, epsl):
  m1_prev, m2_prev = m_prev_ls[0], m_prev_ls[1]
  dx, dy = dspatial[0], dspatial[1]
  delta_phi = Dx_left_increasedim(m1_prev, dx) + Dy_left_increasedim(m2_prev, dy) \
              + Dt_increasedim(rho_prev, dt) + epsl * Dxx_increasedim(rho_prev, dx) \
              + epsl * Dyy_increasedim(rho_prev, dy)  # [nt, nx, ny]
  delta_phi = jnp.concatenate([delta_phi[:-1,...], delta_phi[-1:,...] + c_on_rho/dt], axis = 0)
  phi_next = phi_prev - tau * solver.Poisson_eqt_solver_2d(delta_phi, fv, dt)
  return phi_next

@partial(jax.jit, static_argnames=("fns_dict",))
def update_dual(phi_bar, rho_prev, c_on_rho, m_prev_ls, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr, ndim):
  eps = 1e-4
  m1_prev, m2_prev = m_prev_ls[0], m_prev_ls[1]
  dx, dy = dspatial[0], dspatial[1]
  f_in_H = fns_dict.f_in_H_fn(x_arr, t_arr)
  c_in_H = fns_dict.c_in_H_fn(x_arr, t_arr)  # [1, nx, ny] or [nt-1, nx, ny]

  rho_candidates = []
  z1 = m1_prev + sigma * Dx_right_decreasedim(phi_bar, dx)  # [nt-1, nx, ny]
  z2 = m2_prev + sigma * Dy_right_decreasedim(phi_bar, dy)  # [nt-1, nx, ny]
  z1_xleft = jnp.roll(z1, 1, axis = 1)
  z2_yleft = jnp.roll(z2, 1, axis = 2)
  alp = rho_prev + sigma * (Dt_decreasedim(phi_bar, dt) - epsl * Dxx_decreasedim(phi_bar, dx)\
                            - epsl * Dyy_decreasedim(phi_bar, dy) + f_in_H) # [nt-1, nx, ny]
  rho_candidates_1 = jnp.zeros_like(rho_prev)[None,...]  # left bound, [1,nt-1, nx,ny]
  # 16 candidates using mask
  mask = jnp.array([[0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1], [0,1,0,0], [0,1,0,1], [0,1,1,0], [0,1,1,1], 
                    [1,0,0,0], [1,0,0,1], [1,0,1,0], [1,0,1,1], [1,1,0,0], [1,1,0,1], [1,1,1,0], [1,1,1,1]])
  c_in_H_xleft = jnp.roll(c_in_H, 1, axis = -2)  # [nt-1, nx, ny] or [1,nx,ny]
  c_in_H_yleft = jnp.roll(c_in_H, 1, axis = -1)  # [nt-1, nx, ny] or [1,nx,ny]
  denominator = c_in_H[None,...]**2 * mask[:,0,None,None,None] + c_in_H_xleft[None,...]**2 * mask[:,1,None,None,None] \
              + c_in_H[None,...]**2 * mask[:,2,None,None,None] + c_in_H_yleft[None,...]**2 * mask[:,3,None,None,None]  # [16, nt-1, nx, ny]
  c_in_H_mul_z1 = c_in_H[None,...] * z1[None,...]
  c_in_H_mul_z2 = c_in_H[None,...] * z2[None,...]
  c_in_H_mul_z1_xleft = jnp.roll(c_in_H_mul_z1, 1, axis = -2)  # [1, nt-1, nx, ny]
  c_in_H_mul_z2_yleft = jnp.roll(c_in_H_mul_z2, 1, axis = -1)  # [1, nt-1, nx, ny]
  numerator = c_in_H_mul_z1 * mask[:,0,None,None,None] - c_in_H_mul_z1_xleft * mask[:,1,None,None,None] \
                + c_in_H_mul_z2 * mask[:,2,None,None,None] - c_in_H_mul_z2_yleft * mask[:,3,None,None,None]  # [16, nt-1, nx, ny]
  
  rho_candidates_16 = jnp.maximum((alp[None,...] - numerator) / (1 + denominator), 0)
  rho_candidates_17 = jnp.maximum(- z1/ c_in_H, 0)[None,...]  # [1, nt-1, nx, ny]
  rho_candidates_18 = jnp.maximum(z1_xleft/ c_in_H_xleft, 0)[None,...]  # [1, nt-1, nx, ny]
  rho_candidates_19 = jnp.maximum(- z2/ c_in_H, 0)[None,...]  # [1, nt-1, nx, ny]
  rho_candidates_20 = jnp.maximum(z2_yleft/ c_in_H_yleft, 0)[None,...]  # [1, nt-1, nx, ny]
  rho_candidates = jnp.concatenate([rho_candidates_1, rho_candidates_16, rho_candidates_17, rho_candidates_18,
                                    rho_candidates_19, rho_candidates_20], axis = 0) # [21, nt-1, nx, ny]  (16 candidates, lower bound, and boundary candidates)
  rho_next = get_minimizer_ind(rho_candidates, alp, c_on_rho, z1, z2, c_in_H)

  m1_next = jnp.minimum(jnp.maximum(z1, -rho_next * c_in_H), jnp.roll(rho_next* c_in_H, -1, axis = -2))
  m2_next = jnp.minimum(jnp.maximum(z2, -rho_next * c_in_H), jnp.roll(rho_next * c_in_H, -1, axis = -1))
  return rho_next, (m1_next, m2_next)
  