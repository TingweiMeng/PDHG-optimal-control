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

def update_rho_1d(rho_prev, phi, alp, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr, fwd, precond, fv):
  dx = dspatial[0]
  L_val = fns_dict.L_fn(alp, x_arr, t_arr)
  f_plus_val = fns_dict.f_plus_fn(alp, x_arr, t_arr)
  f_minus_val = fns_dict.f_minus_fn(alp, x_arr, t_arr)
  # print('phi: ', phi)
  # print('f_plus_val: ', f_plus_val)
  # print('f_minus_val: ', f_minus_val)
  # print('Dx_right: ', Dx_right_decreasedim(phi, dx, fwd=fwd))
  # print('Dx_left: ', Dx_left_decreasedim(phi, dx, fwd=fwd))
  # print('Dt: ', Dt_decreasedim(phi, dt))
  # print('L_val: ', L_val)
  vec = Dx_right_decreasedim(phi, dx, fwd=fwd) * f_minus_val + Dx_left_decreasedim(phi, dx, fwd=fwd) * f_plus_val
  vec = vec + Dt_decreasedim(phi, dt) - epsl * Dxx_decreasedim(phi, dx, fwd=fwd)  # [nt-1, nx]
  vec = vec + L_val
  print('vec: ', vec)
  if precond:
    rho_next = rho_prev - sigma * solver.Poisson_eqt_solver_termcond(vec, fv, dt)
  else:
    rho_next = rho_prev - sigma * vec
  
  # print('rho_next in update: ', rho_next, flush=True)
  rho_next = jnp.maximum(rho_next, 0.0)  # [nt-1, nx]
  return rho_next

def update_alp(alp_prev, phi, rho, sigma, dspatial, fns_dict, x_arr, t_arr):
  dx = dspatial[0]
  Dx_phi_left = Dx_left_decreasedim(phi, dx)
  Dx_phi_right = Dx_right_decreasedim(phi, dx)
  # print('Dx_phi_left', Dx_phi_left.shape)
  # print('Dx_phi_right', Dx_phi_right.shape)
  # print('x_arr', x_arr.shape)
  # print('t_arr', t_arr.shape)
  # print('alp_prev', alp_prev)
  # print('sigma', sigma, flush=True)
  alp = fns_dict.opt_alp_fn(Dx_phi_left, Dx_phi_right, x_arr, t_arr, alp_prev, sigma)
  return alp

@partial(jax.jit, static_argnames=("fns_dict", ))
def update_primal_1d(phi_prev, rho_prev, c_on_rho, alp_prev, tau, dt, dspatial, fv, epsl, fwd, precond, fns_dict, x_arr, t_arr):
  dx = dspatial[0]
  eps = 1e-4
  fp = fns_dict.f_plus_fn(alp_prev, x_arr, t_arr)
  fm = fns_dict.f_minus_fn(alp_prev, x_arr, t_arr)
  # print('Dx_left_increasedim(fm, dx): ', Dx_left_increasedim(fm * rho_prev, dx))
  # print('Dx_right_increasedim(fp, dx): ', Dx_right_increasedim(fp * rho_prev, dx))
  delta_phi = Dx_left_increasedim(fm * rho_prev, dx, fwd=fwd) + Dx_right_increasedim(fp * rho_prev, dx, fwd=fwd) \
              + epsl * Dxx_increasedim(rho_prev,dx, fwd=fwd) # [nt, nx]
  # print('delta_phi: ', delta_phi)
  delta_phi += Dt_increasedim(rho_prev,dt) # [nt, nx]
  # print('Dt: ', Dt_increasedim(rho_prev,dt))
  delta_phi = jnp.concatenate([delta_phi[:1,...] - c_on_rho/dt, delta_phi[1:-1,...], 0*delta_phi[-1:,...]], axis = 0)
  # print('delta_phi: ', delta_phi)
  if precond:
    phi_next = phi_prev - tau * solver.Poisson_eqt_solver_termcond(delta_phi, fv, dt)
  else:
    phi_next = phi_prev - tau * delta_phi
  return phi_next

@partial(jax.jit, static_argnames=("fns_dict", "ndim"))
def update_dual_oneiter(phi_bar, rho_prev, c_on_rho, v_prev, sigma, dt, dspatial, epsl, x_arr, t_arr, 
                        fns_dict, ndim, fwd, precond, fv):
  if ndim == 1:
    update_v = update_alp
    update_rho = update_rho_1d
  else:
    raise NotImplementedError
  v_next = update_v(v_prev, phi_bar, rho_prev, sigma, dspatial, fns_dict, x_arr, t_arr)
  rho_next = update_rho(rho_prev, phi_bar, v_next, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr, fwd, precond=precond, fv=fv)
  err = jnp.linalg.norm(rho_next - rho_prev) / jnp.maximum(jnp.linalg.norm(rho_prev), 1.0)
  for v0, v1 in zip(v_prev, v_next):
    err = jnp.maximum(err, jnp.linalg.norm(v1 - v0) / jnp.maximum(jnp.linalg.norm(v0), 1.0))
  return rho_next, v_next, err


def update_dual(phi_bar, rho_prev, c_on_rho, v_prev, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr, ndim, fwd, fv,
                   rho_v_iters=1, eps=1e-7, precond=False):
  '''
  @ parameters:
  fns_dict: dict of functions, see the function set_up_example_fns in solver.py
  '''
  for j in range(rho_v_iters):
    # print('v_prev', v_prev, flush=True)
    # print('rho_prev', v_prev, flush=True)
    rho_next, v_next, err = update_dual_oneiter(phi_bar, rho_prev, c_on_rho, v_prev, sigma, dt, dspatial, epsl,
                                                              x_arr, t_arr, fns_dict, ndim, fwd, precond=precond, fv=fv)
    if err < eps:
      break
    rho_prev = rho_next
    v_prev = v_next
  return rho_next, v_next
  