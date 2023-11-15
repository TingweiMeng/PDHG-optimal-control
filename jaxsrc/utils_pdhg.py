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

# def update_rho_2d(rho_prev, phi, v, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr):
#   vxp, vxm = v[0], v[1]
#   vyp, vym = v[2], v[3]
#   dx, dy = dspatial[0], dspatial[1]
#   if 'Hxstar_plus_fn' in fns_dict._fields and 'Hxstar_minus_fn' in fns_dict._fields and \
#      'Hystar_plus_fn' in fns_dict._fields and 'Hystar_minus_fn' in fns_dict._fields:  # seperable case
#     Hstar_val = fns_dict.Hxstar_plus_fn(vxm, x_arr, t_arr) + fns_dict.Hxstar_minus_fn(vxp, x_arr, t_arr) \
#                 + fns_dict.Hystar_plus_fn(vym, x_arr, t_arr) + fns_dict.Hystar_minus_fn(vyp, x_arr, t_arr)
#   elif 'Hstar_fn' in fns_dict._fields:  # non-seperable case
#     Hstar_val = fns_dict.Hstar_fn(jnp.stack([vxp, vxm, vyp, vym], axis = 0), x_arr, t_arr)
#   else:
#     raise "fns_dict must contain Hstar_fn or Hxstar_plus_fn, Hxstar_minus_fn, Hystar_plus_fn, Hystar_minus_fn"
#   vec = Dx_right_decreasedim(phi, dx) * vxp + Dx_left_decreasedim(phi, dx) * vxm
#   vec = vec + Dy_right_decreasedim(phi, dy) * vyp + Dy_left_decreasedim(phi, dy) * vym
#   vec = vec + Dt_decreasedim(phi,dt) - epsl * Dxx_decreasedim(phi, dx) - epsl * Dyy_decreasedim(phi, dy)  # [nt-1, nx, ny]
#   vec = vec - Hstar_val
#   rho_next = rho_prev + sigma * vec
#   rho_next = jnp.maximum(rho_next, 0.0)  # [nt-1, nx, ny]
#   return rho_next

# def update_v_2d_seperable(v_prev, phi, rho, sigma, dspatial, fns_dict, x_arr, t_arr, eps=1e-4):
#   '''
#   @ parameters:
#     Hstar_plus_prox_fn and Hstar_minus_prox_fn are prox point operator taking (x,t) as input
#     and output argmin_u H(u) + |x-u|^2/(2t), assuming H(p1, p2) = Hx(p1) + Hy(p2)
#   '''
#   vxp_prev, vxm_prev, vyp_prev, vym_prev = v_prev[0], v_prev[1], v_prev[2], v_prev[3]
#   dx, dy = dspatial[0], dspatial[1]
#   Hxstar_plus_prox_fn = fns_dict.Hxstar_plus_prox_fn
#   Hxstar_minus_prox_fn = fns_dict.Hxstar_minus_prox_fn
#   Hystar_plus_prox_fn = fns_dict.Hystar_plus_prox_fn
#   Hystar_minus_prox_fn = fns_dict.Hystar_minus_prox_fn
#   param = sigma / (rho + eps)
#   vxp_next_raw = vxp_prev + param * Dx_right_decreasedim(phi, dx)  # [nt-1, nx, ny]
#   vxm_next_raw = vxm_prev + param * Dx_left_decreasedim(phi, dx)  # [nt-1, nx, ny]
#   vyp_next_raw = vyp_prev + param * Dy_right_decreasedim(phi, dy)  # [nt-1, nx, ny]
#   vym_next_raw = vym_prev + param * Dy_left_decreasedim(phi, dy)  # [nt-1, nx, ny]
#   vxp_next = Hxstar_minus_prox_fn(vxp_next_raw, param, x_arr, t_arr)  # [nt-1, nx, ny]
#   vxm_next = Hxstar_plus_prox_fn(vxm_next_raw, param, x_arr, t_arr)  # [nt-1, nx, ny]  
#   vyp_next = Hystar_minus_prox_fn(vyp_next_raw, param, x_arr, t_arr)  # [nt-1, nx, ny]
#   vym_next = Hystar_plus_prox_fn(vym_next_raw, param, x_arr, t_arr)  # [nt-1, nx, ny]  
#   return (vxp_next, vxm_next, vyp_next, vym_next)


# def update_v_2d_nonseperable(v_prev, phi, rho, sigma, dspatial, fns_dict, x_arr, t_arr, eps=1e-4):
#   '''
#   @ parameters:
#     Hstar_plus_prox_fn and Hstar_minus_prox_fn are prox point operator taking (x,t) as input
#     and output argmin_u H(u) + |x-u|^2/(2t), assume H is general
#   '''
#   vxp_prev, vxm_prev, vyp_prev, vym_prev = v_prev[0], v_prev[1], v_prev[2], v_prev[3]
#   dx, dy = dspatial[0], dspatial[1]
#   param = sigma / (rho + eps)
#   vxp_next_raw = vxp_prev + param * Dx_right_decreasedim(phi, dx)  # [nt-1, nx, ny]
#   vxm_next_raw = vxm_prev + param * Dx_left_decreasedim(phi, dx)  # [nt-1, nx, ny]
#   vyp_next_raw = vyp_prev + param * Dy_right_decreasedim(phi, dy)  # [nt-1, nx, ny]
#   vym_next_raw = vym_prev + param * Dy_left_decreasedim(phi, dy)  # [nt-1, nx, ny]
#   v_next_raw = jnp.stack([vxp_next_raw, vxm_next_raw, vyp_next_raw, vym_next_raw], axis = 0)  # [4, nt-1, nx, ny]
#   if 'Hstar_prox_fn' in fns_dict._fields:
#     v_next = fns_dict.Hstar_prox_fn(v_next_raw, 1/param, x_arr, t_arr)  # [4, nt-1, nx, ny]
#   elif 'H_prox_fn' in fns_dict._fields:
#     p_next = fns_dict.H_prox_fn(v_next_raw/param, param, x_arr, t_arr)  # [4, nt-1, nx, ny]
#     v_next = v_next_raw - param * p_next  # [4, nt-1, nx, ny]
#   else:
#     raise NotImplementedError
#   return v_next


# @jax.jit
# def update_primal_1d_prev(phi_prev, rho_prev, c_on_rho, v_prev, tau, dt, dspatial, fv, epsl):
#   vp_prev, vm_prev = v_prev[0], v_prev[1]
#   dx = dspatial[0]
#   eps = 1e-4
#   mp_prev = (rho_prev + eps) * vp_prev  # [nt-1, nx]
#   mm_prev = (rho_prev + eps) * vm_prev  # [nt-1, nx]
#   delta_phi = Dx_left_increasedim(mp_prev, dx) + Dx_right_increasedim(mm_prev, dx) \
#               + Dt_increasedim(rho_prev,dt) + epsl * Dxx_increasedim(rho_prev,dx) # [nt, nx]
#   delta_phi = jnp.concatenate([delta_phi[:-1,...], delta_phi[-1:,...] + c_on_rho/dt], axis = 0)
#   phi_next = phi_prev - tau * solver.Poisson_eqt_solver(delta_phi, fv, dt)
#   return phi_next

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

# @jax.jit
# def update_primal_2d(phi_prev, rho_prev, c_on_rho, v_prev, tau, dt, dspatial, fv, epsl):
#   vxp_prev, vxm_prev, vyp_prev, vym_prev = v_prev[0], v_prev[1], v_prev[2], v_prev[3]
#   dx, dy = dspatial[0], dspatial[1]
#   eps = 1e-4
#   mxp_prev = (rho_prev + eps) * vxp_prev  # [nt-1, nx]
#   mxm_prev = (rho_prev + eps) * vxm_prev  # [nt-1, nx]
#   myp_prev = (rho_prev + eps) * vyp_prev  # [nt-1, nx]
#   mym_prev = (rho_prev + eps) * vym_prev  # [nt-1, nx]
#   delta_phi = Dx_left_increasedim(mxp_prev, dx) + Dx_right_increasedim(mxm_prev, dx) \
#               + Dy_left_increasedim(myp_prev, dy) + Dy_right_increasedim(mym_prev, dy) \
#               + Dt_increasedim(rho_prev,dt) + epsl * Dxx_increasedim(rho_prev,dx) \
#               + epsl * Dyy_increasedim(rho_prev,dy)  # [nt, nx]
#   delta_phi = jnp.concatenate([delta_phi[:-1,...], delta_phi[-1:,...] + c_on_rho/dt], axis = 0)
#   phi_next = phi_prev - tau * solver.Poisson_eqt_solver_2d(delta_phi, fv, dt)
#   return phi_next


@partial(jax.jit, static_argnames=("fns_dict", "ndim"))
def update_dual_oneiter(phi_bar, rho_prev, c_on_rho, v_prev, sigma, dt, dspatial, epsl, x_arr, t_arr, 
                        fns_dict, ndim, fwd, precond, fv):
  if ndim == 1:
    # if 'opt_alp_fn' in fns_dict._fields:
    update_v = update_alp
    update_rho = update_rho_1d
    # else:  # old version as baseline
    #   update_v = update_v_1d_prev
    #   update_rho = update_rho_1d_prev
  # elif ndim == 2:
    # if 'Hxstar_plus_prox_fn' in fns_dict._fields and 'Hxstar_minus_prox_fn' in fns_dict._fields and \
    #    'Hystar_plus_prox_fn' in fns_dict._fields and 'Hystar_minus_prox_fn' in fns_dict._fields:
    #   update_v = update_v_2d_seperable
    # elif ('Hstar_prox_fn' in fns_dict._fields) or ('H_prox_fn' in fns_dict._fields):
    #   update_v = update_v_2d_nonseperable
    # else:
    #   print(fns_dict)
    #   raise "fns_dict must contain Hstar_prox_fn or H_prox_fn or Hxstar_plus_prox_fn, Hxstar_minus_prox_fn, Hystar_plus_prox_fn, Hystar_minus_prox_fn"
    # update_rho = update_rho_2d
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
  