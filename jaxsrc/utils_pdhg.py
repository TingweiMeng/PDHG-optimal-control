import jax
import jax.numpy as jnp
from functools import partial
import os
import solver

jax.config.update("jax_enable_x64", True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

@jax.jit
def Dx_right_decreasedim(phi, dx):
  '''F phi = (phi_{k+1,i+1}-phi_{k+1,i})/dx
  phi_{k+1,i+1} is periodic in i+1. Can be also used for 2d spatial domain
  @ parameters:
    phi: [nt, nx] or [nt, nx, ny]
  @ return
    out: [nt-1, nx] or [nt-1, nx, ny]
  '''
  phi_ip1 = jnp.roll(phi, -1, axis=1)
  out = phi_ip1 - phi
  out = out[1:,...]/dx
  return out

@jax.jit
def Dx_right_increasedim(m, dx):
  '''F m = (-m[k-1,i] + m[k-1,i+1])/dx
  m[k,i+1] is periodic in i+1
  prepend 0 in axis-0
  @ parameters:
    m: [nt-1, nx] or [nt-1, nx, ny]
  @ return
    out: [nt, nx] or [nt, nx, ny]
  '''
  m_ip1 = jnp.roll(m, -1, axis=1)
  out = -m + m_ip1
  out = out/dx
  out = jnp.concatenate([jnp.zeros_like(out[0:1,...]), out], axis = 0) #prepend 0
  return out

@jax.jit
def Dx_left_decreasedim(phi, dx):
  '''F phi = (phi_{k+1,i}-phi_{k+1,i-1})/dx
  phi_{k+1,i-1} is periodic in i+1
  @ parameters:
    phi: [nt, nx] or [nt, nx, ny]
  @ return
    out: [nt-1, nx] or [nt-1, nx, ny]
  '''
  phi_im1 = jnp.roll(phi, 1, axis=1)
  out = phi - phi_im1
  out = out[1:,...]/dx
  return out

@jax.jit
def Dx_left_increasedim(m, dx):
  '''F m = (-m[k,i-1] + m[k,i])/dx
  m[k,i-1] is periodic in i-1
  prepend 0 in axis-0
  @ parameters:
    m: [nt-1, nx] or [nt-1, nx, ny]
  @ return
    out: [nt, nx] or [nt, nx, ny]
  '''
  m_im1 = jnp.roll(m, 1, axis=1)
  out = -m_im1 + m
  out = out/dx
  out = jnp.concatenate([jnp.zeros_like(out[0:1,...]), out], axis = 0) #prepend 0
  return out


@jax.jit
def Dy_right_decreasedim(phi, dy):
  '''F phi = (phi_{k+1,:,i+1}-phi_{k+1,:,i})/dy
  phi_{k+1,:,i+1} is periodic in i+1.
  @ parameters:
    phi: [nt, nx, ny]
  @ return
    out: [nt-1, nx, ny]
  '''
  phi_ip1 = jnp.roll(phi, -1, axis=2)
  out = phi_ip1 - phi
  out = out[1:,...]/dy
  return out

@jax.jit
def Dy_right_increasedim(m, dy):
  '''F m = (-m[k-1,:,i] + m[k-1,:,i+1])/dy
  m[k,:,i+1] is periodic in i+1
  prepend 0 in axis-0
  @ parameters:
    m: [nt-1, nx, ny]
  @ return
    out: [nt, nx, ny]
  '''
  m_ip1 = jnp.roll(m, -1, axis=2)
  out = -m + m_ip1
  out = out/dy
  out = jnp.concatenate([jnp.zeros_like(out[0:1,...]), out], axis = 0) #prepend 0
  return out

@jax.jit
def Dy_left_decreasedim(phi, dy):
  '''F phi = (phi_{k+1,:,i}-phi_{k+1,:,i-1})/dy
  phi_{k+1,:,i-1} is periodic in i+1
  @ parameters:
    phi: [nt, nx, ny]
  @ return
    out: [nt-1, nx, ny]
  '''
  phi_im1 = jnp.roll(phi, 1, axis=2)
  out = phi - phi_im1
  out = out[1:,...]/dy
  return out

@jax.jit
def Dy_left_increasedim(m, dy):
  '''F m = (-m[k,:,i-1] + m[k,:,i])/dy
  m[k,:,i-1] is periodic in i-1
  prepend 0 in axis-0
  @ parameters:
    m: [nt-1, nx, ny]
  @ return
    out: [nt, nx, ny]
  '''
  m_im1 = jnp.roll(m, 1, axis=2)
  out = -m_im1 + m
  out = out/dy
  out = jnp.concatenate([jnp.zeros_like(out[0:1,...]), out], axis = 0) #prepend 0
  return out


@jax.jit
def Dt_decreasedim(phi, dt):
  '''Dt phi = (phi_{k+1,...}-phi_{k,...})/dt
  phi_{k+1,...} is not periodic
  @ parameters:
    phi: [nt, nx] or [nt, nx, ny]
  @ return
    out: [nt-1, nx] or [nt-1, nx, ny]
  '''
  phi_kp1 = phi[1:,...]
  phi_k = phi[:-1,...]
  out = (phi_kp1 - phi_k) /dt
  return out

@jax.jit
def Dxx_decreasedim(phi, dx):
  '''Dxx phi = (phi_{k+1,i+1}+phi_{k+1,i-1}-2*phi_{k+1,i})/dx^2
  phi_{k+1,i} is periodic in i, but not in k
  @ parameters:
    phi: [nt, nx] or [nt, nx, ny]
  @ return
    out: [nt-1, nx] or [nt-1, nx, ny]
  '''
  phi_kp1 = phi[1:,:]
  phi_ip1 = jnp.roll(phi_kp1, -1, axis=1)
  phi_im1 = jnp.roll(phi_kp1, 1, axis=1)
  out = (phi_ip1 + phi_im1 - 2*phi_kp1)/dx**2
  return out

@jax.jit
def Dyy_decreasedim(phi, dy):
  '''Dxx phi = (phi_{k+1,:,i+1}+phi_{k+1,:,i-1}-2*phi_{k+1,:,i})/dy^2
  phi_{k+1,:,i} is periodic in i, but not in k
  @ parameters:
    phi: [nt, nx, ny]
  @ return
    out: [nt-1, nx, ny]
  '''
  phi_kp1 = phi[1:,...]
  phi_ip1 = jnp.roll(phi_kp1, -1, axis=2)
  phi_im1 = jnp.roll(phi_kp1, 1, axis=2)
  out = (phi_ip1 + phi_im1 - 2*phi_kp1)/dy**2
  return out


@jax.jit
def Dt_increasedim(rho, dt):
  '''Dt rho = (-rho[k-1,...] + rho[k,...])/dt
            #k = 0...(nt-1)
  rho[-1,:] = 0
  @ parameters:
    rho: [nt-1, nx] or [nt-1, nx, ny]
  @ return
    out: [nt, nx] or [nt, nx, ny]
  '''
  rho_km1 = jnp.concatenate([jnp.zeros_like(rho[0:1,...]), rho], axis = 0) #prepend 0
  rho_k = jnp.concatenate([rho, jnp.zeros_like(rho[0:1,...])], axis = 0) #append 0
  out = (-rho_km1 + rho_k)/dt
  return out

@jax.jit
def Dxx_increasedim(rho, dx):
  '''F rho = (rho[k-1,i+1]+rho[k-1,i-1]-2*rho[k-1,i])/dx^2
            #k = 0...(nt-1)
  rho[-1,:] = 0
  @ parameters:
    rho: [nt-1, nx] or [nt-1, nx, ny]
  @ return
    out: [nt, nx] or [nt, nx, ny]
  '''
  rho_km1 = jnp.concatenate([jnp.zeros_like(rho[0:1,...]), rho], axis = 0) #prepend 0
  rho_im1 = jnp.roll(rho_km1, 1, axis=1)
  rho_ip1 = jnp.roll(rho_km1, -1, axis=1)
  out = (rho_ip1 + rho_im1 - 2*rho_km1) /dx**2
  return out

@jax.jit
def Dyy_increasedim(rho, dy):
  '''F rho = (rho[k-1,:,i+1]+rho[k-1,:,i-1]-2*rho[k-1,:,i])/dy^2
            #k = 0...(nt-1)
  rho[-1,:] = 0
  @ parameters:
    rho: [nt-1, nx, ny]
  @ return
    out: [nt, nx, ny]
  '''
  rho_km1 = jnp.concatenate([jnp.zeros_like(rho[0:1,...]), rho], axis = 0) #prepend 0
  rho_im1 = jnp.roll(rho_km1, 1, axis=2)
  rho_ip1 = jnp.roll(rho_km1, -1, axis=2)
  out = (rho_ip1 + rho_im1 - 2*rho_km1) /dy**2
  return out


def compute_HJ_residual_1d(phi, alp, dt, dspatial, fns_dict, epsl, x_arr, t_arr):
  alp1, alp2 = alp
  dx = dspatial[0]
  L_val = fns_dict.L1_fn(alp1, x_arr, t_arr) + fns_dict.L2_fn(alp2, x_arr, t_arr)
  f_plus = fns_dict.f_plus_fn(alp1, x_arr, t_arr)
  f_minus = fns_dict.f_minus_fn(alp2, x_arr, t_arr)
  vec = Dx_right_decreasedim(phi, dx) * f_plus + Dx_left_decreasedim(phi, dx) * f_minus
  vec = - vec + Dt_decreasedim(phi, dt) - epsl * Dxx_decreasedim(phi, dx)  # [nt-1, nx]
  vec = vec - L_val
  return vec

def compute_HJ_residual_2d(phi, alp, dt, dspatial, fns_dict, epsl, x_arr, t_arr):
  alp1_x, alp2_x, alp1_y, alp2_y = alp
  dx, dy = dspatial
  L_val = fns_dict.L1_x_fn(alp1_x, x_arr, t_arr) + fns_dict.L2_x_fn(alp2_x, x_arr, t_arr)
  L_val = L_val + fns_dict.L1_y_fn(alp1_y, x_arr, t_arr) + fns_dict.L2_y_fn(alp2_y, x_arr, t_arr)
  f_plus_x = fns_dict.f_plus_x_fn(alp1_x, x_arr, t_arr)
  f_minus_x = fns_dict.f_minus_x_fn(alp2_x, x_arr, t_arr)
  f_plus_y = fns_dict.f_plus_y_fn(alp1_y, x_arr, t_arr)
  f_minus_y = fns_dict.f_minus_y_fn(alp2_y, x_arr, t_arr)
  vec = Dx_right_decreasedim(phi, dx) * f_plus_x + Dx_left_decreasedim(phi, dx) * f_minus_x
  vec = vec + Dy_right_decreasedim(phi, dy) * f_plus_y + Dy_left_decreasedim(phi, dy) * f_minus_y
  vec = - vec + Dt_decreasedim(phi, dt) - epsl * Dxx_decreasedim(phi, dx)  # [nt-1, nx]
  vec = vec - L_val
  return vec

def compute_cont_residual_1d(rho, alp, dt, dspatial, fns_dict, c_on_rho, epsl, x_arr, t_arr):
  alp1, alp2 = alp
  dx = dspatial[0]
  eps = 1e-4
  f_plus = fns_dict.f_plus_fn(alp1, x_arr, t_arr)
  f_minus = fns_dict.f_minus_fn(alp2, x_arr, t_arr)
  m1 = (rho + eps) * f_plus  # [nt-1, nx]
  m2 = (rho + eps) * f_minus  # [nt-1, nx]
  delta_phi = -Dx_left_increasedim(m1, dx) + -Dx_right_increasedim(m2, dx) \
              + Dt_increasedim(rho,dt) + epsl * Dxx_increasedim(rho,dx) # [nt, nx]
  delta_phi = jnp.concatenate([delta_phi[:-1,...], delta_phi[-1:,...] + c_on_rho/dt], axis = 0)
  return delta_phi

def compute_cont_residual_2d(rho, alp, dt, dspatial, fns_dict, c_on_rho, epsl, x_arr, t_arr):
  alp1_x, alp2_x, alp1_y, alp2_y = alp
  dx, dy = dspatial
  eps = 1e-4
  f_plus_x = fns_dict.f_plus_x_fn(alp1_x, x_arr, t_arr)
  f_minus_x = fns_dict.f_minus_x_fn(alp2_x, x_arr, t_arr)
  f_plus_y = fns_dict.f_plus_y_fn(alp1_y, x_arr, t_arr)
  f_minus_y = fns_dict.f_minus_y_fn(alp2_y, x_arr, t_arr)
  m1_x = (rho + eps) * f_plus_x  # [nt-1, nx, ny]
  m2_x = (rho + eps) * f_minus_x  # [nt-1, nx, ny]
  m1_y = (rho + eps) * f_plus_y  # [nt-1, nx, ny]
  m2_y = (rho + eps) * f_minus_y  # [nt-1, nx, ny]
  delta_phi = - Dx_left_increasedim(m1_x, dx) - Dx_right_increasedim(m2_x, dx) \
              - Dy_left_increasedim(m1_y, dy) - Dy_right_increasedim(m2_y, dy) \
              + Dt_increasedim(rho,dt) + epsl * Dxx_increasedim(rho,dx) # [nt, nx]
  delta_phi = jnp.concatenate([delta_phi[:-1,...], delta_phi[-1:,...] + c_on_rho/dt], axis = 0)
  return delta_phi


def update_rho_1d(rho_prev, phi, alp, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr):
  vec = compute_HJ_residual_1d(phi, alp, dt, dspatial, fns_dict, epsl, x_arr, t_arr)
  rho_next = rho_prev + sigma * vec
  rho_next = jnp.maximum(rho_next, 0.0)  # [nt-1, nx]
  return rho_next

def update_alp_1d(alp_prev, phi, rho, sigma, dspatial, fns_dict, x_arr, t_arr, eps=1e-4):
  '''
  @ parameters:
    Hstar_plus_prox_fn and Hstar_minus_prox_fn are prox point operator taking (x,t) as input
    and output argmin_u H(u) + |x-u|^2/(2t)
  '''
  dx = dspatial[0]
  Dx_right_phi = Dx_right_decreasedim(phi, dx)  # [nt-1, nx]
  Dx_left_phi = Dx_left_decreasedim(phi, dx)  # [nt-1, nx]
  if 'alp_update_fn' in fns_dict._fields:
    alp_next = fns_dict.alp_update_fn(alp_prev, Dx_right_phi, Dx_left_phi, rho, sigma, x_arr, t_arr)
  else:
    raise NotImplementedError
  return alp_next

def update_rho_2d(rho_prev, phi, alp, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr):
  vec = compute_HJ_residual_2d(phi, alp, dt, dspatial, fns_dict, epsl, x_arr, t_arr)
  rho_next = rho_prev + sigma * vec
  rho_next = jnp.maximum(rho_next, 0.0)  # [nt-1, nx]
  return rho_next

def update_alp_2d(alp_prev, phi, rho, sigma, dspatial, fns_dict, x_arr, t_arr, eps=1e-4):
  '''
  @ parameters:
    Hstar_plus_prox_fn and Hstar_minus_prox_fn are prox point operator taking (x,t) as input
    and output argmin_u H(u) + |x-u|^2/(2t)
  '''
  dx, dy = dspatial
  Dx_right_phi = Dx_right_decreasedim(phi, dx)  # [nt-1, nx, ny]
  Dx_left_phi = Dx_left_decreasedim(phi, dx)  # [nt-1, nx, ny]
  Dy_right_phi = Dy_right_decreasedim(phi, dy)  # [nt-1, nx, ny]
  Dy_left_phi = Dy_left_decreasedim(phi, dy)  # [nt-1, nx, ny]
  if 'alp_update_fn' in fns_dict._fields:
    alp_next = fns_dict.alp_update_fn(alp_prev, Dx_right_phi, Dx_left_phi, Dy_right_phi, Dy_left_phi, rho, sigma, x_arr, t_arr)
  else:
    raise NotImplementedError
  return alp_next


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


@partial(jax.jit, static_argnames=("fns_dict",))
def update_primal_1d(phi_prev, rho_prev, c_on_rho, alp_prev, tau, dt, dspatial, fns_dict, fv, epsl, x_arr, t_arr):
  delta_phi = compute_cont_residual_1d(rho_prev, alp_prev, dt, dspatial, fns_dict, c_on_rho, epsl, x_arr, t_arr)
  C = 1.0
  phi_next = phi_prev + tau * solver.Poisson_eqt_solver_1d(delta_phi, fv, dt, C = C)
  return phi_next

@partial(jax.jit, static_argnames=("fns_dict",))
def update_primal_2d(phi_prev, rho_prev, c_on_rho, alp_prev, tau, dt, dspatial, fns_dict, fv, epsl, x_arr, t_arr):
  delta_phi = compute_cont_residual_2d(rho_prev, alp_prev, dt, dspatial, fns_dict, c_on_rho, epsl, x_arr, t_arr)
  C = 1.0
  phi_next = phi_prev + tau * solver.Poisson_eqt_solver_2d(delta_phi, fv, dt, C = C)
  return phi_next


@partial(jax.jit, static_argnames=("fns_dict", "ndim"))
def update_dual_oneiter(phi_bar, rho_prev, c_on_rho, alp_prev, sigma, dt, dspatial, epsl, x_arr, t_arr, fns_dict, ndim):
  if ndim == 1:
    update_alp = update_alp_1d
    update_rho = update_rho_1d
  elif ndim == 2:
    update_alp = update_alp_2d
    # if 'Hxstar_plus_prox_fn' in fns_dict._fields and 'Hxstar_minus_prox_fn' in fns_dict._fields and \
    #    'Hystar_plus_prox_fn' in fns_dict._fields and 'Hystar_minus_prox_fn' in fns_dict._fields:
    #   update_v = update_v_2d_seperable
    # elif ('Hstar_prox_fn' in fns_dict._fields) or ('H_prox_fn' in fns_dict._fields):
    #   update_v = update_v_2d_nonseperable
    # else:
    #   print(fns_dict)
    #   raise "fns_dict must contain Hstar_prox_fn or H_prox_fn or Hxstar_plus_prox_fn, Hxstar_minus_prox_fn, Hystar_plus_prox_fn, Hystar_minus_prox_fn"
    update_rho = update_rho_2d
  else:
    raise NotImplementedError
  alp_next = update_alp(alp_prev, phi_bar, rho_prev, sigma, dspatial, fns_dict, x_arr, t_arr)
  rho_next = update_rho(rho_prev, phi_bar, alp_next, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr)
  err = jnp.linalg.norm(rho_next - rho_prev) / jnp.maximum(jnp.linalg.norm(rho_prev), 1.0)
  for alp_p, alp_n in zip(alp_prev, alp_next):
    err = jnp.maximum(err, jnp.linalg.norm(alp_n - alp_p) / jnp.maximum(jnp.linalg.norm(alp_p), 1.0))
  return rho_next, alp_next, err


def update_dual(phi_bar, rho_prev, c_on_rho, alp_prev, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr, ndim,
                   rho_alp_iters=10, eps=1e-7):
  '''
  @ parameters:
  fns_dict: dict of functions, see the function set_up_example_fns in solver.py
  '''
  for j in range(rho_alp_iters):
    rho_next, alp_next, err = update_dual_oneiter(phi_bar, rho_prev, c_on_rho, alp_prev, sigma, dt, dspatial, epsl,
                                                              x_arr, t_arr, fns_dict, ndim)
    if err < eps:
      break
    rho_prev = rho_next
    alp_prev = alp_next
  return rho_next, alp_next
  