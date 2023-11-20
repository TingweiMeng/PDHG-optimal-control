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


def compute_HJ_residual_1d(phi, v, dt, dspatial, fns_dict, epsl, x_arr, t_arr):
  vp, vm = v[0], v[1]
  dx = dspatial[0]
  if 'Hstar_plus_fn' in fns_dict._fields and 'Hstar_minus_fn' in fns_dict._fields:  # seperable case
    Hstar_val = fns_dict.Hstar_plus_fn(vm, x_arr, t_arr) + fns_dict.Hstar_minus_fn(vp, x_arr, t_arr)
  elif 'Hstar_fn' in fns_dict._fields:  # non-seperable case
    Hstar_val = fns_dict.Hstar_fn(jnp.stack([vp, vm], axis = 0), x_arr, t_arr)
  else:
    raise "fns_dict must contain Hstar_fn or Hxstar_plus_fn, Hxstar_minus_fn, Hystar_plus_fn, Hystar_minus_fn"
  vec = Dx_right_decreasedim(phi, dx) * vp + Dx_left_decreasedim(phi, dx) * vm
  vec = vec + Dt_decreasedim(phi, dt) - epsl * Dxx_decreasedim(phi, dx)  # [nt-1, nx]
  vec = vec - Hstar_val
  return vec

def compute_cont_residual(rho, v, dt, dspatial, c_on_rho, epsl):
  vp, vm = v[0], v[1]
  dx = dspatial[0]
  eps = 1e-4
  mp = (rho + eps) * vp  # [nt-1, nx]
  mm = (rho + eps) * vm  # [nt-1, nx]
  delta_phi = Dx_left_increasedim(mp, dx) + Dx_right_increasedim(mm, dx) \
              + Dt_increasedim(rho,dt) + epsl * Dxx_increasedim(rho,dx) # [nt, nx]
  delta_phi = jnp.concatenate([delta_phi[:-1,...], delta_phi[-1:,...] + c_on_rho/dt], axis = 0)
  return delta_phi


def update_rho_1d(rho_prev, phi, v, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr):
  vec = compute_HJ_residual_1d(phi, v, dt, dspatial, fns_dict, epsl, x_arr, t_arr)
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
  param = sigma / (rho + eps)
  vp_next_raw = vp_prev + param * Dx_right_decreasedim(phi, dx)  # [nt-1, nx]
  vm_next_raw = vm_prev + param * Dx_left_decreasedim(phi, dx)  # [nt-1, nx]
  if 'Hstar_plus_prox_fn' in fns_dict._fields and 'Hstar_minus_prox_fn' in fns_dict._fields:  # seperable case
    vp_next = fns_dict.Hstar_minus_prox_fn(vp_next_raw, param, x_arr, t_arr)  # [nt-1, nx]
    vm_next = fns_dict.Hstar_plus_prox_fn(vm_next_raw, param, x_arr, t_arr)  # [nt-1, nx]  
    v_next = (vp_next, vm_next)
  else:
    v_next_raw = jnp.stack([vp_next_raw, vm_next_raw], axis = 0)  # [2, nt-1, nx, ny] (xp, xm)
    if 'Hstar_prox_fn' in fns_dict._fields:
      v_next = fns_dict.Hstar_prox_fn(v_next_raw, 1/param, x_arr, t_arr)  # [2, nt-1, nx, ny]
    elif 'H_prox_fn' in fns_dict._fields:
      p_next = fns_dict.H_prox_fn(v_next_raw/param, param, x_arr, t_arr)  # [2, nt-1, nx, ny]
      v_next = v_next_raw - param * p_next  # [2, nt-1, nx, ny]
    else:
      raise NotImplementedError
  return v_next

def update_rho_2d(rho_prev, phi, v, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr):
  vxp, vxm = v[0], v[1]
  vyp, vym = v[2], v[3]
  dx, dy = dspatial[0], dspatial[1]
  if 'Hxstar_plus_fn' in fns_dict._fields and 'Hxstar_minus_fn' in fns_dict._fields and \
     'Hystar_plus_fn' in fns_dict._fields and 'Hystar_minus_fn' in fns_dict._fields:  # seperable case
    Hstar_val = fns_dict.Hxstar_plus_fn(vxm, x_arr, t_arr) + fns_dict.Hxstar_minus_fn(vxp, x_arr, t_arr) \
                + fns_dict.Hystar_plus_fn(vym, x_arr, t_arr) + fns_dict.Hystar_minus_fn(vyp, x_arr, t_arr)
  elif 'Hstar_fn' in fns_dict._fields:  # non-seperable case
    Hstar_val = fns_dict.Hstar_fn(jnp.stack([vxp, vxm, vyp, vym], axis = 0), x_arr, t_arr)
  else:
    raise "fns_dict must contain Hstar_fn or Hxstar_plus_fn, Hxstar_minus_fn, Hystar_plus_fn, Hystar_minus_fn"
  vec = Dx_right_decreasedim(phi, dx) * vxp + Dx_left_decreasedim(phi, dx) * vxm
  vec = vec + Dy_right_decreasedim(phi, dy) * vyp + Dy_left_decreasedim(phi, dy) * vym
  vec = vec + Dt_decreasedim(phi,dt) - epsl * Dxx_decreasedim(phi, dx) - epsl * Dyy_decreasedim(phi, dy)  # [nt-1, nx, ny]
  vec = vec - Hstar_val
  rho_next = rho_prev + sigma * vec
  rho_next = jnp.maximum(rho_next, 0.0)  # [nt-1, nx, ny]
  return rho_next

def update_v_2d_seperable(v_prev, phi, rho, sigma, dspatial, fns_dict, x_arr, t_arr, eps=1e-4):
  '''
  @ parameters:
    Hstar_plus_prox_fn and Hstar_minus_prox_fn are prox point operator taking (x,t) as input
    and output argmin_u H(u) + |x-u|^2/(2t), assuming H(p1, p2) = Hx(p1) + Hy(p2)
  '''
  vxp_prev, vxm_prev, vyp_prev, vym_prev = v_prev[0], v_prev[1], v_prev[2], v_prev[3]
  dx, dy = dspatial[0], dspatial[1]
  Hxstar_plus_prox_fn = fns_dict.Hxstar_plus_prox_fn
  Hxstar_minus_prox_fn = fns_dict.Hxstar_minus_prox_fn
  Hystar_plus_prox_fn = fns_dict.Hystar_plus_prox_fn
  Hystar_minus_prox_fn = fns_dict.Hystar_minus_prox_fn
  param = sigma / (rho + eps)
  vxp_next_raw = vxp_prev + param * Dx_right_decreasedim(phi, dx)  # [nt-1, nx, ny]
  vxm_next_raw = vxm_prev + param * Dx_left_decreasedim(phi, dx)  # [nt-1, nx, ny]
  vyp_next_raw = vyp_prev + param * Dy_right_decreasedim(phi, dy)  # [nt-1, nx, ny]
  vym_next_raw = vym_prev + param * Dy_left_decreasedim(phi, dy)  # [nt-1, nx, ny]
  vxp_next = Hxstar_minus_prox_fn(vxp_next_raw, param, x_arr, t_arr)  # [nt-1, nx, ny]
  vxm_next = Hxstar_plus_prox_fn(vxm_next_raw, param, x_arr, t_arr)  # [nt-1, nx, ny]  
  vyp_next = Hystar_minus_prox_fn(vyp_next_raw, param, x_arr, t_arr)  # [nt-1, nx, ny]
  vym_next = Hystar_plus_prox_fn(vym_next_raw, param, x_arr, t_arr)  # [nt-1, nx, ny]  
  return (vxp_next, vxm_next, vyp_next, vym_next)


def update_v_2d_nonseperable(v_prev, phi, rho, sigma, dspatial, fns_dict, x_arr, t_arr, eps=1e-4):
  '''
  @ parameters:
    Hstar_plus_prox_fn and Hstar_minus_prox_fn are prox point operator taking (x,t) as input
    and output argmin_u H(u) + |x-u|^2/(2t), assume H is general
  '''
  vxp_prev, vxm_prev, vyp_prev, vym_prev = v_prev[0], v_prev[1], v_prev[2], v_prev[3]
  dx, dy = dspatial[0], dspatial[1]
  param = sigma / (rho + eps)
  vxp_next_raw = vxp_prev + param * Dx_right_decreasedim(phi, dx)  # [nt-1, nx, ny]
  vxm_next_raw = vxm_prev + param * Dx_left_decreasedim(phi, dx)  # [nt-1, nx, ny]
  vyp_next_raw = vyp_prev + param * Dy_right_decreasedim(phi, dy)  # [nt-1, nx, ny]
  vym_next_raw = vym_prev + param * Dy_left_decreasedim(phi, dy)  # [nt-1, nx, ny]
  v_next_raw = jnp.stack([vxp_next_raw, vxm_next_raw, vyp_next_raw, vym_next_raw], axis = 0)  # [4, nt-1, nx, ny]
  if 'Hstar_prox_fn' in fns_dict._fields:
    v_next = fns_dict.Hstar_prox_fn(v_next_raw, 1/param, x_arr, t_arr)  # [4, nt-1, nx, ny]
  elif 'H_prox_fn' in fns_dict._fields:
    p_next = fns_dict.H_prox_fn(v_next_raw/param, param, x_arr, t_arr)  # [4, nt-1, nx, ny]
    v_next = v_next_raw - param * p_next  # [4, nt-1, nx, ny]
  else:
    raise NotImplementedError
  return v_next


@jax.jit
def update_primal_1d(phi_prev, rho_prev, c_on_rho, v_prev, tau, dt, dspatial, fv, epsl):
  delta_phi = compute_cont_residual(rho_prev, v_prev, dt, dspatial, c_on_rho, epsl)
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
    if 'Hxstar_plus_prox_fn' in fns_dict._fields and 'Hxstar_minus_prox_fn' in fns_dict._fields and \
       'Hystar_plus_prox_fn' in fns_dict._fields and 'Hystar_minus_prox_fn' in fns_dict._fields:
      update_v = update_v_2d_seperable
    elif ('Hstar_prox_fn' in fns_dict._fields) or ('H_prox_fn' in fns_dict._fields):
      update_v = update_v_2d_nonseperable
    else:
      print(fns_dict)
      raise "fns_dict must contain Hstar_prox_fn or H_prox_fn or Hxstar_plus_prox_fn, Hxstar_minus_prox_fn, Hystar_plus_prox_fn, Hystar_minus_prox_fn"
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
  fns_dict: dict of functions, see the function set_up_example_fns in solver.py
  '''
  for j in range(rho_v_iters):
    rho_next, v_next, err = update_dual_oneiter(phi_bar, rho_prev, c_on_rho, v_prev, sigma, dt, dspatial, epsl,
                                                              x_arr, t_arr, fns_dict, ndim)
    if err < eps:
      break
    rho_prev = rho_next
    v_prev = v_next
  return rho_next, v_next
  