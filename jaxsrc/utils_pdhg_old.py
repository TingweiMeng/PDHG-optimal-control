import jax
import jax.numpy as jnp
from functools import partial
import os
import solver

jax.config.update("jax_enable_x64", True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

@partial(jax.jit, static_argnums=(2,))
def Dx_right_decreasedim(phi, dx, fwd = False):
  '''F phi = (phi_{k+1,i+1}-phi_{k+1,i})/dx
  phi_{k+1,i+1} is periodic in i+1. Can be also used for 2d spatial domain
  @ parameters:
    phi: [nt, nx] or [nt, nx, ny]
  @ return
    out: [nt-1, nx] or [nt-1, nx, ny]
  '''
  phi_ip1 = jnp.roll(phi, -1, axis=1)
  out = phi_ip1 - phi
  if fwd:
    out = out[:-1,...]/dx
  else:
    out = out[1:,...]/dx
  return out

@partial(jax.jit, static_argnums=(2,))
def Dx_right_increasedim(m, dx, fwd = False):
  '''F m = (-m[k-1,i] + m[k-1,i+1])/dx
  m[k,i+1] is periodic in i+1
  postpend 0 in axis-0 if fwd = True
  prepend 0 in axis-0 if fwd = False
  @ parameters:
    m: [nt-1, nx] or [nt-1, nx, ny]
  @ return
    out: [nt, nx] or [nt, nx, ny]
  '''
  m_ip1 = jnp.roll(m, -1, axis=1)
  out = -m + m_ip1
  out = out/dx
  if fwd: 
    out = jnp.concatenate([out, jnp.zeros_like(out[0:1,...])], axis = 0)
  else:
    out = jnp.concatenate([jnp.zeros_like(out[0:1,...]), out], axis = 0) #prepend 0
  return out

@partial(jax.jit, static_argnums=(2,))
def Dx_left_decreasedim(phi, dx, fwd = False):
  '''F phi = (phi_{k+1,i}-phi_{k+1,i-1})/dx
  phi_{k+1,i-1} is periodic in i+1
  @ parameters:
    phi: [nt, nx] or [nt, nx, ny]
  @ return
    out: [nt-1, nx] or [nt-1, nx, ny]
  '''
  phi_im1 = jnp.roll(phi, 1, axis=1)
  out = phi - phi_im1
  if fwd:
    out = out[:-1,...]/dx
  else:
    out = out[1:,...]/dx
  return out

@partial(jax.jit, static_argnums=(2,))
def Dx_left_increasedim(m, dx, fwd = False):
  '''F m = (-m[k,i-1] + m[k,i])/dx
  m[k,i-1] is periodic in i-1
  postpend 0 in axis-0 if fwd = True
  prepend 0 in axis-0 if fwd = False
  @ parameters:
    m: [nt-1, nx] or [nt-1, nx, ny]
  @ return
    out: [nt, nx] or [nt, nx, ny]
  '''
  m_im1 = jnp.roll(m, 1, axis=1)
  out = -m_im1 + m
  out = out/dx
  if fwd: 
    out = jnp.concatenate([out, jnp.zeros_like(out[0:1,...])], axis = 0) #postpend 0
  else:
    out = jnp.concatenate([jnp.zeros_like(out[0:1,...]), out], axis = 0) #prepend 0
  return out


@partial(jax.jit, static_argnums=(2,))
def Dy_right_decreasedim(phi, dy, fwd = False):
  '''F phi = (phi_{k+1,:,i+1}-phi_{k+1,:,i})/dy
  phi_{k+1,:,i+1} is periodic in i+1.
  @ parameters:
    phi: [nt, nx, ny]
  @ return
    out: [nt-1, nx, ny]
  '''
  phi_ip1 = jnp.roll(phi, -1, axis=2)
  out = phi_ip1 - phi
  if fwd:
    out = out[:-1,...]/dy
  else:
    out = out[1:,...]/dy
  return out

@partial(jax.jit, static_argnums=(2,))
def Dy_right_increasedim(m, dy, fwd = False):
  '''F m = (-m[k-1,:,i] + m[k-1,:,i+1])/dy
  m[k,:,i+1] is periodic in i+1
  postpend 0 in axis-0 if fwd = True
  prepend 0 in axis-0 if fwd = False
  @ parameters:
    m: [nt-1, nx, ny]
  @ return
    out: [nt, nx, ny]
  '''
  m_ip1 = jnp.roll(m, -1, axis=2)
  out = -m + m_ip1
  out = out/dy
  if fwd:
    out = jnp.concatenate([out, jnp.zeros_like(out[0:1,...])], axis = 0)
  else:
    out = jnp.concatenate([jnp.zeros_like(out[0:1,...]), out], axis = 0) #prepend 0
  return out

@partial(jax.jit, static_argnums=(2,))
def Dy_left_decreasedim(phi, dy, fwd = False):
  '''F phi = (phi_{k+1,:,i}-phi_{k+1,:,i-1})/dy
  phi_{k+1,:,i-1} is periodic in i+1
  @ parameters:
    phi: [nt, nx, ny]
  @ return
    out: [nt-1, nx, ny]
  '''
  phi_im1 = jnp.roll(phi, 1, axis=2)
  out = phi - phi_im1
  if fwd:
    out = out[:-1,...]/dy
  else:
    out = out[1:,...]/dy
  return out

@partial(jax.jit, static_argnums=(2,))
def Dy_left_increasedim(m, dy, fwd = False):
  '''F m = (-m[k,:,i-1] + m[k,:,i])/dy
  m[k,:,i-1] is periodic in i-1
  postpend 0 in axis-0 if fwd = True
  prepend 0 in axis-0 if fwd = False
  @ parameters:
    m: [nt-1, nx, ny]
  @ return
    out: [nt, nx, ny]
  '''
  m_im1 = jnp.roll(m, 1, axis=2)
  out = -m_im1 + m
  out = out/dy
  if fwd:
    out = jnp.concatenate([out, jnp.zeros_like(out[0:1,...])], axis = 0)
  else:
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

@partial(jax.jit, static_argnums=(2,))
def Dxx_decreasedim(phi, dx, fwd = False):
  '''Dxx phi = (phi_{k+1,i+1}+phi_{k+1,i-1}-2*phi_{k+1,i})/dx^2
  phi_{k+1,i} is periodic in i, but not in k
  @ parameters:
    phi: [nt, nx] or [nt, nx, ny]
  @ return
    out: [nt-1, nx] or [nt-1, nx, ny]
  '''
  if fwd:
    phi_kp1 = phi[:-1,...]
  else:
    phi_kp1 = phi[1:,:]
  phi_ip1 = jnp.roll(phi_kp1, -1, axis=1)
  phi_im1 = jnp.roll(phi_kp1, 1, axis=1)
  out = (phi_ip1 + phi_im1 - 2*phi_kp1)/dx**2
  return out

@partial(jax.jit, static_argnums=(2,))
def Dyy_decreasedim(phi, dy, fwd = False):
  '''Dxx phi = (phi_{k+1,:,i+1}+phi_{k+1,:,i-1}-2*phi_{k+1,:,i})/dy^2
  phi_{k+1,:,i} is periodic in i, but not in k
  @ parameters:
    phi: [nt, nx, ny]
  @ return
    out: [nt-1, nx, ny]
  '''
  if fwd:
    phi_kp1 = phi[:-1,...]
  else:
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

@partial(jax.jit, static_argnums=(2,))
def Dxx_increasedim(rho, dx, fwd = False):
  '''F rho = (rho[k-1,i+1]+rho[k-1,i-1]-2*rho[k-1,i])/dx^2
            #k = 0...(nt-1)
  prepend 0 in axis-0 if fwd = False
  postpend 0 in axis-0 if fwd = True
  @ parameters:
    rho: [nt-1, nx] or [nt-1, nx, ny]
  @ return
    out: [nt, nx] or [nt, nx, ny]
  '''
  if fwd:
    rho_km1 = jnp.concatenate([rho, jnp.zeros_like(rho[0:1,...])], axis = 0) #append 0
  else:
    rho_km1 = jnp.concatenate([jnp.zeros_like(rho[0:1,...]), rho], axis = 0) #prepend 0
  rho_im1 = jnp.roll(rho_km1, 1, axis=1)
  rho_ip1 = jnp.roll(rho_km1, -1, axis=1)
  out = (rho_ip1 + rho_im1 - 2*rho_km1) /dx**2
  return out

@partial(jax.jit, static_argnums=(2,))
def Dyy_increasedim(rho, dy, fwd = False):
  '''F rho = (rho[k-1,:,i+1]+rho[k-1,:,i-1]-2*rho[k-1,:,i])/dy^2
            #k = 0...(nt-1)
  prepend 0 in axis-0 if fwd = False
  postpend 0 in axis-0 if fwd = True
  @ parameters:
    rho: [nt-1, nx, ny]
  @ return
    out: [nt, nx, ny]
  '''
  if fwd:
    rho_km1 = jnp.concatenate([rho, jnp.zeros_like(rho[0:1,...])], axis = 0) #append 0
  else:
    rho_km1 = jnp.concatenate([jnp.zeros_like(rho[0:1,...]), rho], axis = 0) #prepend 0
  rho_im1 = jnp.roll(rho_km1, 1, axis=2)
  rho_ip1 = jnp.roll(rho_km1, -1, axis=2)
  out = (rho_ip1 + rho_im1 - 2*rho_km1) /dy**2
  return out


def HJ_residual(phi, alp, dspatial, dt, epsl, fns_dict, x_arr, t_arr, fwd):
  dx = dspatial[0]
  L_val = fns_dict.L_fn(alp, x_arr, t_arr)
  f_plus_val = fns_dict.f_plus_fn(alp, x_arr, t_arr)
  f_minus_val = fns_dict.f_minus_fn(alp, x_arr, t_arr)
  # NOTE: the old version switch Dx right and left
  vec = Dx_left_decreasedim(phi, dx, fwd=fwd) * f_minus_val + Dx_right_decreasedim(phi, dx, fwd=fwd) * f_plus_val
  vec = vec + Dt_decreasedim(phi, dt) - epsl * Dxx_decreasedim(phi, dx, fwd=fwd)  # [nt-1, nx]
  vec = vec + L_val
  return vec

def cont_residual(rho, alp, dspatial, dt, epsl, fns_dict, x_arr, t_arr, fwd, c_on_rho):
  dx = dspatial[0]
  fp = fns_dict.f_plus_fn(alp, x_arr, t_arr)
  fm = fns_dict.f_minus_fn(alp, x_arr, t_arr)
  # NOTE: the old version switch Dx right and left
  delta_phi = Dx_right_increasedim(fm * rho, dx, fwd=fwd) + Dx_left_increasedim(fp * rho, dx, fwd=fwd) \
              + epsl * Dxx_increasedim(alp, dx, fwd=fwd) # [nt, nx]
  # print('delta_phi: ', delta_phi)
  delta_phi += Dt_increasedim(rho,dt) # [nt, nx]
  # print('Dt: ', Dt_increasedim(rho_prev,dt))
  delta_phi = jnp.concatenate([delta_phi[:1,...] - c_on_rho/dt, delta_phi[1:-1,...], 0*delta_phi[-1:,...]], axis = 0)
  return delta_phi

def update_rho_1d(rho_prev, phi, alp, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr, fwd, precond, fv):
  vec = HJ_residual(phi, alp, dspatial, dt, epsl, fns_dict, x_arr, t_arr, fwd)
  # print('vec: ', vec)
  if precond and rho_prev.shape[0] > 1:
    rho_next = rho_prev - sigma * solver.Poisson_eqt_solver_termcond(vec, fv, dt)
  else:
    rho_next = rho_prev - sigma * vec
  
  # print('rho_next in update: ', rho_next, flush=True)
  rho_next = jnp.maximum(rho_next, 0.0)  # [nt-1, nx]
  return rho_next

def update_alp(alp_prev, phi, rho, sigma, dspatial, fns_dict, x_arr, t_arr):
  dx = dspatial[0]
  # Dx_phi_left = Dx_left_decreasedim(phi, dx)
  # Dx_phi_right = Dx_right_decreasedim(phi, dx)
  # NOTE: old version is to use right derivative, so switch the order here
  Dx_phi_left = Dx_right_decreasedim(phi, dx)
  Dx_phi_right = Dx_left_decreasedim(phi, dx)

  alp = fns_dict.opt_alp_fn(Dx_phi_left, Dx_phi_right, x_arr, t_arr, alp_prev, sigma, rho)
  return alp

# @partial(jax.jit, static_argnames=("fns_dict", "fwd",))
def update_primal_1d(phi_prev, rho_prev, c_on_rho, alp_prev, tau, dt, dspatial, fv, epsl, fwd, precond, fns_dict, x_arr, t_arr):
  delta_phi = cont_residual(rho_prev, alp_prev, dspatial, dt, epsl, fns_dict, x_arr, t_arr, fwd, c_on_rho)
  # print('delta_phi: ', delta_phi)
  if precond:
    phi_next = phi_prev - tau * solver.Poisson_eqt_solver_termcond(delta_phi, fv, dt)
  else:
    phi_next = phi_prev - tau * delta_phi
  return phi_next

# @partial(jax.jit, static_argnames=("fns_dict", "ndim", "fwd", ))
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
                   rho_v_iters=50, eps=1e-7, precond=False):
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
    if j == rho_v_iters - 1:
      print('rho_v_iters: ', rho_v_iters)
  return rho_next, v_next
  