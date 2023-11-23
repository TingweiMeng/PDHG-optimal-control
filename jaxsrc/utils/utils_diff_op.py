import jax
import jax.numpy as jnp

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