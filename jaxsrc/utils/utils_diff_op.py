import jax
import jax.numpy as jnp
from functools import partial

'''
bc: 0 for periodic, 1 for neumann, 2 for dirichlet
'''

def Dx_right_base(phi, dx, bc):
  '''out = (phi_{k,i+1}-phi_{k,i})/dx
  @ parameters:
    phi: [nt, nx] or [nt, nx, ny]
    bc: 0, 1, 2
  @ return
    out: [nt, nx] or [nt, nx, ny]
  '''
  if bc == 0:
    out = jnp.roll(phi, -1, axis=1) - phi
  elif bc == 1:
    out = jnp.concatenate([phi[:,1:]-phi[:,:-1], jnp.zeros_like(phi[:,0:1])], axis = 1)
  elif bc == 2:
    out = jnp.concatenate([phi[:,1:], jnp.zeros_like(phi[:,0:1])], axis = 1) - phi
  return out/dx

@partial(jax.jit, static_argnames=('bc',))
def Dx_right_decreasedim(phi, dx, bc):
  '''out = (phi_{k+1,i+1}-phi_{k+1,i})/dx
  @ parameters:
    phi: [nt, nx] or [nt, nx, ny]
    bc: 0, 1, 2
  @ return
    out: [nt-1, nx] or [nt-1, nx, ny]
  '''
  out = Dx_right_base(phi, dx, bc)
  return out[1:,...]

@partial(jax.jit, static_argnames=('bc',))
def Dx_right_increasedim(m, dx, bc):
  '''F m = (-m[k-1,i] + m[k-1,i+1])/dx
  prepend 0 in axis-0
  @ parameters:
    m: [nt-1, nx] or [nt-1, nx, ny]
    bc: 0, 1, 2
  @ return
    out: [nt, nx] or [nt, nx, ny]
  '''
  out = Dx_right_base(m, dx, bc)  # [nt-1, nx] or [nt-1, nx, ny]
  out = jnp.concatenate([jnp.zeros_like(out[0:1,...]), out], axis = 0) #prepend 0
  return out

def Dx_left_base(phi, dx, bc):
  '''out = (phi_{k,i}-phi_{k,i-1})/dx
  @ parameters:
    phi: [nt, nx] or [nt, nx, ny]
    bc: 0, 1, 2
  @ return
    out: [nt, nx] or [nt, nx, ny]
  '''
  if bc == 0:
    out = phi - jnp.roll(phi, 1, axis=1)
  elif bc == 1:
    out = jnp.concatenate([jnp.zeros_like(phi[:,0:1]), phi[:,1:]-phi[:,:-1]], axis = 1)
  elif bc == 2:
    out = phi - jnp.concatenate([jnp.zeros_like(phi[:,0:1]), phi[:,:-1]], axis = 1)
  return out/dx

@partial(jax.jit, static_argnames=('bc',))
def Dx_left_decreasedim(phi, dx, bc):
  '''F phi = (phi_{k+1,i}-phi_{k+1,i-1})/dx
  @ parameters:
    phi: [nt, nx] or [nt, nx, ny]
    bc: 0, 1, 2
  @ return
    out: [nt-1, nx] or [nt-1, nx, ny]
  '''
  out = Dx_left_base(phi, dx, bc)
  return out[1:,...]

@partial(jax.jit, static_argnames=('bc',))
def Dx_left_increasedim(m, dx, bc):
  '''F m = (-m[k-1,i-1] + m[k-1,i])/dx
  prepend 0 in axis-0
  @ parameters:
    m: [nt-1, nx] or [nt-1, nx, ny]
    bc: 0, 1, 2
  @ return
    out: [nt, nx] or [nt, nx, ny]
  '''
  out = Dx_left_base(m, dx, bc)  # [nt-1, nx] or [nt-1, nx, ny]
  out = jnp.concatenate([jnp.zeros_like(out[0:1,...]), out], axis = 0) #prepend 0
  return out

def Dy_right_base(phi, dy, bc):
  '''out = (phi_{k,i+1}-phi_{k,i})/dy
  @ parameters:
    phi: [nt, nx, ny]
    bc: 0, 1, 2
  @ return
    out: [nt, nx, ny]
  '''
  if bc == 0:
    out = jnp.roll(phi, -1, axis=2) - phi
  elif bc == 1:
    out = jnp.concatenate([phi[:,:,1:]-phi[:,:,:-1], jnp.zeros_like(phi[:,:,0:1])], axis = 2)
  elif bc == 2:
    out = jnp.concatenate([phi[:,:,1:], jnp.zeros_like(phi[:,:,0:1])], axis = 2) - phi
  return out/dy

@partial(jax.jit, static_argnames=('bc',))
def Dy_right_decreasedim(phi, dy, bc):
  '''F phi = (phi_{k+1,:,i+1}-phi_{k+1,:,i})/dy
  @ parameters:
    phi: [nt, nx, ny]
    bc: 0, 1, 2
  @ return
    out: [nt-1, nx, ny]
  '''
  out = Dy_right_base(phi, dy, bc)
  return out[1:,...]

@partial(jax.jit, static_argnames=('bc',))
def Dy_right_increasedim(m, dy, bc):
  '''F m = (-m[k-1,:,i] + m[k-1,:,i+1])/dy
  prepend 0 in axis-0
  @ parameters:
    m: [nt-1, nx, ny]
    bc: 0, 1, 2
  @ return
    out: [nt, nx, ny]
  '''
  out = Dy_right_base(m, dy, bc)  # [nt-1, nx, ny]
  out = jnp.concatenate([jnp.zeros_like(out[0:1,...]), out], axis = 0) #prepend 0
  return out

def Dy_left_base(phi, dy, bc):
  '''out = (phi_{k,i}-phi_{k,i-1})/dy
  @ parameters:
    phi: [nt, nx, ny]
    bc: 0, 1, 2
  @ return
    out: [nt, nx, ny]
  '''
  if bc == 0:
    out = phi - jnp.roll(phi, 1, axis=2)
  elif bc == 1:
    out = jnp.concatenate([jnp.zeros_like(phi[:,:,0:1]), phi[:,:,1:]-phi[:,:,:-1]], axis = 2)
  elif bc == 2:
    out = phi - jnp.concatenate([jnp.zeros_like(phi[:,:,0:1]), phi[:,:,:-1]], axis = 2)
  return out/dy

@partial(jax.jit, static_argnames=('bc',))
def Dy_left_decreasedim(phi, dy, bc):
  '''F phi = (phi_{k+1,:,i}-phi_{k+1,:,i-1})/dy
  phi_{k+1,:,i-1} is periodic in i+1
  @ parameters:
    phi: [nt, nx, ny]
    bc: 0, 1, 2
  @ return
    out: [nt-1, nx, ny]
  '''
  out = Dy_left_base(phi, dy, bc)
  return out[1:,...]

@partial(jax.jit, static_argnames=('bc',))
def Dy_left_increasedim(m, dy, bc):
  '''F m = (-m[k-1,:,i-1] + m[k-1,:,i])/dy
  prepend 0 in axis-0
  @ parameters:
    m: [nt-1, nx, ny]
    bc: 0, 1, 2
  @ return
    out: [nt, nx, ny]
  '''
  out = Dy_left_base(m, dy, bc)  # [nt-1, nx, ny]
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
  out = (rho_k - rho_km1) /dt
  return out

def Dxx_base(phi, dx, bc):
  '''Dxx phi = (phi_{k,i+1}+phi_{k,i-1}-2*phi_{k,i})/dx^2
  @ parameters:
    phi: [nt, nx] or [nt, nx, ny]
    bc: 0, 1, 2
  @ return
    out: [nt, nx] or [nt, nx, ny]
  '''
  if bc == 0:
    phi_ip1 = jnp.roll(phi, -1, axis=1)
    phi_im1 = jnp.roll(phi, 1, axis=1)
  elif bc == 1:
    phi_ip1 = jnp.concatenate([phi[:,1:], phi[:,-1:]], axis = 1)
    phi_im1 = jnp.concatenate([phi[:,0:1], phi[:,:-1]], axis = 1)
  elif bc == 2:
    phi_ip1 = jnp.concatenate([phi[:,1:], jnp.zeros_like(phi[:,0:1])], axis = 1)
    phi_im1 = jnp.concatenate([jnp.zeros_like(phi[:,0:1]), phi[:,:-1]], axis = 1)
  out = (phi_ip1 + phi_im1 - 2*phi)/dx**2
  return out

@partial(jax.jit, static_argnames=('bc',))
def Dxx_decreasedim(phi, dx, bc):
  '''Dxx phi = (phi_{k+1,i+1}+phi_{k+1,i-1}-2*phi_{k+1,i})/dx^2
  @ parameters:
    phi: [nt, nx] or [nt, nx, ny]
    bc: 0, 1, 2
  @ return
    out: [nt-1, nx] or [nt-1, nx, ny]
  '''
  out = Dxx_base(phi, dx, bc)
  return out[1:,...]


@partial(jax.jit, static_argnames=('bc',))
def Dxx_increasedim(rho, dx, bc):
  '''F rho = (rho[k-1,i+1]+rho[k-1,i-1]-2*rho[k-1,i])/dx^2
  prepend 0 in axis-0
  @ parameters:
    rho: [nt-1, nx] or [nt-1, nx, ny]
    bc: 0, 1, 2 (bc in x)
  @ return
    out: [nt, nx] or [nt, nx, ny]
  '''
  out = Dxx_base(rho, dx, bc)  # [nt-1, nx] or [nt-1, nx, ny]
  out = jnp.concatenate([jnp.zeros_like(out[0:1,...]), out], axis = 0)
  return out

def Dyy_base(phi, dy, bc):
  '''Dyy phi = (phi_{k,i,j+1}+phi_{k,i,j-1}-2*phi_{k,i,j})/dy^2
  @ parameters:
    phi: [nt, nx, ny]
    bc: 0, 1, 2 (bc in y)
  @ return
    out: [nt, nx, ny]
  '''
  if bc == 0:
    phi_ip1 = jnp.roll(phi, -1, axis=2)
    phi_im1 = jnp.roll(phi, 1, axis=2)
  elif bc == 1:
    phi_ip1 = jnp.concatenate([phi[:,:,1:], phi[:,:,-1:]], axis = 2)
    phi_im1 = jnp.concatenate([phi[:,:,0:1], phi[:,:,:-1]], axis = 2)
  elif bc == 2:
    phi_ip1 = jnp.concatenate([phi[:,:,1:], jnp.zeros_like(phi[:,:,0:1])], axis = 2)
    phi_im1 = jnp.concatenate([jnp.zeros_like(phi[:,:,0:1]), phi[:,:,:-1]], axis = 2)
  out = (phi_ip1 + phi_im1 - 2*phi)/dy**2
  return out

@partial(jax.jit, static_argnames=('bc',))
def Dyy_decreasedim(phi, dy, bc):
  '''Dxx phi = (phi_{k+1,:,j+1}+phi_{k+1,:,j-1}-2*phi_{k+1,:,j})/dy^2
  @ parameters:
    phi: [nt, nx, ny]
    bc: 0, 1, 2 (bc in y)
  @ return
    out: [nt-1, nx, ny]
  '''
  out = Dyy_base(phi, dy, bc)
  return out[1:,...]

@partial(jax.jit, static_argnames=('bc',))
def Dyy_increasedim(rho, dy, bc):
  '''F rho = (rho[k-1,:,j+1]+rho[k-1,:,j-1]-2*rho[k-1,:,j])/dy^2
  prepend 0 in axis-0
  @ parameters:
    rho: [nt-1, nx, ny]
    bc: 0, 1, 2 (bc in y)
  @ return
    out: [nt, nx, ny]
  '''
  out = Dyy_base(rho, dy, bc)  # [nt-1, nx, ny]
  out = jnp.concatenate([jnp.zeros_like(out[0:1,...]), out], axis = 0)
  return out