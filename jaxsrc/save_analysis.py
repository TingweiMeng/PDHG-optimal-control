import pickle
import jax
import jax.numpy as jnp

def compute_HJ_residual_EO_1d_xdep(phi, dt, dx, f_in_H, c_in_H, epsl):
  '''
  H is c*|p| + f, 1-dimensional
  @parameters:
    phi: [nt, nx]
    dt: scalar
    dx: scalar
    f_in_H: [1, nx]
    c_in_H: [1, nx]
  @ return:
    HJ_residual: [nt-1, nx]
  '''
  dphidx_left = (phi - jnp.roll(phi, 1, axis = 1))/dx
  dphidx_right = (jnp.roll(phi, -1, axis=1) - phi)/dx
  H_val = jnp.maximum(-dphidx_right, 0) + jnp.maximum(dphidx_left, 0)
  H_val = c_in_H * H_val + f_in_H
  Lap = (dphidx_right - dphidx_left)/dx
  HJ_residual = (phi[1:,:] - phi[:-1,:])/dt + H_val[1:,:] - epsl * Lap[1:,:]
  return HJ_residual

def compute_HJ_residual_EO_2d_xdep(phi, dt, dx, dy, f_in_H, c_in_H, epsl):
  '''
  H is c*|p| + f, 2-dimensional
  @parameters:
    phi: [nt, nx, ny]
    dt: scalar
    dx, dy: scalar
    f_in_H: [1, nx, ny]
    c_in_H: [1, nx, ny]
    epsl: scalar, diffusion coefficient
  @ return:
    HJ_residual: [nt-1, nx, ny]
  '''
  dphidx_left = (phi - jnp.roll(phi, 1, axis = 1))/dx
  dphidx_right = (jnp.roll(phi, -1, axis=1) - phi)/dx
  dphidy_left = (phi - jnp.roll(phi, 1, axis = 2))/dy
  dphidy_right = (jnp.roll(phi, -1, axis=2) - phi)/dy
  H1_val = jnp.maximum(-dphidx_right, 0) + jnp.maximum(dphidx_left, 0)
  H2_val = jnp.maximum(-dphidy_right, 0) + jnp.maximum(dphidy_left, 0)
  H_val = c_in_H * (H1_val + H2_val) + f_in_H
  Lap = (dphidx_right - dphidx_left) / dx + (dphidy_right - dphidy_left) / dy
  HJ_residual = (phi[1:,...] - phi[:-1,...])/dt + H_val[1:,...] - epsl * Lap[1:,...]
  return HJ_residual
