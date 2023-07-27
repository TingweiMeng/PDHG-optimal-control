import pickle
import jax
import jax.numpy as jnp
import os

def compute_HJ_residual_EO_1d_general(phi, dt, dspatial, fns_dict, epsl, x_arr, t_arr):
  '''
  @parameters:
    phi: [nt, nx]
    dt: scalar
    dspatial: list of scalars
    fns_dict: constaining H_plus_fn, H_minus_fn: functions taking p([nt, nx]), x([1,nx]), t([nt-1,1]) and returning [nt, nx]
    epsl: scalar, diffusion coefficient
    x_arr, t_arr: [1, nx], [nt-1, 1]
  @ return:
    HJ_residual: [nt-1, nx]
  '''
  dx = dspatial[0]
  H_plus_fn = fns_dict.H_plus_fn
  H_minus_fn = fns_dict.H_minus_fn
  dphidx_left = (phi - jnp.roll(phi, 1, axis = 1))/dx
  dphidx_right = (jnp.roll(phi, -1, axis=1) - phi)/dx
  H_val = H_plus_fn(dphidx_left[1:,:], x_arr, t_arr) + H_minus_fn(dphidx_right[1:,:], x_arr, t_arr)  # [nt-1, nx]
  Lap = (dphidx_right - dphidx_left)/dx
  HJ_residual = (phi[1:,:] - phi[:-1,:])/dt + H_val - epsl * Lap[1:,:]
  return HJ_residual

def compute_HJ_residual_EO_2d_general(phi, dt, dspatial, fns_dict, epsl, x_arr, t_arr):
  '''
  H is c*|p| + f, 1-dimensional
  @parameters:
    phi: [nt, nx]
    dt: scalar
    dx: scalar
    H_plus, H_minus: functions taking [nt, nx] and returning [nt, nx]
    epsl: scalar, diffusion coefficient
    x_arr, t_arr: [1, nx], [nt-1, 1]
  @ return:
    HJ_residual: [nt-1, nx]
  '''
  dx, dy = dspatial[0], dspatial[1]
  Hx_plus_fn = fns_dict.Hx_plus_fn
  Hx_minus_fn = fns_dict.Hx_minus_fn
  Hy_plus_fn = fns_dict.Hy_plus_fn
  Hy_minus_fn = fns_dict.Hy_minus_fn
  dphidx_left = (phi - jnp.roll(phi, 1, axis = 1))/dx
  dphidx_right = (jnp.roll(phi, -1, axis=1) - phi)/dx
  dphidy_left = (phi - jnp.roll(phi, 1, axis = 2))/dy
  dphidy_right = (jnp.roll(phi, -1, axis=2) - phi)/dy
  H_val = Hx_plus_fn(dphidx_left[1:,...], x_arr, t_arr) + Hx_minus_fn(dphidx_right[1:,...], x_arr, t_arr)  # [nt-1, nx, ny]
  H_val += Hy_plus_fn(dphidy_left[1:,...], x_arr, t_arr) + Hy_minus_fn(dphidy_right[1:,...], x_arr, t_arr)  # [nt-1, nx, ny]
  Lap = (dphidx_right - dphidx_left)/dx + (dphidy_right - dphidy_left)/dy
  HJ_residual = (phi[1:,...] - phi[:-1,...])/dt + H_val - epsl * Lap[1:,...]
  return HJ_residual


def save(save_dir, filename, results):
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  filename_full = save_dir + '/{}.pickle'.format(filename)
  with open(filename_full, 'wb') as file:
    pickle.dump(results, file)
    print('saved to {}'.format(file), flush = True)