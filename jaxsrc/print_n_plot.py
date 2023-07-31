import jax
import jax.numpy as jnp
from einshape import jax_einshape as einshape
import utils
import numpy as np
from absl import app, flags, logging
import pickle
import matplotlib.pyplot as plt
import os
from solver import set_up_example_fns
from save_analysis import compute_HJ_residual_EO_1d_general, compute_HJ_residual_EO_2d_general, save
from functools import partial

def compute_xarr(ndim, n_spatial, period_spatial):
  '''
    @params:
      ndim: integer
      n_spatial: list of integers, containing nx and maybe ny
      period_spatial: list of floats, containing x_period and maybe y_period
    @return:
      x_arr: [nx, 1] or [nx, ny, 2]
  '''
  if ndim == 1:
    nx = n_spatial[0]
    x_period = period_spatial[0]
    out = jnp.linspace(0.0, x_period, num = nx, endpoint = False)[:,None]  # [nx, 1]
  elif ndim == 2:
    nx, ny = n_spatial[0], n_spatial[1]
    x_period, y_period = period_spatial[0], period_spatial[1]
    x_arr = jnp.linspace(0.0, x_period, num = nx, endpoint = False)  # [nx]
    y_arr = jnp.linspace(0.0, y_period, num = ny, endpoint = False)  # [ny]
    x_mesh, y_mesh = jnp.meshgrid(x_arr, y_arr, indexing='ij')  # [nx, ny]
    out = jnp.stack([x_mesh, y_mesh], axis = -1)  # [nx, ny, 2]
  else:
    raise ValueError("ndim must be 1 or 2")
  return out

def H_L1_true_sol_1d(x, t, fn_J):
  '''
  @params:
    x: [1]
    t: scalar
    fn_J: function taking [...,1] and returning [...]
  @return:
    phi: scalar
  '''
  # phi(x,t) = min_{ |u-x| <= t } J(u)
  n_grids = 101
  x_tests = jnp.linspace(x[0]-t, x[0]+t, num = n_grids)[...,None]  # [n_grids, 1]
  J_tests = fn_J(x_tests)  # [n_grids]
  return jnp.min(J_tests)


def H_L1_true_sol_2d(x, t, fn_J):
  '''
  @params:
    x: [2]
    t: scalar
    fn_J: function taking [...,2] and returning [...]
  @return:
    phi: scalar
  '''
  # phi(x,t) = min_{ |u1-x1| <= t, |u2-x2| <= t } J(u)
  n_grids = 21
  x_tests = jnp.linspace(x[0]-t, x[0]+t, num = n_grids)  # [n_grids]
  y_tests = jnp.linspace(x[0]-t, x[0]+t, num = n_grids)  # [n_grids]
  x_mesh, y_mesh = jnp.meshgrid(x_tests, y_tests, indexing='ij')  # [n_grids, n_grids]
  x_arr2 = jnp.stack([x_mesh, y_mesh], axis = -1)  # [n_grids, n_grids, 2]
  x_arr2 = einshape('ijk->(ij)k', x_arr2)  # [n_grids**2, 2]
  J_tests = fn_J(x_tests)  # [n_grids**2]
  return jnp.min(J_tests)

H_L1_true_sol_1d_batch = partial(jax.jit, static_argnums=(2))(jax.vmap(H_L1_true_sol_1d, in_axes=(0, 0, None), out_axes=0))
H_L1_true_sol_1d_batch2 = jax.vmap(H_L1_true_sol_1d_batch, in_axes=(0, 0, None), out_axes=0)

H_L1_true_sol_2d_batch = partial(jax.jit, static_argnums=(2))(jax.vmap(H_L1_true_sol_1d, in_axes=(0, 0, None), out_axes=0))
H_L1_true_sol_2d_batch2 = jax.vmap(H_L1_true_sol_2d_batch, in_axes=(0, 0, None), out_axes=0)

# @jax.jit
def compute_EO_forward_solution_1d(nt, dx, dt, f_in_H, c_in_H, g, epsl = 0.0):
  '''
  @parameters:
    nt: integers
    dx, dt: floats
    f_in_H, c_in_H, g: [nx]
    g: [nx]
    epsl: float, diffusion coefficient
  @return:
    phi: [nt, nx]
  '''
  phi = []
  phi.append(g)
  for i in range(nt-1):
    dphidx_left = (phi[i] - jnp.roll(phi[i], 1))/dx
    dphidx_right = (jnp.roll(phi[i], -1) - phi[i])/dx
    H_1norm = jnp.maximum(-dphidx_right, 0) + jnp.maximum(dphidx_left, 0)
    H_val = c_in_H * H_1norm + f_in_H
    diffusion = epsl * (jnp.roll(phi[i], -1) - 2 * phi[i] + jnp.roll(phi[i], 1)) / (dx**2)
    phi.append(phi[i] - dt * H_val + dt * diffusion)
  phi_arr = jnp.stack(phi, axis = 0)
  print("phi dimension {}".format(jnp.shape(phi_arr)))
  print("phi {}".format(phi_arr))
  return phi_arr

def compute_EO_forward_solution_1d_general(nt, dt, dspatial, fns_dict, g, x_arr, epsl = 0.0):
  '''
  @parameters:
    nt: integers
    dt: floats
    dspatial: list of scalars
    fns_dict: constaining H_plus_fn, H_minus_fn: functions taking p([nt, nx]), x([1,nx]), t([nt-1,1]) and returning [nt, nx]
    g: [nx]
    epsl: float, diffusion coefficient
  @return:
    phi: [nt, nx]
  '''
  dx = dspatial[0]
  H_plus_fn = fns_dict.H_plus_fn
  H_minus_fn = fns_dict.H_minus_fn
  phi = []
  phi.append(g)
  for i in range(nt-1):
    # print('index for t: {}'.format(i))
    dphidx_left = (phi[i] - jnp.roll(phi[i], 1))/dx
    dphidx_right = (jnp.roll(phi[i], -1) - phi[i])/dx
    H_val = H_plus_fn(dphidx_left, x_arr, i*dt) + H_minus_fn(dphidx_right, x_arr, i*dt)
    diffusion = epsl * (jnp.roll(phi[i], -1) - 2 * phi[i] + jnp.roll(phi[i], 1)) / (dx**2)
    phi_new = phi[i] - dt * H_val + dt * diffusion
    phi.append(phi_new)
    if jnp.any(jnp.isnan(phi_new)):
      break
  phi_arr = jnp.stack(phi, axis = 0)
  print("phi dimension {}".format(jnp.shape(phi_arr)))
#   print("phi {}".format(phi_arr))
  return phi_arr
    
def compute_EO_forward_solution_2d(nt, dx, dy, dt, f_in_H, c_in_H, g, epsl=0.0):
  '''
  @parameters:
    nt: integers
    dx, dy, dt: floats
    f_in_H, c_in_H, g: [nx, ny]
    epsl: float, diffusion coefficient
  @return:
    phi: [nt, nx, ny]
  '''
  phi = []
  phi.append(g)
  for i in range(nt-1):
    dphidx_left = (phi[i] - jnp.roll(phi[i], 1, axis = 0))/dx
    dphidx_right = (jnp.roll(phi[i], -1, axis = 0) - phi[i])/dx
    dphidy_left = (phi[i] - jnp.roll(phi[i], 1, axis = 1))/dy
    dphidy_right = (jnp.roll(phi[i], -1, axis = 1) - phi[i])/dy
    H_1norm = jnp.maximum(-dphidx_right, 0) + jnp.maximum(dphidx_left, 0) \
                + jnp.maximum(-dphidy_right, 0) + jnp.maximum(dphidy_left, 0)  # [nx, ny]
    H_val = c_in_H * H_1norm + f_in_H  # [nx, ny]
    diffusion = epsl * (jnp.roll(phi[i], -1, axis = 0) - 2 * phi[i] + jnp.roll(phi[i], 1, axis = 0)) / (dx**2) \
                + epsl * (jnp.roll(phi[i], -1, axis = 1) - 2 * phi[i] + jnp.roll(phi[i], 1, axis = 1)) / (dy**2)
    phi.append(phi[i] - dt * H_val - dt * diffusion)
  phi_arr = jnp.array(phi)
  print("phi dimension {}".format(jnp.shape(phi_arr)))
  return phi_arr

def compute_EO_forward_solution_2d_general(nt, dt, dspatial, fns_dict, g, x_arr, epsl = 0.0):
  '''
  @parameters:
    nt: integers
    dx, dt: floats
    f_in_H, c_in_H, g: [nx]
    g: [nx]
    epsl: float, diffusion coefficient
  @return:
    phi: [nt, nx]
  '''
  Hx_plus_fn = fns_dict.Hx_plus_fn
  Hx_minus_fn = fns_dict.Hx_minus_fn
  Hy_plus_fn = fns_dict.Hy_plus_fn
  Hy_minus_fn = fns_dict.Hy_minus_fn
  dx, dy = dspatial[0], dspatial[1]
  phi = []
  phi.append(g)
  for i in range(nt-1):
    # print('index for t: {}'.format(i))
    dphidx_left = (phi[i] - jnp.roll(phi[i], 1, axis = 0))/dx
    dphidx_right = (jnp.roll(phi[i], -1, axis = 0) - phi[i])/dx
    dphidy_left = (phi[i] - jnp.roll(phi[i], 1, axis = 1))/dy
    dphidy_right = (jnp.roll(phi[i], -1, axis = 1) - phi[i])/dy
    H_val = Hx_plus_fn(dphidx_left, x_arr, i*dt) + Hx_minus_fn(dphidx_right, x_arr, i*dt)
    H_val += Hy_plus_fn(dphidy_left, x_arr, i*dt) + Hy_minus_fn(dphidy_right, x_arr, i*dt)
    diffusion = epsl * (jnp.roll(phi[i], -1, axis = 0) - 2 * phi[i] + jnp.roll(phi[i], 1, axis = 0)) / (dx**2) \
                + epsl * (jnp.roll(phi[i], -1, axis = 1) - 2 * phi[i] + jnp.roll(phi[i], 1, axis = 1)) / (dy**2)
    phi_new = phi[i] - dt * H_val + dt * diffusion
    phi.append(phi_new)
    if jnp.any(jnp.isnan(phi_new)):
        break
  phi_arr = jnp.stack(phi, axis = 0)
  print("phi dimension {}".format(jnp.shape(phi_arr)))
#   print("phi {}".format(phi_arr))
  return phi_arr

def read_solution(filename):
  with open(filename, 'rb') as f:
    results, errors = pickle.load(f)
  # compute errors
  phi = results[-1][-1]
  return phi

def read_raw_file(filename):
  with open(filename, 'rb') as f:
    result = pickle.load(f)
  return result

def get_sol_on_coarse_grid_1d(sol, coarse_nt, coarse_nx):
  nt, nx = jnp.shape(sol)
  if (nt - 1) % (coarse_nt - 1) != 0 or nx % coarse_nx != 0:
    raise ValueError("nx and nt-1 must be divisible by coarse_nx and coarse_nt-1")
  else:
    nt_factor = (nt-1) // (coarse_nt-1)
    nx_factor = nx // coarse_nx
    sol_coarse = sol[::nt_factor, ::nx_factor]
  return sol_coarse

def get_sol_on_coarse_grid_2d(sol, coarse_nt, coarse_nx, coarse_ny):
  nt, nx, ny = jnp.shape(sol)
  if (nt - 1) % (coarse_nt - 1) != 0 or nx % coarse_nx != 0 or ny % coarse_ny != 0:
    raise ValueError("nx, ny and nt-1 must be divisible by coarse_nx, coarse_ny and coarse_nt-1")
  else:
    nt_factor = (nt-1) // (coarse_nt-1)
    nx_factor = nx // coarse_nx
    ny_factor = ny // coarse_ny
    sol_coarse = sol[::nt_factor, ::nx_factor, ::ny_factor]
  return sol_coarse

def compute_err_1d(phi, true_soln):
  '''
  @parameters:
    phi: [nt, nx]
    true_soln: [nt_dense, nx_dense]
  @return:
    err_l1, err_l1_rel: scalar
    error: [nt, nx]
  '''
  nt, nx = jnp.shape(phi)
  phi_true = get_sol_on_coarse_grid_1d(true_soln, nt, nx)
  error = phi - phi_true
  err_l1 = jnp.mean(jnp.abs(error))
  err_l1_rel = err_l1/ jnp.mean(jnp.abs(phi_true))
  return err_l1, err_l1_rel, error

def compute_err_2d(phi, true_soln):
  '''
  @parameters:
    phi: [nt, nx, ny]
    true_soln: [nt_dense, nx_dense, ny_dense]
  @return:
    err_l1, err_l1_rel: scalar
    error: [nt, nx, ny]
  '''
  nt, nx, ny = jnp.shape(phi)
  phi_true = get_sol_on_coarse_grid_2d(true_soln, nt, nx, ny)
  error = phi - phi_true
  err_l1 = jnp.mean(jnp.abs(error))
  err_l1_rel = err_l1/ jnp.mean(jnp.abs(phi_true))
  return err_l1, err_l1_rel, error

def plot_solution_1d(phi, error, nt, nspatial, T, period_spatial, figname, epsl = 0):
  '''
  @ parameters:
    phi, error: [nt, nx]
    nt, nx: integer
    T, x_period: float
    figname: string for saving figure
  '''
  nx = nspatial[0]
  x_period = period_spatial[0]
  name_prefix = 'nt{}_nx{}'.format(nt, nx)
  if epsl == 0:
    name_prefix += '_epsl0_'
  elif epsl == 0.1:
    name_prefix += '_epsl0p1_'
  else:
    raise ValueError("epsl must be 0 or 0.1")
  x_arr = np.linspace(0.0, x_period, num = nx + 1, endpoint = True)
  t_arr = np.linspace(0.0, T, num = nt, endpoint = True)
  t_mesh, x_mesh = np.meshgrid(t_arr, x_arr)
  print("shape x_arr {}, t_arr {}".format(np.shape(x_arr), np.shape(t_arr)))
  print("shape x_mesh {}, t_mesh {}".format(np.shape(x_mesh), np.shape(t_mesh)))
  # plot solution
  phi_trans = einshape('ij->ji', phi)  # [nx, nt]
  phi_np = jax.device_get(jnp.concatenate([phi_trans, phi_trans[0:1,:]], axis = 0))  # [nx+1, nt]
  fig = plt.figure()
  plt.contourf(x_mesh, t_mesh, phi_np)
  plt.colorbar()
  plt.xlabel('x')
  plt.ylabel('t')
  plt.savefig(figname + name_prefix + 'solution.png')
  # plot error
  err_trans = einshape('ij->ji', error)  # [nx, nt]
  err_np = jax.device_get(jnp.concatenate([err_trans, err_trans[0:1,:]], axis = 0))  # [nx+1, nt]
  fig = plt.figure()
  plt.contourf(x_mesh, t_mesh, err_np)
  plt.colorbar()
  plt.xlabel('x')
  plt.ylabel('t')
  plt.savefig(figname + name_prefix + 'error.png')


def plot_solution_2d(phi, error, nt, n_spatial, T, period_spatial, figname, epsl=0.0):
  '''
  @ parameters:
    phi, error: [nt, nx, ny]
    nt, nx, ny: integer
    T, x_period, y_period: float
    figname: string for saving figure
  '''
  nx, ny = n_spatial[0], n_spatial[1]
  x_period, y_period = period_spatial[0], period_spatial[1]
  name_prefix = 'nt{}_nx{}_ny{}'.format(nt, nx, ny)
  if epsl == 0:
    name_prefix += '_epsl0_'
  elif epsl == 0.1:
    name_prefix += '_epsl0p1_'
  else:
    raise ValueError("epsl must be 0 or 0.1")
  x_arr = np.linspace(0.0, x_period, num = nx + 1, endpoint = True)
  y_arr = np.linspace(0.0, y_period, num = ny + 1, endpoint = True)
  x_mesh, y_mesh = np.meshgrid(x_arr, y_arr, indexing='ij')
  # plot solution
  phi_terminal = jnp.concatenate([phi[-1,...], phi[-1, 0:1, :]], axis = 0)  # [nx+1, ny]
  phi_terminal_np = jnp.concatenate([phi_terminal, phi_terminal[:,0:1]], axis = 1)  # [nx+1, ny+1]
  fig = plt.figure()
  plt.contourf(x_mesh, y_mesh, phi_terminal_np)
  plt.colorbar()
  plt.xlabel('x')
  plt.ylabel('y')
  plt.savefig(figname + name_prefix + 'solution.png')
  # plot error
  err_terminal = jnp.concatenate([error[-1,...], error[-1, 0:1, :]], axis = 0)  # [nx+1, ny]
  err_terminal_np = jnp.concatenate([err_terminal, err_terminal[:,0:1]], axis = 1)  # [nx+1, ny+1]
  fig = plt.figure()
  plt.contourf(x_mesh, y_mesh, err_terminal_np)
  plt.colorbar()
  plt.xlabel('x')
  plt.ylabel('y')
  plt.savefig(figname + name_prefix + 'error.png')

def get_save_dir(time_stamp, egno, ndim, nt, nx, ny):
  save_dir = './check_points/{}'.format(time_stamp) + '/eg{}_{}d'.format(egno, ndim)
  if ndim == 1:
    filename_prefix = 'nt{}_nx{}'.format(nt, nx)
  elif ndim == 2:
    filename_prefix = 'nt{}_nx{}_ny{}'.format(nt, nx, ny)
  return save_dir, filename_prefix


def get_cfl_condition(nx_dense, T, x_period, epsl=0, ndim=1):
  dx_dense = x_period / nx_dense
  dt_dense = 0.9/(epsl / (dx_dense**2) + 2/dx_dense) / ndim
  nt_dense = int(T / dt_dense) + 2
  return nt_dense

def compute_true_soln_eg0(nt, n_spatial, ndim, T, period_spatial, J):
  ''' compute the true solution of HJ PDE with no diffusion and L1 Hamiltonian using LO formula
  '''
  x_arr = compute_xarr(ndim, n_spatial, period_spatial)
  t_arr = jnp.linspace(0.0, T, num = nt)  # [nt]
  phi_dense_list = []
  for i in range(nt):
    if ndim == 1:
      phi_dense_list.append(H_L1_true_sol_1d_batch(x_arr, t_arr[i] + jnp.zeros_like(x_arr[...,0]), J))
    else:
      phi_dense_list.append(H_L1_true_sol_2d_batch2(x_arr, t_arr[i] + jnp.zeros_like(x_arr[...,0]), J))
  phi_dense = jnp.stack(phi_dense_list, axis = 0)
  return phi_dense

def compute_true_soln_eg10(nt, n_spatial, ndim, T, period_spatial):
  ''' compute the true solution of HJ PDE with no diffusion and quad Hamiltonian and quad initial condition
        phi(x,t) = |x|^2/2/(1+t)
  '''
  x_arr = compute_xarr(ndim, n_spatial, period_spatial)
  t_arr = jnp.linspace(0.0, T, num = nt)  # [nt]
  phi_dense_list = []
  for i in range(nt):
    phi_dense_list.append(jnp.sum(x_arr**2, axis = -1)/2/(1+t_arr[i]))
  phi_dense = jnp.stack(phi_dense_list, axis = 0)
  return phi_dense

def compute_ground_truth(egno, nt_dense, n_spatial, ndim, T, period_spatial, epsl = 0.0):
  dt_dense = T / (nt_dense - 1)
  dspatial_dense = [period_spatial[i] / n_spatial[i] for i in range(ndim)]
  print('ground truth: nt {}, n_spatial {}'.format(nt_dense, n_spatial))
  if ndim == 1:
    J, fns_dict = set_up_example_fns(egno, ndim, period_spatial, theoretical_ver=False)
    if egno == 0 and epsl == 0.0:
      print('compute true soln using eg0 epsl 0.0 solver')
      phi_dense = compute_true_soln_eg0(nt_dense, n_spatial, ndim, T, period_spatial, J)
    # elif egno == 10 and epsl == 0.0:
    #   print('compute true soln using eg10 epsl 0.0 solver')
    #   phi_dense = compute_true_soln_eg10(nt_dense, n_spatial, ndim, T, period_spatial)
    else:
      print('compute true soln using general solver')
      x_arr = compute_xarr(ndim, n_spatial, period_spatial)
      g = J(x_arr)  # [nx_dense]
      print('shape g {}'.format(jnp.shape(g)))
      trial_num = 10
      for j in range(trial_num):
        print('trial {}: nt_dense {}, nspatial_dense {}'.format(j, nt_dense, n_spatial))
        phi_dense = compute_EO_forward_solution_1d_general(nt_dense, dt_dense, dspatial_dense,
                                                        fns_dict, g, x_arr, epsl = epsl)
        if not jnp.any(jnp.isnan(phi_dense)):
          break
        nt_dense = 2 * (nt_dense - 1) + 1
  elif ndim == 2:
    J, fns_dict = set_up_example_fns(egno, ndim, period_spatial, theoretical_ver=False)
    if egno == 0 and epsl == 0.0:
      print('compute true soln using eg0 epsl 0.0 solver')
      phi_dense = compute_true_soln_eg0(nt_dense, n_spatial, ndim, T, period_spatial, J)
    # elif egno == 10 and epsl == 0.0:
    #   print('compute true soln using eg10 epsl 0.0 solver')
    #   phi_dense = compute_true_soln_eg10(nt_dense, n_spatial, ndim, T, period_spatial)
    else:
      print('compute true soln using general solver')
      trial_num = 5
      x_arr = compute_xarr(ndim, n_spatial, period_spatial)
      g = J(x_arr)  # [nx_dense, ny_dense]
      for j in range(trial_num):
        print('trial {}: nt_dense {}, nspatial_dense {}'.format(j, nt_dense, n_spatial))
        phi_dense = compute_EO_forward_solution_2d_general(nt_dense, dt_dense, dspatial_dense, 
                                                    fns_dict, g, x_arr, epsl=epsl)
        if not jnp.any(jnp.isnan(phi_dense)):
          break
        nt_dense = 2 * (nt_dense - 1) + 1
  else:
    raise ValueError("ndim should be 1 or 2")
  print('finished computing, shape phi_dense {}'.format(jnp.shape(phi_dense)))
  return phi_dense

def main(argv):
  for key, value in FLAGS.__flags.items():
    print(value.name, ": ", value._value, flush=True)

  nt = FLAGS.nt
  nx = FLAGS.nx
  ny = FLAGS.ny
  ndim = FLAGS.ndim
  nx_dense = FLAGS.nx_dense
  ny_dense = FLAGS.ny_dense
  nt_dense = FLAGS.nt_dense
  egno = FLAGS.egno
  epsl = FLAGS.epsl
  num_filename = FLAGS.numerical_sol_filename
  true_filename = FLAGS.true_sol_filename
  if_compute_true_sol = FLAGS.if_compute_true_sol

  plot_foldername = "eg{}_{}d/".format(egno, ndim)
  if FLAGS.plot_folder != '':
    plot_foldername += FLAGS.plot_folder + '/'
  if FLAGS.hero_folder != '':
    plot_foldername = FLAGS.hero_folder + '/' + plot_foldername
  if not os.path.exists(plot_foldername):
    os.makedirs(plot_foldername)

  T = 1.0
  x_period = 2.0
  y_period = 2.0
  # setup example
  if ndim == 1:
    period_spatial = [x_period]
  else:
    period_spatial = [x_period, y_period]
  J, fns_dict = set_up_example_fns(egno, ndim, period_spatial, theoretical_ver=FLAGS.theoretical_scheme)

  # compute or read true soln
  if if_compute_true_sol == False:
    if not os.path.exists(true_filename):
      print('true solution file {} does not exist, exit'.format(true_filename))
      return
    true_sol = read_raw_file(true_filename)
  else: # compute true solution
    if (egno == 0 or egno == 10) and (epsl == 0.0):
      if nx_dense == 0:
        nx_dense, ny_dense = nx, ny
        nt_dense = nt
    else:
      if nx_dense == 0:
        nx_dense, ny_dense = 800, 800
        nt_dense_min = get_cfl_condition(nx_dense, T, x_period, epsl=epsl, ndim=ndim)
        nt_dense = ((nt_dense_min-1) // (nt - 1) + 1) * (nt - 1) + 1
        # nt_dense = 1201
    if ndim == 1:
      n_spatial = [nx_dense]
    else:
      n_spatial = [nx_dense, ny_dense]
    true_sol = compute_ground_truth(egno, nt_dense, n_spatial, ndim, T, period_spatial, epsl=epsl)
    with open(true_filename, 'wb') as file:
      pickle.dump(true_sol, file)
      print('saved to {}'.format(file), flush = True)

  dt = T / (nt - 1)
  dx = x_period / nx
  dy = y_period / ny
  num_sol = read_solution(num_filename)
  err_l1_rel = -1
  # compute error
  if ndim == 1:
    n_spatial = [nx]
    period_spatial = [x_period]
    dspatial = [dx]
    t_arr = jnp.linspace(dt, T, num = nt - 1)[:,None]  # [nt-1, 1]
    compute_residual_fn = compute_HJ_residual_EO_1d_general
    compute_err_fn = compute_err_1d
    plot_fn = plot_solution_1d
  else:
    n_spatial = [nx, ny]
    period_spatial = [x_period, y_period]
    dspatial = [dx, dy]
    t_arr = jnp.linspace(dt, T, num = nt - 1)[:,None,None]  # [nt-1, 1,1]
    compute_residual_fn = compute_HJ_residual_EO_2d_general
    compute_err_fn = compute_err_2d
    plot_fn = plot_solution_2d
    
  x_arr = compute_xarr(ndim, n_spatial, period_spatial)[None,...]  # [1, nx, 1] or [1, nx, ny, 2]
  HJ_residual = compute_residual_fn(num_sol, dt, dspatial, fns_dict, epsl, x_arr, t_arr)
  if not jnp.any(jnp.isnan(true_sol)):
    err_l1, err_l1_rel, error = compute_err_fn(num_sol, true_sol)
    plot_fn(num_sol, error, nt, n_spatial, T, period_spatial, plot_foldername, epsl=epsl)
  else:
    print('true solution contains nan')
    return
  print('row 1: HJ residual {:.2E}'.format(jnp.mean(jnp.abs(HJ_residual))))
  print("row 2: err_l1_rel {:.2E}".format(err_l1_rel))
    

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('nt', 11, 'size of t grids')
    flags.DEFINE_integer('nx', 20, 'size of x grids')
    flags.DEFINE_integer('ny', 20, 'size of y grids')
    flags.DEFINE_integer('nt_dense', 0, 'size of t grids for dense solution')
    flags.DEFINE_integer('nx_dense', 0, 'size of x grids for dense solution')
    flags.DEFINE_integer('ny_dense', 0, 'size of y grids for dense solution')
    flags.DEFINE_integer('ndim', 1, 'dimensionality')
    flags.DEFINE_integer('egno', 1, 'index of example')
    flags.DEFINE_string('numerical_sol_filename', '', 'the name of the pickle file of numerical solution to read')
    flags.DEFINE_string('true_sol_filename', '', 'the name of the true solution, if does not exists, compute and save')
    flags.DEFINE_float('epsl', 0.0, 'diffusion coefficient')
    flags.DEFINE_boolean('theoretical_scheme', True, 'true if aligned with theory')
    flags.DEFINE_boolean('if_compute_true_sol', False, 'true if compute and save true solution')

    flags.DEFINE_string('plot_folder', '', 'the folder name of plot')
    flags.DEFINE_string('hero_folder', '', 'the folder name of hero run')
    
    app.run(main)

    
    
