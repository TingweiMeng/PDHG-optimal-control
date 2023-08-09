import jax
import jax.numpy as jnp
from einshape import jax_einshape as einshape
import numpy as np
from absl import app, flags
import pickle
import matplotlib.pyplot as plt
import os
from solver import set_up_example_fns, compute_xarr, compute_ground_truth, compute_err_1d, compute_err_2d
from solver import compute_HJ_residual_EO_1d_general, compute_HJ_residual_EO_2d_general
from solver import read_solution, read_raw_file
from functools import partial



def plot_solution_1d(phi, error, nt, nspatial, T, period_spatial, figname, epsl = 0):
  ''' plot solution and error of 1d HJ PDE
  @ parameters:
    phi, error: [nt, nx]
    nt: integer
    nspatial: list of one number nx
    T: float
    period_spatial: list of one number x_period
    figname: string for saving figure
    epsl: float, diffusion coefficient
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
  ''' plot solution and error of 2d HJ PDE at terminal time
  @ parameters:
    phi, error: [nt, nx, ny]
    nt: integer
    n_spatial: list of two numbers nx, ny
    T: float
    period_spatial: list of two numbers x_period, y_period
    figname: string for saving figure
    epsl: float, diffusion coefficient
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
  if FLAGS.child_folder != '':
    plot_foldername += FLAGS.child_folder + '/'
  if FLAGS.parent_folder != '':
    plot_foldername = FLAGS.parent_folder + '/' + plot_foldername
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
  J, fns_dict = set_up_example_fns(egno, ndim, period_spatial)

  # compute or read true soln
  if if_compute_true_sol == False:
    if not os.path.exists(true_filename):
      print('true solution file {} does not exist, exit'.format(true_filename))
      return
    true_sol = read_raw_file(true_filename)
  else: # compute and save true solution
    if egno == 0 and (epsl == 0.0) and ndim == 1:
      if nx_dense == 0:
        nx_dense, ny_dense = nx, ny
        nt_dense = nt
    else:
      if nx_dense == 0:
        nx_dense, ny_dense = 800, 800
        nt_dense = 1201
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
    flags.DEFINE_boolean('if_compute_true_sol', False, 'true if compute and save true solution')

    flags.DEFINE_string('child_folder', '', 'the folder name of child folder for saving figures')
    flags.DEFINE_string('parent_folder', '', 'the folder name of parent folder for saving figures')
    
    app.run(main)

    
    
