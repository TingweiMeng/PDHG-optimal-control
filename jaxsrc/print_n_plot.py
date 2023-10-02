import jax.numpy as jnp
from absl import app, flags
import os
from solver import set_up_example_fns, compute_xarr, compute_ground_truth, compute_err_1d, compute_err_2d
from solver import compute_HJ_residual_EO_1d_general, compute_HJ_residual_EO_2d_general
from solver import read_solution, read_raw_file, save_raw
import plot_soln



def main(argv):
  for key, value in FLAGS.__flags.items():
    print(value.name, ": ", value._value, flush=True)

  ndim = FLAGS.ndim
  nx_dense = FLAGS.nx_dense
  ny_dense = FLAGS.ny_dense
  nt_dense = FLAGS.nt_dense
  egno = FLAGS.egno
  epsl = FLAGS.epsl
  num_filenames = FLAGS.numerical_sol_filenames
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

  if ndim == 1:
    compute_residual_fn = compute_HJ_residual_EO_1d_general
    compute_err_fn = compute_err_1d
    plot_fn = plot_soln.plot_solution_1d
  else:
    compute_residual_fn = compute_HJ_residual_EO_2d_general
    compute_err_fn = compute_err_2d
    plot_fn = plot_soln.plot_solution_2d

  # compute or read true soln
  if if_compute_true_sol == False:
    if not os.path.exists(true_filename):
      print('true solution file {} does not exist, exit'.format(true_filename))
      return
    true_sol = read_raw_file(true_filename)
    if true_sol.ndim != ndim + 1:
      print('true solution file {} has wrong dimension, exit'.format(true_filename))
      return
  else: # compute and save true solution
    if nx_dense == 0:
      nx_dense, ny_dense = 800, 800
      nt_dense = 1201
    if ndim == 1:
      n_spatial_dense = [nx_dense]
    else:
      n_spatial_dense = [nx_dense, ny_dense]
    true_sol = compute_ground_truth(egno, nt_dense, n_spatial_dense, ndim, T, period_spatial, epsl=epsl)
    # save true solution
    save_raw(true_filename, true_sol)
    # plot true solution
    if not jnp.any(jnp.isnan(true_sol)):
      n_spatial_dense = true_sol.shape[1:]
      nt_dense = true_sol.shape[0]
      plot_fn(true_sol, nt_dense, n_spatial_dense, T, period_spatial, plot_foldername, 'true_solution', epsl=epsl)
  
  HJ_residuals = []
  err_l1_rels = []
  nspatials = []
  nts = []
  for num_filename in num_filenames:
    num_sol = read_solution(num_filename)
    if num_sol.ndim != ndim + 1:
      print('numerical solution file {} has wrong dimension, exit'.format(num_filename))
      return
    nt = num_sol.shape[0]
    n_spatial = num_sol.shape[1:]
    nspatials.append(n_spatial)
    nts.append(nt)
    dt = T / (nt - 1)
    # compute error
    if ndim == 1:
      period_spatial = [x_period]
      dspatial = [x_period / n_spatial[0]]
      t_arr = jnp.linspace(dt, T, num = nt - 1)[:,None]  # [nt-1, 1]
    else:
      period_spatial = [x_period, y_period]
      dspatial = [x_period / n_spatial[0], y_period / n_spatial[1]]
      t_arr = jnp.linspace(dt, T, num = nt - 1)[:,None,None]  # [nt-1, 1,1]
    # read numerical solution, compute residual and error, plot solution and error
    x_arr = compute_xarr(ndim, n_spatial, period_spatial)[None,...]  # [1, nx, 1] or [1, nx, ny, 2]
    HJ_residual = compute_residual_fn(num_sol, dt, dspatial, fns_dict, epsl, x_arr, t_arr)
    HJ_residuals.append(jnp.mean(jnp.abs(HJ_residual)))
    if not jnp.any(jnp.isnan(true_sol)):
      err_l1, err_l1_rel, error = compute_err_fn(num_sol, true_sol)
      plot_fn(num_sol, nt, n_spatial, T, period_spatial, plot_foldername, 'solution', epsl=epsl)
      plot_fn(error, nt, n_spatial, T, period_spatial, plot_foldername, 'error', epsl=epsl)
      err_l1_rels.append(err_l1_rel)
    else:
      print('true solution contains nan')
      return
  if len(num_filenames) == 1:
    print('row 1: HJ residual {:.2E}'.format(jnp.mean(jnp.abs(HJ_residual))))
    print("row 2: err_l1_rel {:.2E}".format(err_l1_rel))
  else:
    print('=============== error table ===============')
    if ndim == 1:
      print('$n_x\\times n_t$', end=' ')
      for nt, n_spatial in zip(nts, nspatials):
        print('& ${}\\times {}$'.format(n_spatial[0], nt), end=' ')
    else:
      print('$n_x\\times n_y\\times n_t$', end=' ')
      for nt, n_spatial in zip(nts, nspatials):
        print('& ${}\\times {}\\times {}$'.format(n_spatial[0], n_spatial[1], nt), end=' ')
    print('\\\\')
    print('\\hline')
    print('Averaged absolute residual of HJ PDE', end=' ')
    for HJ_residual in HJ_residuals:
      print('& {:.2E}'.format(HJ_residual), end=' ')
    print('\\\\')
    print('\\hline')
    print('$\\ell^1$ relative error', end=' ')
    for err_l1_rel in err_l1_rels:
      print('& {:.2E}'.format(err_l1_rel), end=' ')
    print('\\\\')


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('nt_dense', 0, 'size of t grids for dense solution')
    flags.DEFINE_integer('nx_dense', 0, 'size of x grids for dense solution')
    flags.DEFINE_integer('ny_dense', 0, 'size of y grids for dense solution')
    flags.DEFINE_integer('ndim', 1, 'dimensionality')
    flags.DEFINE_integer('egno', 1, 'index of example')
    flags.DEFINE_list("numerical_sol_filenames", [], "A list of the name of the pickle file of numerical solution to read")
    flags.DEFINE_string('true_sol_filename', '', 'the name of the true solution, if does not exists, compute and save')
    flags.DEFINE_float('epsl', 0.0, 'diffusion coefficient')
    flags.DEFINE_boolean('if_compute_true_sol', False, 'true if compute and save true solution')

    flags.DEFINE_string('child_folder', '', 'the folder name of child folder for saving figures')
    flags.DEFINE_string('parent_folder', '', 'the folder name of parent folder for saving figures')
    
    app.run(main)

    
    
