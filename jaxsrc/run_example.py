import jax.numpy as jnp
from absl import app, flags, logging
from set_fns import set_up_example_fns, set_up_J
import pytz
from datetime import datetime
from utils.utils_pdhg_solver import PDHG_multi_step
from solver import save
import update_fns_in_pdhg as pdhg
from utils.utils_precond import compute_Dxx_fft_fv
import solver
import tensorflow as tf
import os
import plot_soln

def main(argv):
  for key, value in FLAGS.__flags.items():
    print(value.name, ": ", value._value, flush=True)

  nt = FLAGS.nt
  nx = FLAGS.nx
  ny = FLAGS.ny
  ndim = FLAGS.ndim
  egno = FLAGS.egno
  ifsave = FLAGS.ifsave
  stepsz_param = FLAGS.stepsz_param
  c_on_rho = FLAGS.c_on_rho
  epsl = FLAGS.epsl
  time_step_per_PDHG = FLAGS.time_step_per_PDHG
  eps = FLAGS.eps
  T = FLAGS.T

  if ndim == 1:
    prob_name = 'eg{}_nt{}_nx{}_epsl{:.2f}_c{}'.format(egno, nt, nx, epsl, int(c_on_rho))
  elif ndim == 2:
    prob_name = 'eg{}_nt{}_nx{}_ny{}_epsl{:.2f}_c{}'.format(egno, nt, nx, ny, epsl, int(c_on_rho))
  else:
    raise NotImplementedError

  if FLAGS.tfboard:
    time_stamp = datetime.now(pytz.timezone('America/Los_Angeles')).strftime("%Y%m%d-%H%M%S")
    stamp = time_stamp
    results_dir = f'./tf_save/{prob_name}/'+ stamp
    print("tf foldername: ", results_dir)
    file_writer = tf.summary.create_file_writer(results_dir)
    file_writer.set_as_default()

  print('nx: ', nx)
  print('ny: ', ny)

  N_maxiter = FLAGS.N_maxiter
  print_freq = 10000
  x_period, y_period = 2, 2

  time_stamp = datetime.now(pytz.timezone('America/Los_Angeles')).strftime("%Y%m%d-%H%M%S")
  logging.info("current time: " + datetime.now(pytz.timezone('America/Los_Angeles')).strftime("%Y%m%d-%H%M%S"))
  save_dir = './check_points/{}'.format(time_stamp) + '/eg{}_{}d'.format(egno, ndim)
  if ndim == 1:
    filename_prefix = 'nt{}_nx{}'.format(nt, nx)
  elif ndim == 2:
    filename_prefix = 'nt{}_nx{}_ny{}'.format(nt, nx, ny)

  dx = x_period / (nx)
  dy = y_period / (ny)
  dt = T / (nt-1)
  x_arr = jnp.linspace(0.0, x_period - dx, num = nx)[None,:,None]  # [1, nx, 1]

  if ndim == 1:
    period_spatial = (x_period,)
  else:
    period_spatial = (x_period, y_period)
  
  J = set_up_J(egno, ndim, period_spatial)
  fns_dict = set_up_example_fns(egno, ndim)
  
  if ndim == 1:
    x_arr = jnp.linspace(0.0, x_period - dx, num = nx)[None,:,None]  # [1, nx, 1]
    t_arr = jnp.linspace(0.0, T, num = nt)[:,None]  # [nt, 1]
  else:
    x_arr = jnp.linspace(0.0, x_period - dx, num = nx)  
    y_arr = jnp.linspace(0.0, x_period - dy, num = ny)
    x_mesh, y_mesh = jnp.meshgrid(x_arr, y_arr, indexing='ij')  # [nx, ny]
    x_arr = jnp.stack([x_mesh, y_mesh], axis = -1)[None,...]  # [1, nx, ny, 2]
    t_arr = jnp.linspace(0.0, T, num = nt)[:,None,None]  # [nt, 1, 1]

  g = J(x_arr)  # [1, nx] or [1, nx, ny]
  print('shape of g: ', g.shape)

  if ndim == 1:
    fn_update_primal = pdhg.update_primal_1d
    def fn_compute_err(phi, dt, dspatial, fns_dict, epsl, x_arr, t_arr):
      HJ_residual = solver.compute_HJ_residual_EO_1d_general(phi, dt, dspatial, fns_dict, epsl, x_arr, t_arr)
      return jnp.mean(jnp.abs(HJ_residual))
  else:
    fn_update_primal = pdhg.update_primal_2d
    def fn_compute_err(phi, dt, dspatial, fns_dict, epsl, x_arr, t_arr):
      HJ_residual = solver.compute_HJ_residual_EO_2d_general(phi, dt, dspatial, fns_dict, epsl, x_arr, t_arr)
      return jnp.mean(jnp.abs(HJ_residual))
  fn_update_dual = pdhg.update_dual

  if ndim == 1:
    dspatial = (dx, )
    nspatial = (nx, )
  else:
    dspatial = (dx, dy)
    nspatial = (nx, ny)
    print('dspatial: ', dspatial)
    print('nspatial: ', nspatial)
  
  # fv for preconditioning
  fv = compute_Dxx_fft_fv(ndim, nspatial, dspatial)

  results, errs_all = PDHG_multi_step(fn_update_primal, fn_update_dual, fn_compute_err, fns_dict, g, x_arr, 
                                       ndim, nt, nspatial, dt, dspatial, c_on_rho, time_step_per_PDHG = time_step_per_PDHG,
                                       epsl = epsl, stepsz_param=stepsz_param, fv=fv,
                                       N_maxiter = N_maxiter, print_freq = print_freq, eps = eps, tfboard = FLAGS.tfboard)
  if ifsave:
    save(save_dir, filename_prefix, (results, errs_all))

  phi = results[-1][1]
  rho = results[-1][2]
  alp = results[-1][3]

  if FLAGS.tfboard:
    if ndim == 1:
      plot_fn = plot_soln.plot_solution_1d
      alp_titles = ['alp_1', 'alp_2']
    elif ndim == 2:
      plot_fn = plot_soln.plot_solution_2d
      alp_titles = ['alp_11', 'alp_12', 'alp_21', 'alp_22']
    else:
      raise NotImplementedError
    fig_phi = plot_fn(phi, x_arr, t_arr, title = 'phi', tfboard = True)
    tf.summary.image('phi', fig_phi, step = 0)
    fig_rho = plot_fn(rho, x_arr, t_arr[:-1,...], title = 'rho', tfboard = True)
    tf.summary.image('rho', fig_rho, step = 0)
    for i in range(2**ndim):
      fig_alp = plot_fn(alp[i,...,0], x_arr, t_arr[:-1,...], title = alp_titles[i] + ', x-coordinate', tfboard = True)
      tf.summary.image(alp_titles[i] + '_x', fig_alp, step = 0)
      if ndim == 2:
        fig_alp = plot_fn(alp[i,...,1], x_arr, t_arr[:-1,...], title = alp_titles[i] + ', y-coordinate', tfboard = True)
      tf.summary.image(alp_titles[i] + '_y', fig_alp, step = 0)
  
  print('phi: ', phi)





if __name__ == '__main__':
  FLAGS = flags.FLAGS
  flags.DEFINE_integer('egno', 1, 'index of example')
  flags.DEFINE_integer('ndim', 1, 'spatial dimension')
  flags.DEFINE_integer('nt', 11, 'size of t grids')
  flags.DEFINE_integer('nx', 20, 'size of x grids')
  flags.DEFINE_integer('ny', 20, 'size of y grids')
  
  flags.DEFINE_float('epsl', 0.0, 'diffusion coefficient')
  flags.DEFINE_float('T', 1.0, 'final time')
  flags.DEFINE_float('c_on_rho', 10.0, 'the constant added on rho')
  
  flags.DEFINE_integer('time_step_per_PDHG', 2, 'number of time discretization per PDHG iteration')
  flags.DEFINE_integer('N_maxiter', 1000000, 'maximum number of iterations')
  flags.DEFINE_float('stepsz_param', 0.1, 'default step size constant')
  flags.DEFINE_float('eps', 1e-6, 'the error threshold')

  flags.DEFINE_boolean('ifsave', True, 'if save to pickle')
  flags.DEFINE_boolean('tfboard', False, 'if use tfboard')
  
  app.run(main)
