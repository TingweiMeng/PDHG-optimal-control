import jax.numpy as jnp
from absl import app, flags, logging
from set_fns import set_up_example_fns, set_up_J
import pytz
from datetime import datetime
from utils.utils_pdhg_solver import PDHG_multi_step
from solver import save, load_solution
import update_fns_in_pdhg as pdhg
from utils.utils_precond import compute_Dxx_fft_fv
import solver
import tensorflow as tf
import os
import utils.utils_plot as utils_plot
import scipy.interpolate as interpolate

def compute_traj_1d(x_terminal, alp, fn_f, nt, x_arr, t_arr):
  ''' x' = -f(alp,x,t), alp = alp(x,t)'''
  ''' x_arr: [nx], t_arr: [nt], alp: [nt-1, nx] '''
  traj_alp = []
  traj_x = [x_terminal]
  x_curr = x_terminal  # [n_sample]
  # repeat bdry: TODO: handle periodic bdry
  x_arr = jnp.concatenate([x_arr, x_arr[0:1]], axis = 0)  # [nx+1]
  alp = jnp.concatenate([alp, alp[:, 0:1]], axis = 1)  # [nt-1, nx+1]
  for i in range(nt):
    ind = -1-i
    dt = t_arr[ind] - t_arr[ind-1]
    # interpolation
    alp_x = jnp.interp(x_curr, x_arr, alp[ind,:])  # [n_sample]
    traj_alp.append(alp_x)
    vel = fn_f(alp_x, x_curr, t_arr[ind])  # [n_sample]
    x_curr = x_curr - vel * dt
    traj_x.append(x_curr)
  # reverse the order to get a forward time
  traj_alp = jnp.stack(traj_alp[::-1], axis = 0)  # [nt, n_sample]
  traj_x = jnp.stack(traj_x[::-1], axis = 0)  # [nt, n_sample]
  return traj_alp, traj_x

def compute_traj_2d(x_terminal, alp, fn_f, nt, x_arr, t_arr):
  ''' x' = -f(alp,x,t), alp = alp(x,t)'''
  ''' x_arr: [nx, ny, 2], t_arr: [nt], alp: [nt-1, nx, ny, nstate] '''
  traj_alp = []
  traj_x = [x_terminal]
  x_curr = x_terminal  # [n_sample, 2]
  # repeat bdry
  x_arr = jnp.concatenate([x_arr, x_arr[0:1,:,:]], axis = 0)  # [nx+1, ny, 2]
  x_arr = jnp.concatenate([x_arr, x_arr[:,0:1,:]], axis = 1)  # [nx+1, ny+1, 2]
  alp = jnp.concatenate([alp, alp[:, 0:1,:,:]], axis = 1)  # [nt-1, nx+1, ny, nstate]
  alp = jnp.concatenate([alp, alp[:,:,0:1,:]], axis = 2)  # [nt-1, nx+1, ny+1, nstate]
  for i in range(nt):
    ind = -1-i
    dt = t_arr[ind] - t_arr[ind-1]
    # interpolation
    alp_x = interpolate.interpn((x_arr[0,:,0], x_arr[:,0,1]), alp[ind,:,:,:], x_curr, method='linear')  # [n_sample, nstate]
    traj_alp.append(alp_x)
    vel = fn_f(alp_x, x_curr, t_arr[ind])  # [n_sample, 2]
    x_curr = x_curr - vel * dt
    traj_x.append(x_curr)


def solve_HJ(ndim, egno, epsl, nx, ny, nt, x_period, y_period, T, x_arr, 
             c_on_rho, time_step_per_PDHG, stepsz_param, N_maxiter, print_freq, eps):
  dt = T / (nt-1)
  dx = x_period / (nx)
  dy = y_period / (ny)
  if ndim == 1:
    period_spatial = (x_period,)
    dspatial = (dx, )
    nspatial = (nx, )
  else:
    period_spatial = (x_period, y_period)
    dspatial = (dx, dy)
    nspatial = (nx, ny)
  print('period_spatial: ', period_spatial)
  print('dspatial: ', dspatial)
  print('nspatial: ', nspatial)

  J = set_up_J(egno, ndim, period_spatial)
  fns_dict = set_up_example_fns(egno, ndim, FLAGS.numerical_L_ind)
  g = J(x_arr)  # [1, nx] or [1, nx, ny]
  print('shape of g: ', g.shape)

  # fv for preconditioning
  fv = compute_Dxx_fft_fv(ndim, nspatial, dspatial)
  if ndim == 1:
    fn_update_primal = lambda phi_prev, rho_prev, c_on_rho, alp_prev, tau, dt, dspatial, fns_dict, fv, epsl, x_arr, t_arr: \
      pdhg.update_primal_1d(phi_prev, rho_prev, c_on_rho, alp_prev, tau, dt, dspatial, fns_dict, fv, epsl, x_arr, t_arr, 
                            C = FLAGS.C, pow = FLAGS.pow, Ct = FLAGS.Ct)
    if FLAGS.method == 0:
      fn_update_dual = pdhg.update_dual_alternative
    else:
      fn_update_dual = pdhg.update_dual_Newton_1d
  else:
    fn_update_primal = pdhg.update_primal_2d
    if FLAGS.method == 0:
      fn_update_dual = pdhg.update_dual_alternative
    else:
      fn_update_dual = pdhg.update_dual_Newton_2d

  results, errs_all = PDHG_multi_step(fn_update_primal, fn_update_dual, fns_dict, g, x_arr, 
                                       ndim, nt, nspatial, dt, dspatial, c_on_rho, time_step_per_PDHG = time_step_per_PDHG,
                                       epsl = epsl, stepsz_param=stepsz_param, fv=fv,
                                       N_maxiter = N_maxiter, print_freq = print_freq, eps = eps, tfboard = FLAGS.tfboard)
  return results, errs_all

def main(argv):
  for key, value in FLAGS.__flags.items():
    print(value.name, ": ", value._value, flush=True)

  nt = FLAGS.nt
  nx = FLAGS.nx
  ny = FLAGS.ny
  ndim = FLAGS.ndim
  egno = FLAGS.egno
  c_on_rho = FLAGS.c_on_rho
  epsl = FLAGS.epsl
  T = FLAGS.T

  if ndim == 1:
    filename_prefix = 'nt{}_nx{}'.format(nt, nx)
  elif ndim == 2:
    filename_prefix = 'nt{}_nx{}_ny{}'.format(nt, nx, ny)
  else:
    raise NotImplementedError

  time_stamp = datetime.now(pytz.timezone('America/Los_Angeles')).strftime("%Y%m%d-%H%M%S")
  logging.info("current time: " + time_stamp)
  save_dir = './check_points/{}'.format(time_stamp) + '/eg{}_{}d'.format(egno, ndim)
  save_plot_dir = './plots/{}'.format(time_stamp) + '/eg{}_{}d'.format(egno, ndim)

  if FLAGS.tfboard:
    results_dir = f'./tf_save/{filename_prefix}/'+ time_stamp
    print("tf foldername: ", results_dir)
    file_writer = tf.summary.create_file_writer(results_dir)
    file_writer.set_as_default()

  x_period, y_period = 2, 2
  if ndim == 1:
    x_arr = jnp.linspace(0.0, x_period, num = nx, endpoint = False)[None,:,None]  # [1, nx, 1]
    t_arr = jnp.linspace(0.0, T, num = nt)[:,None]  # [nt, 1]
  else:
    x_arr = jnp.linspace(0.0, x_period, num = nx, endpoint = False)  # [nx]
    y_arr = jnp.linspace(0.0, y_period, num = ny, endpoint = False)  # [ny]
    x_mesh, y_mesh = jnp.meshgrid(x_arr, y_arr, indexing='ij')  # [nx, ny]
    x_arr = jnp.stack([x_mesh, y_mesh], axis = -1)[None,...]  # [1, nx, ny, 2]
    t_arr = jnp.linspace(0.0, T, num = nt)[:,None,None]  # [nt, 1, 1]

  
  if FLAGS.load:
    results, errs_all = load_solution(save_dir, filename_prefix)
  else:
    results, errs_all = solve_HJ(ndim, egno, epsl, nx, ny, nt, x_period, y_period, T, x_arr, 
            c_on_rho, FLAGS.time_step_per_PDHG, FLAGS.stepsz_param, FLAGS.N_maxiter, FLAGS.print_freq, FLAGS.eps)
    if FLAGS.save:
      save(save_dir, filename_prefix, (results, errs_all))

  # results: list of (num_iter, phi, rho, alp)
  phi = results[-1][1]
  rho = results[-1][2]
  alp = results[-1][3]

  if FLAGS.plot:
    if ndim == 1:
      plot_fn = utils_plot.plot_solution_1d
      alp_titles = ['alp_1', 'alp_2']
    elif ndim == 2:
      plot_fn = utils_plot.plot_solution_2d
      alp_titles = ['alp_11', 'alp_12', 'alp_21', 'alp_22']
    else:
      raise NotImplementedError
    fig_phi = plot_fn(phi, x_arr, t_arr, tfboard = True)
    utils_plot.save_fig(fig_phi, 'phi', tfboard = FLAGS.tfboard, foldername = save_plot_dir)
    fig_rho = plot_fn(rho, x_arr, t_arr[:-1,...], tfboard = True)
    utils_plot.save_fig(fig_rho, 'rho', tfboard = FLAGS.tfboard, foldername = save_plot_dir)
    for i in range(2**ndim):
      fig_alp = plot_fn(alp[i,...,0], x_arr, t_arr[:-1,...], tfboard = True)
      utils_plot.save_fig(fig_alp, alp_titles[i] + '_x', tfboard = FLAGS.tfboard, foldername = save_plot_dir)
      if ndim == 2:
        fig_alp = plot_fn(alp[i,...,1], x_arr, t_arr[:-1,...], tfboard = True)
        utils_plot.save_fig(fig_alp, alp_titles[i] + '_y', tfboard = FLAGS.tfboard, foldername = save_plot_dir)
    
    if FLAGS.plot_traj_num > 0:
      # compute trajectories of x
      if ndim == 1:
        x_samples = jnp.linspace(0.0, x_period, num = FLAGS.plot_traj_num, endpoint = False) # [n_sample, 1]

  
  print('phi: ', phi)
  print('end')





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
  flags.DEFINE_integer('print_freq', 10000, 'print frequency')
  flags.DEFINE_float('stepsz_param', 0.1, 'default step size constant')
  flags.DEFINE_float('eps', 1e-6, 'the error threshold')

  flags.DEFINE_boolean('save', True, 'if save to pickle')
  flags.DEFINE_boolean('tfboard', False, 'if use tfboard')
  flags.DEFINE_boolean('load', False, 'if load from pickle')
  flags.DEFINE_boolean('plot', False, 'if plot')
  flags.DEFINE_integer('plot_traj_num', 0, 'number of trajectories to plot')

  flags.DEFINE_float('C', 1.0, 'constant in preconditioning')
  flags.DEFINE_float('pow', 1.0, 'power in preconditioning')
  flags.DEFINE_float('Ct', 1.0, 'constant in preconditioning')
  flags.DEFINE_integer('numerical_L_ind', 0, 'index of numerical L')

  flags.DEFINE_integer('method', 0, 'method: 0 for alternative update rho and alp, 1 for Newton')


  
  app.run(main)
