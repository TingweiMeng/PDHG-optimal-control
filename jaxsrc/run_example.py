import jax.numpy as jnp
from absl import app, flags, logging
from set_fns import set_up_example_fns, set_up_J
import pytz
from datetime import datetime
from utils.utils_pdhg_solver import PDHG_multi_step
from solver import save, load_solution
import update_fns_in_pdhg as pdhg
from utils.utils_precond import compute_Dxx_fft_fv
from update_fns_in_pdhg import get_f_vals_1d, get_f_vals_2d
import solver
import tensorflow as tf
import os
import utils.utils_plot as utils_plot
import scipy.interpolate as interpolate
import numpy as np
import matplotlib.pyplot as plt

def compute_traj_1d(x_init, alp, f_fn, nt, x_arr, t_arr, x_period, T, epsl = 0.0):
  ''' dx_t = f(alp_t,x_t,t)dt + sqrt{2 * epsl} dW_t, alp_t = alp(x_t,t) 
  NOTE: the time direction in this function is different from PDEs, time of f is the same with PDEs
  @ parameters:
    x_arr: [nx], t_arr: [nt], alp: [2, nt-1, nx], x_init: [n_sample]
  @ returns:
    traj_alp: [nt-1, n_sample, 1], traj_x: [nt, n_sample]'''
  traj_alp = []
  traj_x = [x_init]
  x_curr = x_init  # [n_sample]
  for i in range(nt-1):
    ind = i
    dt = t_arr[ind + 1] - t_arr[ind]
    # interpolation
    alp_1 = jnp.interp(x_curr, x_arr, alp[0,ind,:], period = x_period)[:,None]  # [n_sample, 1]
    alp_2 = jnp.interp(x_curr, x_arr, alp[1,ind,:], period = x_period)[:,None]  # [n_sample, 1]
    traj_alp.append(alp_1 + alp_2)
    # convert x_curr to [0,period]
    f1, f2 = get_f_vals_1d(f_fn, (alp_1, alp_2), x_curr[:,None] % x_period, T - t_arr[ind])  # [n_sample, ]
    vel = f1 + f2  # [n_sample, ]
    # vel = f_fn(alp_1, x_curr % x_period, T - t_arr[ind]) + f_fn(alp_2, x_curr % x_period, T - t_arr[ind])  # [n_sample]
    x_curr = x_curr + vel * dt + jnp.sqrt(2 * epsl * dt) * np.random.normal(size = x_curr.shape)
    traj_x.append(x_curr)
  traj_alp = jnp.stack(traj_alp, axis = 0)  # [nt-1, n_sample, 1]
  traj_x = jnp.stack(traj_x, axis = 0)  # [nt, n_sample]
  return traj_alp, traj_x

def extend_bdry_2d(x_arr, x_min, x_max, val_arr, period, axis, bc, center = False):
  ''' extend bdry periodically for interpolation
  @ parameters:
    x_arr: [n_pts], val_arr: [:,n1, n2, val_dim], period: float, 
    axis: int (along which axis to extend, if axis==1, n_pts==n1 and x_arr==x1_arr; if axis==2, n_pts==n2 and x_arr==x2_arr)
    bc: int (0 for periodic, 1 for Neumann, 2 for Dirichlet)
    center: bool (if the x_arr is centered at 0)
  @ returns:
    x_arr: [n_ext+1], val_arr: [:,n_ext+1, n2, val_dim] (if axis==1) or [:,n1, n_ext+1, val_dim] (if axis==2)
  '''
  # compute how many periods and the bounds for the extended array
  if center:
    n_period_lb = int(np.floor(x_min / period + 0.5))
    n_period_ub = int(np.floor(x_max / period + 0.5))
    # print('x_min: ', x_min, 'x_max: ', x_max, 'n_period_lb: ', n_period_lb, 'n_period_ub: ', n_period_ub, flush=True)
  else:
    n_period_lb = int(np.floor(x_min / period))
    n_period_ub = int(np.floor(x_max / period))
  # put the original array in the middle
  n_period_lb = min(n_period_lb, 0)
  n_period_ub = max(n_period_ub, 0)
  # compute the number of periods
  n_period = n_period_ub - n_period_lb + 1
  if bc == 0:  # periodic
    val_arr = np.concatenate([val_arr] * n_period, axis = axis)  # [n_ext, n2, val_dim] or [n1, n_ext, val_dim]
    if axis == 1:
      val_arr = np.concatenate([val_arr, val_arr[:,:1,:,:]], axis = 1)  # [:, n_ext+1, n2, val_dim]
    elif axis == 2:
      val_arr = np.concatenate([val_arr, val_arr[:,:,:1,:]], axis = 2)  # [:, n1, n_ext+1, val_dim]
    else:
      raise NotImplementedError
  else:
    num_per_period = val_arr.shape[axis]  # n1 or n2
    if axis == 1:
      val_left = val_arr[:,0:1,...]  # [..., 1, n2, val_dim]
      val_right = val_arr[:,-1:,...]  # [..., 1, n2, val_dim]
      if bc == 2:  # Dirichlet
        val_left = np.zeros_like(val_left)
        val_right = np.zeros_like(val_right)
    elif axis == 2:
      val_left = val_arr[:,:,0:1,:]  # [..., n1, 1, val_dim]
      val_right = val_arr[:,:,-1:,:]  # [..., n1, 1, val_dim]
      if bc == 2:  # Dirichlet
        val_left = np.zeros_like(val_left)
        val_right = np.zeros_like(val_right)
    else:
      raise NotImplementedError
    if n_period_lb < 0:
      val_arr = np.concatenate([val_left] * (-n_period_lb) * num_per_period + [val_arr], axis = axis)  # [..., n_ext1, n2, val_dim] or [..., n1, n_ext1, val_dim]
    if n_period_ub > 0:
      val_arr = np.concatenate([val_arr] + [val_right] * (n_period_ub * num_per_period), axis = axis)  # [..., n_ext2, n2, val_dim] or [..., n1, n_ext2, val_dim]
    val_arr = np.concatenate([val_arr, val_right], axis = axis)  # [..., n_ext+1, n2, val_dim] or [..., n1, n_ext+1, val_dim]
  # compute the extended x_arr
  x_arr_new = np.stack([x_arr] * n_period, axis = 0)  # [n_period, n_pts]
  x_arr_new += np.arange(n_period_lb, n_period_ub+1)[:,None] * period  # [n_period, n_pts]
  x_arr_new = np.reshape(x_arr_new, (-1,))  # [n_ext]
  # repeat bdry
  x_arr_new = np.concatenate([x_arr_new, x_arr_new[0:1] + period * n_period], axis = 0)  # [n_ext+1]
  # print('shape of x_arr_new: ', x_arr_new.shape)
  # print('shape of val_arr: ', val_arr.shape, flush=True)
  # print('extended min: ', x_arr_new[0], 'extended max: ', x_arr_new[-1], 'required min: ', x_min, 'required max: ', x_max, flush=True)
  return x_arr_new, val_arr


def compute_traj_2d(x_init, alp, f_fn, nt, x1_arr, x2_arr, t_arr, x_period, y_period, T, bc, center, epsl = 0.0):
  ''' dx_t = f(alp_t,x_t,t)dt + sqrt{2 * epsl} dW_t, alp_t = alp(x_t,t)
  @ parameters:
    x_arr: [nx, ny, 2], t_arr: [nt], alp: [4, nt-1, nx, ny, n_ctrl], x_init: [n_sample, 2] 
    center: tuple of bool (if the x_arr is centered at 0)
  @ returns:
    traj_alp: [nt-1, n_sample, n_ctrl], traj_x: [nt, n_sample, 2]  
  '''
  traj_alp = []
  x_init = np.array(x_init)
  x1_arr = np.array(x1_arr)
  x2_arr = np.array(x2_arr)
  alp = np.array(alp)
  traj_x = [x_init]
  x_curr = x_init  # [n_sample, 2]
  bc_x, bc_y = bc
  center_x, center_y = center
  for i in range(nt-1):
    ind = i
    dt = t_arr[ind + 1] - t_arr[ind]
    # check bound and extend bdry
    x_curr_min = np.min(x_curr, axis = 0)  # [2]
    x_curr_max = np.max(x_curr, axis = 0)  # [2]
    x1_grid_curr, alp_curr = extend_bdry_2d(x1_arr, x_curr_min[0], x_curr_max[0], alp[:,ind,:,:,:], x_period, bc=bc_x, axis = 1, center = center_x)
    x2_grid_curr, alp_curr = extend_bdry_2d(x2_arr, x_curr_min[1], x_curr_max[1], alp_curr, y_period, bc=bc_y, axis = 2, center = center_y)
    # interpolation
    alp1_x = interpolate.interpn((x1_grid_curr, x2_grid_curr), alp_curr[0], x_curr, method='linear')  # [n_sample, n_ctrl]
    alp2_x = interpolate.interpn((x1_grid_curr, x2_grid_curr), alp_curr[1], x_curr, method='linear')  # [n_sample, n_ctrl]
    alp1_y = interpolate.interpn((x1_grid_curr, x2_grid_curr), alp_curr[2], x_curr, method='linear')  # [n_sample, n_ctrl]
    alp2_y = interpolate.interpn((x1_grid_curr, x2_grid_curr), alp_curr[3], x_curr, method='linear')  # [n_sample, n_ctrl]
    traj_alp.append(alp1_x + alp2_x + alp1_y + alp2_y)
    x_curr_in_period = x_curr % np.array([x_period, y_period])  # [n_sample, 2]
    f1_x, f2_x, f1_y, f2_y = get_f_vals_2d(f_fn, (alp1_x, alp2_x, alp1_y, alp2_y), x_curr_in_period, T - t_arr[ind])
    vel = np.array([f1_x + f2_x, f1_y + f2_y]).T  # [n_sample, 2]
    # vel = f_fn(alp1_x, x_curr_in_period, T - t_arr[ind]) + f_fn(alp2_x, x_curr_in_period, T - t_arr[ind]) + \
    #       f_fn(alp1_y, x_curr_in_period, T - t_arr[ind]) + f_fn(alp2_y, x_curr_in_period, T - t_arr[ind])  # [n_sample, 2]
    x_curr = x_curr + vel * dt + np.sqrt(2 * epsl * dt) * np.random.normal(size = x_curr.shape)
    traj_x.append(x_curr)
  traj_alp = np.stack(traj_alp, axis = 0)  # [nt-1, n_sample, n_ctrl]
  traj_x = np.stack(traj_x, axis = 0)  # [nt, n_sample, 2]
  return traj_alp, traj_x

# def compute_traj_1d_fwd(x_init, alp, fn_f, nt, x_arr, t_arr, x_period, epsl = 0.0):
#   ''' dx_t = f(alp,x,t)dt + sqrt{2 * epsl} dW_t, alp = alp(x,t)
#   @ parameters:
#     x_arr: [nx], t_arr: [nt], alp: [nt-1, nx], x_init: [n_sample]
#   @ returns:
#     traj_alp: [nt-1, n_sample], traj_x: [nt, n_sample]'''
#   traj_alp = []
#   traj_x = [x_init]
#   x_curr = x_init  # [n_sample]
#   # use alp at dt to approximate alp at 0
#   alp = jnp.concatenate([alp[:1,...], alp], axis = 0)  # [nt, nx+1]
#   for i in range(nt-1):
#     ind = i
#     dt = t_arr[ind+1] - t_arr[ind]
#     # interpolation
#     alp_x = jnp.interp(x_curr, x_arr, alp[ind,:], period = x_period)  # [n_sample]
#     traj_alp.append(alp_x)
#     vel = fn_f(alp_x, x_curr, t_arr[ind])  # [n_sample]
#     x_curr = x_curr + vel * dt + jnp.sqrt(2 * epsl * dt) * jnp.random.normal(size = x_curr.shape)
#     traj_x.append(x_curr)
#   traj_alp = jnp.stack(traj_alp, axis = 0)  # [nt-1, n_sample]
#   traj_x = jnp.stack(traj_x, axis = 0)  # [nt, n_sample]
#   return traj_alp, traj_x

# def compute_traj_2d_fwd(x_init, alp, fn_f, nt, x1_arr, x2_arr, t_arr, x_period, y_period, epsl = 0.0):
#   ''' dx_t = f(alp,x,t)dt + sqrt{2 * epsl} dW_t, alp = alp(x,t)
#   @ parameters:
#     x_arr: [nx, ny, 2], t_arr: [nt], alp: [nt-1, nx, ny, nstate], x_init: [n_sample, 2]
#   @ returns:
#     traj_alp: [nt-1, n_sample, nstate], traj_x: [nt, n_sample, 2]  
#   '''
#   traj_alp = []
#   traj_x = [x_init]
#   x_curr = x_init  # [n_sample, 2]
#   # use alp at dt to approximate alp at 0
#   alp = np.concatenate([alp[:1,...], alp], axis = 0)  # [nt, nx+1, ny+1, nstate]
#   for i in range(nt-1):
#     ind = i
#     dt = t_arr[ind+1] - t_arr[ind]
#     # check bound and extend bdry
#     x1_grid_curr, alp_curr = extend_bdry_2d(x1_arr, alp[ind,:,:,:], x_period, axis = 0)
#     x2_grid_curr, alp_curr = extend_bdry_2d(x2_arr, alp_curr, y_period, axis = 1)
#     # interpolation
#     alp_x = interpolate.interpn((x1_grid_curr, x2_grid_curr), alp_curr, x_curr, method='linear')  # [n_sample, nstate]
#     traj_alp.append(alp_x)
#     vel = fn_f(alp_x, x_curr, t_arr[ind])  # [n_sample, 2]
#     x_curr = x_curr + vel * dt + jnp.sqrt(2 * epsl * dt) * jnp.random.normal(size = x_curr.shape)
#     traj_x.append(x_curr)
#   traj_alp = np.stack(traj_alp, axis = 0)  # [nt-1, n_sample, nstate]
#   traj_x = np.stack(traj_x, axis = 0)  # [nt, n_sample, 2]
#   return traj_alp, traj_x

def solve_HJ(ndim, n_ctrl, egno, epsl, fns_dict, nx, ny, nt, x_period, y_period, T, x_arr, 
             c_on_rho, time_step_per_PDHG, stepsz_param, N_maxiter, print_freq, eps, bc, save_dir = None):
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
  g = J(x_arr)  # [1, nx] or [1, nx, ny]
  print('shape of g: ', g.shape)
  # plot g
  if save_dir is not None:
    filename = save_dir + '/init_cond.png'
    fig = plt.figure()
    plt.contourf(x_arr[0,...,0], x_arr[0,...,1], g[0])
    plt.colorbar()
    fig.savefig(filename)

  # fv for preconditioning
  fv = compute_Dxx_fft_fv(ndim, nspatial, dspatial, bc)
  if ndim == 1:
    fn_update_primal = lambda phi_prev, rho_prev, c_on_rho, alp_prev, tau, dt, dspatial, fns_dict, fv, epsl, x_arr, t_arr: \
      pdhg.update_primal_1d(phi_prev, rho_prev, c_on_rho, alp_prev, tau, dt, dspatial, fns_dict, fv, epsl, x_arr, t_arr, bc,
                            C = FLAGS.C, pow = FLAGS.pow, Ct = FLAGS.Ct)
    if FLAGS.method == 0:
      fn_update_dual = lambda phi_bar, rho_prev, c_on_rho, alp_prev, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr, ndim, eps: \
        pdhg.update_dual_alternative(phi_bar, rho_prev, c_on_rho, alp_prev, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr, ndim, bc, eps = eps)
    else:
      # fn_update_dual = pdhg.update_dual_Newton_1d
      raise NotImplementedError
  else:
    fn_update_primal = lambda phi_prev, rho_prev, c_on_rho, alp_prev, tau, dt, dspatial, fns_dict, fv, epsl, x_arr, t_arr: \
      pdhg.update_primal_2d(phi_prev, rho_prev, c_on_rho, alp_prev, tau, dt, dspatial, fns_dict, fv, epsl, x_arr, t_arr, bc,
                            C = FLAGS.C, pow = FLAGS.pow, Ct = FLAGS.Ct)
    if FLAGS.method == 0:
      fn_update_dual = lambda phi_bar, rho_prev, c_on_rho, alp_prev, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr, ndim, eps: \
        pdhg.update_dual_alternative(phi_bar, rho_prev, c_on_rho, alp_prev, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr, ndim, bc, eps = eps)
    else:
      # fn_update_dual = pdhg.update_dual_Newton_2d
      raise NotImplementedError

  results, errs_all = PDHG_multi_step(fn_update_primal, fn_update_dual, fns_dict, g, x_arr,
                                       ndim, nt, nspatial, dt, dspatial, c_on_rho, time_step_per_PDHG = time_step_per_PDHG,
                                       epsl = epsl, stepsz_param=stepsz_param, fv=fv, n_ctrl=n_ctrl,
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

  if egno == 30:  # Newton, n_ctrl = 1, ndim = 2
    assert ndim == 2
    n_ctrl = 1
    # 0 for periodic, 1 for Neumann, 2 for Dirichlet
    bc = (1,0)
    x_period, y_period = 4, 2
    x_centered, y_centered = True, True
  else:
    n_ctrl = ndim
    x_period, y_period = 2, 2
    x_centered, y_centered = False, False
    if ndim == 1:
      bc = 0
    else:
      bc = (0,0)

  if ndim == 1:
    filename_prefix = 'nt{}_nx{}'.format(nt, nx)
  elif ndim == 2:
    filename_prefix = 'nt{}_nx{}_ny{}'.format(nt, nx, ny)
  else:
    raise NotImplementedError

  if FLAGS.load:
    assert FLAGS.load_timestamp != ''
    time_stamp = FLAGS.load_timestamp
  else:
    time_stamp = datetime.now(pytz.timezone('America/Los_Angeles')).strftime("%Y%m%d-%H%M%S")
    logging.info("current time: " + time_stamp)
    
  save_dir = './check_points/{}'.format(time_stamp) + '/eg{}_{}d'.format(egno, ndim)
  save_plot_dir = './plots/{}'.format(time_stamp) + '/eg{}_{}d'.format(egno, ndim)
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  if not os.path.exists(save_plot_dir):
    os.makedirs(save_plot_dir)

  if FLAGS.tfboard:
    results_dir = f'./tf_save/{filename_prefix}/'+ time_stamp
    print("tf foldername: ", results_dir)
    file_writer = tf.summary.create_file_writer(results_dir)
    file_writer.set_as_default()

  fns_dict = set_up_example_fns(egno, ndim, FLAGS.numerical_L_ind)
  if ndim == 1:
    x_arr = jnp.linspace(0.0, x_period, num = nx, endpoint = False)[None,:,None]  # [1, nx, 1]
    if x_centered:
      x_arr = x_arr - x_period / 2
    t_arr = jnp.linspace(0.0, T, num = nt)[:,None]  # [nt, 1]
  else:
    x1_arr = jnp.linspace(0.0, x_period, num = nx, endpoint = False)  # [nx]
    if x_centered:
      x1_arr = x1_arr - x_period / 2
    x2_arr = jnp.linspace(0.0, y_period, num = ny, endpoint = False)  # [ny]
    if y_centered:
      x2_arr = x2_arr - y_period / 2
    x_mesh, y_mesh = jnp.meshgrid(x1_arr, x2_arr, indexing='ij')  # [nx, ny]
    x_arr = jnp.stack([x_mesh, y_mesh], axis = -1)[None,...]  # [1, nx, ny, 2]
    t_arr = jnp.linspace(0.0, T, num = nt)[:,None,None]  # [nt, 1, 1]

  
  if FLAGS.load:
    results, errs_all = load_solution(save_dir, filename_prefix)
  else:
    results, errs_all = solve_HJ(ndim, n_ctrl, egno, epsl, fns_dict, nx, ny, nt, x_period, y_period, T, x_arr, 
            c_on_rho, FLAGS.time_step_per_PDHG, FLAGS.stepsz_param, FLAGS.N_maxiter, FLAGS.print_freq, FLAGS.eps, bc, save_dir = save_plot_dir)
    if FLAGS.save:
      save(save_dir, filename_prefix, (results, errs_all))

  # results: list of (num_iter, phi, rho, alp)
  phi = results[-1][1]
  alp = results[-1][3]

  if FLAGS.plot:
    if ndim == 1:
      plot_phi_fn = utils_plot.plot_solution_1d
      plot_alp_fn = utils_plot.plot_solution_1d
      alp_titles = ['alp_1', 'alp_2']
    elif ndim == 2:
      plot_phi_fn = utils_plot.plot_solution_2d
      plot_alp_fn = utils_plot.plot_solution_2d
      alp_titles = ['alp_11', 'alp_12', 'alp_21', 'alp_22']
    else:
      raise NotImplementedError
    
    if egno == 30:
      num_cols = 1
    else:
      num_cols = 2

    fig_phi = plot_phi_fn(phi, x_arr, t_arr, tfboard = FLAGS.tfboard, num_cols = num_cols)
    utils_plot.save_fig(fig_phi, 'phi', tfboard = FLAGS.tfboard, foldername = save_plot_dir)
    for i in range(2**ndim):
      fig_alp = plot_alp_fn(alp[i,...,0], x_arr, t_arr[:-1,...], tfboard = FLAGS.tfboard, num_cols = num_cols)
      utils_plot.save_fig(fig_alp, alp_titles[i] + '_x', tfboard = FLAGS.tfboard, foldername = save_plot_dir)
      if n_ctrl == 2:
        fig_alp = plot_alp_fn(alp[i,...,1], x_arr, t_arr[:-1,...], tfboard = FLAGS.tfboard, num_cols = num_cols)
        utils_plot.save_fig(fig_alp, alp_titles[i] + '_y', tfboard = FLAGS.tfboard, foldername = save_plot_dir)
    # plot sum of alp
    alp_sum = jnp.sum(alp, axis = 0)  # [nt-1, nx, ny, nstate] or [nt-1, nx, nstate]
    fig_alp_sum = plot_alp_fn(alp_sum[...,0], x_arr, t_arr[:-1,...], tfboard = FLAGS.tfboard, num_cols = num_cols)
    utils_plot.save_fig(fig_alp_sum, 'alp_sum_x', tfboard = FLAGS.tfboard, foldername = save_plot_dir)
    if n_ctrl == 2:
      fig_alp_sum = plot_alp_fn(alp_sum[...,1], x_arr, t_arr[:-1,...], tfboard = FLAGS.tfboard, num_cols = num_cols)
      utils_plot.save_fig(fig_alp_sum, 'alp_sum_y', tfboard = FLAGS.tfboard, foldername = save_plot_dir)
    
    if FLAGS.plot_traj_num_1d > 0:
      center = (x_centered, y_centered)
      # compute trajectories of x. NOTE: time direction of trajs is different from PDE
      # reverse the time direction to be consistent with control
      alp_combined = alp[:,::-1,...]  # [2,nt-1, nx, n_ctrl] or [4,nt-1, nx, ny, n_ctrl]
      if egno == 30:  # for Newton, plot samples are with x' = 0
        y_plot_lb = -y_period/2
        y_plot_ub = y_period/2
        x_samples = jnp.linspace(y_plot_lb, y_plot_ub, num = FLAGS.plot_traj_num_1d)[:,None] # [n_sample, 1]
        x_samples = jnp.pad(x_samples, ((0,0), (1,0)), mode = 'constant', constant_values = 0.5)  # [n_sample, 2]  (x'(0)=0)
        x_samples = jnp.concatenate([x_samples, -x_samples], axis = 0)  # [2*n_sample, 2]
        traj_alp, traj_x = compute_traj_2d(x_samples, alp_combined, fns_dict.f_fn, nt, x1_arr, x2_arr, t_arr[:,0], 
                                           x_period, y_period, T, bc, center, epsl)
        fig_traj_vel = utils_plot.plot_traj_1d(traj_x[...,0], t_arr[:,0], tfboard = FLAGS.tfboard)
        utils_plot.save_fig(fig_traj_vel, 'traj_vel', tfboard = FLAGS.tfboard, foldername = save_plot_dir)
        fig_traj_pos = utils_plot.plot_traj_1d(traj_x[...,1], t_arr[:,0], tfboard = FLAGS.tfboard)
        utils_plot.save_fig(fig_traj_pos, 'traj_pos', tfboard = FLAGS.tfboard, foldername = save_plot_dir)
        fig_traj_acc = utils_plot.plot_traj_1d(traj_alp[...,0], t_arr[:-1,0], tfboard = FLAGS.tfboard)
        utils_plot.save_fig(fig_traj_acc, 'traj_acc', tfboard = FLAGS.tfboard, foldername = save_plot_dir)
      else:
        x_plot_lb = 0
        x_plot_ub = x_period
        y_plot_lb = 0
        y_plot_ub = y_period
        if ndim == 1:
          x_samples = jnp.linspace(x_plot_lb, x_plot_ub, num = FLAGS.plot_traj_num_1d) # [n_sample]
          traj_alp, traj_x = compute_traj_1d(x_samples, alp_combined[...,0], fns_dict.f_fn, nt, x_arr[0,:,0], t_arr[:,0], x_period, T, epsl)
          fig_traj_x = utils_plot.plot_traj_1d(traj_x, t_arr[:,0], tfboard = FLAGS.tfboard)
          utils_plot.save_fig(fig_traj_x, 'traj_x', tfboard = FLAGS.tfboard, foldername = save_plot_dir)
        elif ndim == 2:
          x_samples = jnp.linspace(x_plot_lb, x_plot_ub, num = FLAGS.plot_traj_num_1d)
          y_samples = jnp.linspace(y_plot_lb, y_plot_ub, num = FLAGS.plot_traj_num_1d)
          x_mesh, y_mesh = jnp.meshgrid(x_samples, y_samples, indexing='ij')  # [n_sample, n_sample]
          x_samples = jnp.stack([x_mesh.flatten(), y_mesh.flatten()], axis = -1)  # [n_sample**2, 2]
          traj_alp, traj_x = compute_traj_2d(x_samples, alp_combined, fns_dict.f_fn, nt, x1_arr, x2_arr, t_arr[:,0], x_period, y_period, T, bc, center, epsl)
          fig_traj_x = utils_plot.plot_traj_2d(traj_x, tfboard = FLAGS.tfboard)
          utils_plot.save_fig(fig_traj_x, 'traj_x', tfboard = FLAGS.tfboard, foldername = save_plot_dir)
        
        if n_ctrl == 1:
          fig_traj_alp = utils_plot.plot_traj_1d(traj_alp[...,0], t_arr[:-1,0], tfboard = FLAGS.tfboard)
          utils_plot.save_fig(fig_traj_alp, 'traj_alp', tfboard = FLAGS.tfboard, foldername = save_plot_dir)
        else:
          fig_traj_alp = utils_plot.plot_traj_2d(traj_alp, tfboard = FLAGS.tfboard)
          utils_plot.save_fig(fig_traj_alp, 'traj_alp', tfboard = FLAGS.tfboard, foldername = save_plot_dir)
  
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
  flags.DEFINE_string('load_timestamp', '', 'the timestamp of the folder to load from')
  flags.DEFINE_boolean('plot', False, 'if plot')
  flags.DEFINE_integer('plot_traj_num_1d', 0, 'number of trajectories to plot')

  flags.DEFINE_float('C', 1.0, 'constant in preconditioning')
  flags.DEFINE_float('pow', 1.0, 'power in preconditioning')
  flags.DEFINE_float('Ct', 1.0, 'constant in preconditioning')
  flags.DEFINE_integer('numerical_L_ind', 0, 'index of numerical L')

  flags.DEFINE_integer('method', 0, 'method: 0 for alternative update rho and alp, 1 for Newton')


  
  app.run(main)
