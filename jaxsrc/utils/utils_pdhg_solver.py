import jax.numpy as jnp
import utils.utils as utils
from einshape import jax_einshape as einshape
import tensorflow as tf
import utils.utils as utils
from solver import save, load_middle_solution


def PDHG_solver_oneiter(fn_update_primal, fn_update_dual, fns_dict, phi0, rho0, alp0, x_arr, t_arr, 
                   ndim, dt, dspatial, c_on_rho, epsl = 0.0, stepsz_param=0.9, fv=None,
                   N_maxiter = 1000000, print_freq = 1000, eps = 1e-6, tfboard = False, tfrecord_ind = 0):
  '''
  @ parameters:
    fn_update_primal: function to update primal variable, takes p, d, delta_p, and other parameters, 
                        output p_next, and other parameters
    fn_update_dual: function to update dual variable, takes d, p, delta_d, and other parameters
                        output d_next, and other parameters
    fns_dict: dict of functions, see the function set_up_example_fns in solver.py
    phi0: [nt, nx] for 1d, [nt, nx, ny] for 2d
    rho0: [nt-1, nx] for 1d, [nt-1, nx, ny] for 2d
    alp0: (alp1, alp2), where alp1 and alp2 are [nt-1, nx, dim_ctrl] for 1d
          (alp1_x, alp2_x, alp1_y, alp2_y), where alp1_x, alp2_x, alp1_y, alp2_y are [nt-1, nx, ny, dim_ctrl] for 2d
    x_arr: array that can be broadcasted to [1, nx, 1] for 1d, [1, nx, ny, 2] for 2d
    t_arr: array that can be broadcasted to [nt-1, 1] for 1d, [nt-1, 1, 1] for 2d
    ndim: int, 1 or 2
    dt, c_on_rho: scalar
    dspatial: (dx, dy) for 2d, (dx,) for 1d
    epsl: scalar, diffusion coefficient
    stepsz_param: scalar, step size parameter
    fv: [nx] for 1d, [nx, ny] for 2d, FFT of Laplacian operator, or None if do not use preconditioning
    N_maxiter: int, maximum number of iterations
    print_gap: int, gap for printing and saving results
    eps: scalar, stopping criterion
    tfboard: bool, whether to use tensorboard
    tfrecord_ind: int, index for tensorboard
  @ returns:
    results_all: list of tuples, each tuple is (iter, phi_next, rho_next, alp_next)
    error_all: [N_maxiter//print_freq, 2], primal error, dual error
  '''
  phi_prev = phi0
  rho_prev = rho0
  alp_prev = alp0

  scale = 1.5  # adjust the stepsize to balance the primal and dual updates
  tau_phi = stepsz_param / scale
  tau_rho = stepsz_param * scale
  
  error_all = []
  results_all = []

  for i in range(N_maxiter):
    err_tol = eps
    phi_next = fn_update_primal(phi_prev, rho_prev, c_on_rho, alp_prev, tau_phi, dt, dspatial, fns_dict, fv, epsl, x_arr, t_arr)
    # extrapolation
    phi_bar = 2 * phi_next - phi_prev
    rho_next, alp_next = fn_update_dual(phi_bar, rho_prev, c_on_rho, alp_prev, tau_rho, dt, dspatial, epsl, 
                                      fns_dict, x_arr, t_arr, ndim, eps = err_tol)
    # primal error
    err1 = jnp.linalg.norm(phi_next - phi_prev) / jnp.linalg.norm(phi_prev)
    # err2: dual error
    err2 = jnp.linalg.norm(rho_next - rho_prev) / jnp.linalg.norm(rho_prev)
    for alp_p, alp_n in zip(alp_prev, alp_next):
      norm_alp = jnp.linalg.norm(alp_p)
      norm_err = jnp.linalg.norm(alp_p - alp_n)
      if norm_alp < 1e-6 and norm_err > 1e-6:
        err2 += norm_err
      elif norm_alp >= 1e-6:
        err2 += norm_err / norm_alp

    if tfboard:
      tf.summary.scalar('primal error', err1, step = tfrecord_ind + i)
      tf.summary.scalar('dual error', err2, step = tfrecord_ind + i)
    
    error = jnp.array([err1, err2])
    if error[0] < eps and error[1] < eps:
      print('PDHG converges at iter {}'.format(i), flush=True)
      break
    if jnp.any(jnp.isnan(phi_next)) or jnp.any(jnp.isnan(rho_next)):
      print("Nan error at iter {}".format(i))
      break
    if print_freq > 0 and i % print_freq == 0:
      results_all.append((i, phi_prev, rho_prev, alp_next))
      error_all.append(error)
      print('iteration {}, primal error {:.2E}, dual error {:.2E}, min rho {:.2f}, max rho {:.2f}'.format(i, 
                  error[0],  error[1], jnp.min(rho_next), jnp.max(rho_next)), flush = True)
    phi_prev = phi_next
    rho_prev = rho_next
    alp_prev = alp_next
  # print the final error
  print('iteration {}, primal error with prev step {:.2E}, dual error with prev step {:.2E}, eqt error {:.2E}'.format(i, error[0],  error[1],  error[2]), flush = True)
  results_all.append((i+1, phi_next, rho_next, alp_next))
  error_all.append(error)
  error_all = jnp.array(error_all)
  return results_all, error_all


def PDHG_multi_step(fn_update_primal, fn_update_dual, fns_dict, g, x_arr, 
                    ndim, nt, nspatial, dt, dspatial, c_on_rho, time_step_per_PDHG = 2,
                    epsl = 0.0, stepsz_param=0.9, n_ctrl = None, fv=None,
                    N_maxiter = 1000000, print_freq = 1000, eps = 1e-6, tfboard = False,
                    save_middle_dir = None, save_middle_prefix = None,
                    load_middle_dir = None, load_middle_prefix = None):
  '''
  @ parameters:
    fn_update_primal, fn_update_dual, fns_dict: see PDHG_solver_oneiter
    g: [1, nx] for 1d, [1, nx, ny] for 2d
    x_arr, ndim, nt, nspatial, c_on_rho, epsl, stepsz_param, fv: see PDHG_solver_oneiter
    dt: scalar, time step size
    dspatial: (dx,) for 1d, (dx, dy) for 2d
    time_step_per_PDHG: int, number of time steps per time block (run PDHG_solver_oneiter in each block)
    n_ctrl: int, number of control variables
    N_maxiter, print_freq, eps, tfboard: see PDHG_solver_oneiter
    save_middle_dir, save_middle_prefix: str, directory and prefix for saving middle results, or None
    load_middle_dir, load_middle_prefix: str, directory and prefix for loading middle results, or None
  @ returns:
    results_out: list of tuples, each tuple is (iter, phi_next, rho_next, alp_next)
    error_all: [N_maxiter//print_freq, 2], primal error, dual error
  '''
  if n_ctrl is None:
    n_ctrl = ndim
  assert (nt-1) % (time_step_per_PDHG-1) == 0  # make sure nt-1 is divisible by time_step_per_PDHG-1
  nt_PDHG = (nt-1) // (time_step_per_PDHG-1)
  phi0 = einshape("i...->(ki)...", g, k=time_step_per_PDHG)  # repeat each row of g to nt times, [nt, nx] or [nt, nx, ny]
  if ndim == 1:
    nx = nspatial[0]
    rho0 = jnp.zeros([time_step_per_PDHG-1, nx]) + c_on_rho
    alp1_0 = jnp.zeros([time_step_per_PDHG-1, nx, n_ctrl])
    alp2_0 = jnp.zeros([time_step_per_PDHG-1, nx, n_ctrl])
    alp0 = (alp1_0, alp2_0)
  else:
    nx, ny = nspatial
    rho0 = jnp.zeros([time_step_per_PDHG-1, nx, ny]) + c_on_rho
    alp1_x_0 = jnp.zeros([time_step_per_PDHG-1, nx, ny, n_ctrl])
    alp2_x_0 = jnp.zeros([time_step_per_PDHG-1, nx, ny, n_ctrl])
    alp1_y_0 = jnp.zeros([time_step_per_PDHG-1, nx, ny, n_ctrl])
    alp2_y_0 = jnp.zeros([time_step_per_PDHG-1, nx, ny, n_ctrl])
    alp0 = (alp1_x_0, alp2_x_0, alp1_y_0, alp2_y_0)

  if load_middle_dir is not None and load_middle_prefix is not None:
    results_out = load_middle_solution(load_middle_dir, load_middle_prefix)
    max_iters, phi_all, rho_all, alp_all, errs_all = results_out[0], results_out[1], results_out[2], results_out[3], results_out[4]
    init_t_ind = len(phi_all)
    assert init_t_ind == len(rho_all) and init_t_ind == len(alp_all) and init_t_ind == len(errs_all)
    if init_t_ind > 0:
      phi0 = phi_all[-1:]
      rho0 = rho_all[-1:]
      alp0 = alp_all[-1:]
  else:
    init_t_ind = 0
    max_iters = 0
    phi_all = []
    rho_all = []
    alp_all = []
    errs_all = []

  print('shape of phi0: ', jnp.shape(phi0), flush = True)
  print('shape of rho0: ', jnp.shape(rho0), flush = True)
  print('shape of alp0: ', jnp.shape(alp0), flush = True)
  
  stepsz_param_min = stepsz_param / 10
  stepsz_param_delta = stepsz_param / 10
  sol_nan = False
  
  tfrecord_ind = 0
  utils.timer.tic("time estimate")  
  for i in range(init_t_ind, nt_PDHG):
    print('=================== nt_PDHG = {}, i = {} ==================='.format(nt_PDHG, i), flush=True)
    t_arr = jnp.linspace(i* dt* (time_step_per_PDHG-1), (i+1)* dt* (time_step_per_PDHG-1), num = time_step_per_PDHG)[1:]  # [time_step_per_PDHG-1]
    if ndim == 1:
      t_arr = t_arr[:,None]  # [time_step_per_PDHG-1, 1]
    else:
      t_arr = t_arr[:,None,None]  # [time_step_per_PDHG-1, 1, 1]

    while True:  # decrease step size if necessary
      results_all, errs = PDHG_solver_oneiter(fn_update_primal, fn_update_dual, fns_dict,
                                    phi0, rho0, alp0, x_arr, t_arr, ndim, dt, dspatial, c_on_rho, 
                                    epsl = epsl, stepsz_param=stepsz_param, fv=fv,
                                    N_maxiter = N_maxiter, print_freq = print_freq, eps = eps, 
                                    tfboard = tfboard, tfrecord_ind = tfrecord_ind)
      if jnp.any(jnp.isnan(errs)):
        if stepsz_param > stepsz_param_min + stepsz_param_delta:  # if nan, decrease step size
          stepsz_param -= stepsz_param_delta
          print('pdhg does not conv at t_ind = {}, decrease step size to {}'.format(i, stepsz_param), flush = True)
        else:  # if still nan, algorithm failed
          print('pdhg does not conv at t_ind = {}, algorithm failed'.format(i), flush = True)
          sol_nan = True
          break
      else:  # if not nan, compute max error and iters, save results, and go to next time block
        pdhg_iters, phi_curr, rho_curr, alp_curr = results_all[-1]
        tfrecord_ind += pdhg_iters
        max_iters = jnp.maximum(max_iters, pdhg_iters)
        # save results
        if i < nt_PDHG-1:  # if not the last time block, exclude the last time step
          phi_all.append(phi_curr[:-1,:])
        else:
          phi_all.append(phi_curr)
        rho_all.append(rho_curr)
        alp_all.append(jnp.stack(alp_curr, axis = 0))  # [4, time_step_per_PDHG-1, nx, ny, dim_ctrl] for 2d and [2, time_step_per_PDHG-1, nx, dim_ctrl] for 1d
        errs_all.append(errs)
        # set initial values for next time block
        g_diff = phi_curr[-1:,...] - phi0[0:1,...]
        print('g_diff err: ', jnp.linalg.norm(g_diff), flush = True)
        phi0 = phi0 + g_diff
        rho0 = rho_curr
        alp0 = alp_curr
        break
    ratio = (i+1) / nt_PDHG
    samples_processed = pdhg_iters
    utils.timer.estimate_time("time estimate", ratio, samples_processed)
      
    if save_middle_dir is not None and save_middle_prefix is not None:
      save(save_middle_dir, save_middle_prefix, [max_iters, phi_all, rho_all, alp_all, errs_all])
    if sol_nan:
      break
  phi_out = jnp.concatenate(phi_all, axis = 0)  # [nt, nx] or [nt, nx, ny]
  rho_out = jnp.concatenate(rho_all, axis = 0)  # [nt-1, nx] or [nt-1, nx, ny]
  alp_out = jnp.concatenate(alp_all, axis = 1)  # [4, nt-1, nx, ny, dim_ctrl] for 2d and [2, nt-1, nx, dim_ctrl] for 1d
  results_out = [(max_iters, phi_out, rho_out, alp_out)]
  print('\n\n')
  print('===========================================')
  if sol_nan:
    print('pdhg does not conv, please decrease stepsize to be less than {}'.format(stepsz_param), flush = True)
  else:
    print('pdhg conv. Max err is {:.2E}. Max iters is {}'.format(jnp.max(jnp.array(errs_all)), max_iters), flush = True)
  return results_out, errs_all
