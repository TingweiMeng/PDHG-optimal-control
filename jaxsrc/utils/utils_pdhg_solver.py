import jax.numpy as jnp
import utils.utils as utils
from einshape import jax_einshape as einshape
import matplotlib.pyplot as plt


def PDHG_solver_oneiter(fn_update_primal, fn_update_dual, fn_compute_err, fns_dict, phi0, rho0, alp0, x_arr, t_arr, 
                   ndim, dt, dspatial, c_on_rho, epsl = 0.0, stepsz_param=0.9, fv=None,
                   N_maxiter = 1000000, print_freq = 1000, eps = 1e-6):
  '''
  @ parameters:
    fn_update_primal: function to update primal variable, takes p, d, delta_p, and other parameters, 
                        output p_next, and other parameters
    fn_update_dual: function to update dual variable, takes d, p, delta_d, and other parameters
                        output d_next, and other parameters
    fn_compute_err: function to compute error, take phi and other parameters
    fns_dict: dict of functions, see the function set_up_example_fns in solver.py
    phi0: [nt, nx] for 1d, [nt, nx, ny] for 2d
    rho0: [nt-1, nx] for 1d, [nt-1, nx, ny] for 2d
    alp0: (alp1, alp2), where alp1 and alp2 are [nt-1, nx, dim_ctrl] for 1d
          (alp1_x, alp2_x, alp1_y, alp2_y), where alp1_x, alp2_x, alp1_y, alp2_y are [nt-1, nx, ny, dim_ctrl] for 2d
    ndim: int, 1 or 2
    dt, c_on_rho: scalar
    dspatial: (dx, dy) for 2d, (dx,) for 1d
    x_arr: array that can be broadcasted to [1, nx, 1] for 1d, [1, nx, ny, 2] for 2d
    t_arr: array that can be broadcasted to [nt-1, 1] for 1d, [nt-1, 1, 1] for 2d
    epsl: scalar, diffusion coefficient
    N_maxiter: int, maximum number of iterations
    print_gap: int, gap for printing and saving results
    eps: scalar, stopping criterion
    stepsz_param: scalar, step size parameter
    fv: [nx] for 1d, [nx, ny] for 2d, FFT of Laplacian operator, or None if do not use preconditioning
  @ returns:
    results_all: list of tuples, each tuple is (iter, phi_next, rho_next, alp_next)
    error_all: [N_maxiter//print_freq, 3], primal error, dual error, equation error
  '''
  phi_prev = phi0
  rho_prev = rho0
  alp_prev = alp0

  scale = 1.5
  tau_phi = stepsz_param / scale
  tau_rho = stepsz_param * scale
  
  error_all = []
  results_all = []

  for i in range(N_maxiter):
    phi_next = fn_update_primal(phi_prev, rho_prev, c_on_rho, alp_prev, tau_phi, dt, dspatial, fns_dict, fv, epsl, x_arr, t_arr)
    # extrapolation
    phi_bar = 2 * phi_next - phi_prev
    rho_next, alp_next = fn_update_dual(phi_bar, rho_prev, c_on_rho, alp_prev, tau_rho, dt, dspatial, epsl, 
                                      fns_dict, x_arr, t_arr, ndim)

    # primal error
    err1 = jnp.linalg.norm(phi_next - phi_prev) / jnp.maximum(jnp.linalg.norm(phi_prev), 1.0)
    # err2: dual error
    err2 = jnp.linalg.norm(rho_next - rho_prev) / jnp.maximum(jnp.linalg.norm(rho_prev), 1.0)
    for alp_p, alp_n in zip(alp_prev, alp_next):
      err2 += jnp.linalg.norm(alp_p - alp_n) / jnp.maximum(jnp.linalg.norm(alp_p), 1.0)
    # err3: equation error
    err3 = fn_compute_err(phi_next, dt, dspatial, fns_dict, epsl, x_arr, t_arr)
    
    error = jnp.array([err1, err2, err3])
    if error[2] < eps:
      print('PDHG converges at iter {}'.format(i), flush=True)
      break
    if jnp.isnan(error[0]) or jnp.isnan(error[1]):
      print("Nan error at iter {}".format(i))
      break
    if print_freq > 0 and i % print_freq == 0:
      results_all.append((i, phi_prev, rho_prev, alp_next))
      error_all.append(error)
      print('iteration {}, primal error {:.2E}, dual error {:.2E}, eqt error {:.2E}, min rho {:.2f}, max rho {:.2f}'.format(i, 
                  error[0],  error[1],  error[2], jnp.min(rho_next), jnp.max(rho_next)), flush = True)
    phi_prev = phi_next
    rho_prev = rho_next
    alp_prev = alp_next
  # print the final error
  print('iteration {}, primal error with prev step {:.2E}, dual error with prev step {:.2E}, eqt error {:.2E}'.format(i, error[0],  error[1],  error[2]), flush = True)
  results_all.append((i+1, phi_next, rho_next, alp_next))
  error_all.append(error)
  error_all = jnp.array(error_all)
  # plot pdhg errors for debugging [TODO: remove this]
  plt.figure()
  # create subfigs
  plt.subplot(1, 3, 1)
  plt.plot(error_all[:,0])
  plt.title('primal error')
  plt.subplot(1, 3, 2)
  plt.plot(error_all[:,1])
  plt.title('dual error')
  plt.subplot(1, 3, 3)
  plt.plot(error_all[:,2])
  plt.title('equation error')
  plt.savefig('dim{}_pdhg_errors.png'.format(ndim))
  plt.close()
  return results_all, error_all


def PDHG_multi_step(fn_update_primal, fn_update_dual, fn_compute_err, fns_dict, g, x_arr, 
                    ndim, nt, nspatial, dt, dspatial, c_on_rho, time_step_per_PDHG = 2,
                    epsl = 0.0, stepsz_param=0.9, n_ctrl = None, fv=None,
                    N_maxiter = 1000000, print_freq = 1000, eps = 1e-6):
  '''
  @ parameters:
    fn_update_primal, fn_update_dual, fn_compute_err, fns_dict, x_arr: see PDHG_solver_oneiter
    g: [1, nx] for 1d, [1, nx, ny] for 2d
    ndim, nt, nspatial, c_on_rho, epsl, stepsz_param, fv: see PDHG_solver_oneiter
    dt: scalar, time step size
    dspatial: (dx,) for 1d, (dx, dy) for 2d
    time_step_per_PDHG: int, number of time steps per time block (run PDHG_solver_oneiter in each block)
    n_ctrl: int, number of control variables
    N_maxiter, print_freq, eps: see PDHG_solver_oneiter
  @ returns:
    results_out: list of tuples, each tuple is (iter, phi_next, rho_next, alp_next)
    error_all: [N_maxiter//print_freq, 3], primal error, dual error, equation error
  '''
  if n_ctrl is None:
    n_ctrl = ndim
  assert (nt-1) % (time_step_per_PDHG-1) == 0  # make sure nt-1 is divisible by time_step_per_PDHG
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
  print('shape of phi0: ', jnp.shape(phi0), flush = True)
  print('shape of rho0: ', jnp.shape(rho0), flush = True)
  print('shape of alp0: ', jnp.shape(alp0), flush = True)

  phi_all = []
  rho_all = []
  alp_all = []
  errs_all = []
  
  stepsz_param_min = stepsz_param / 10
  stepsz_param_delta = stepsz_param / 10
  sol_nan = False
  max_err, max_iters = 0.0, 0
  
  utils.timer.tic('all_time')
  for i in range(nt_PDHG):
    utils.timer.tic('time_block_{}'.format(i))
    print('=================== nt_PDHG = {}, i = {} ==================='.format(nt_PDHG, i), flush=True)
    t_arr = jnp.linspace(i* dt* (time_step_per_PDHG-1), (i+1)* dt* (time_step_per_PDHG-1), num = time_step_per_PDHG)[1:]  # [time_step_per_PDHG-1]
    if ndim == 1:
      t_arr = t_arr[:,None]  # [time_step_per_PDHG-1, 1]
    else:
      t_arr = t_arr[:,None,None]  # [time_step_per_PDHG-1, 1, 1]

    while True:  # decrease step size if necessary
      results_all, errs = PDHG_solver_oneiter(fn_update_primal, fn_update_dual, fn_compute_err, fns_dict,
                                    phi0, rho0, alp0, x_arr, t_arr, ndim, dt, dspatial, c_on_rho, 
                                    epsl = epsl, stepsz_param=stepsz_param, fv=fv,
                                    N_maxiter = N_maxiter, print_freq = print_freq, eps = eps)
      if jnp.any(jnp.isnan(errs)):
        if stepsz_param > stepsz_param_min + stepsz_param_delta:  # if nan, decrease step size
          stepsz_param -= stepsz_param_delta
          print('pdhg does not conv at t_ind = {}, decrease step size to {}'.format(i, stepsz_param), flush = True)
        else:  # if still nan, algorithm failed
          print('pdhg does not conv at t_ind = {}, algorithm failed'.format(i), flush = True)
          sol_nan = True
          break
      else:  # if not nan, compute max error and iters, save results, and go to next time block
        max_err = jnp.maximum(max_err, errs[-1][-1])
        pdhg_iters, phi_curr, rho_curr, alp_curr = results_all[-1]
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
    utils.timer.toc('time_block_{}'.format(i))
    utils.timer.toc('all_time')
    if sol_nan:
      break
  phi_out = jnp.concatenate(phi_all, axis = 0)  # [nt, nx] or [nt, nx, ny]
  rho_out = jnp.concatenate(rho_all, axis = 0)  # [nt-1, nx] or [nt-1, nx, ny]
  alp_out = jnp.concatenate(alp_all, axis = 0)  # [nt-1, nx, ny, dim_ctrl, 4] for 2d and [nt-1, nx, dim_ctrl, 2] for 1d
  results_out = [(max_iters, phi_out, rho_out, alp_out)]
  print('\n\n')
  print('===========================================')
  utils.timer.toc('all_time')
  if sol_nan:
    print('pdhg does not conv, please decrease stepsize to be less than {}'.format(stepsz_param), flush = True)
  else:
    print('pdhg conv. Max err is {:.2E}. Max iters is {}'.format(max_err, max_iters), flush = True)
  return results_out, errs_all
