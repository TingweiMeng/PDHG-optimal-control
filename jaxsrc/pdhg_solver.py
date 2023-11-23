import jax
import jax.numpy as jnp
import utils
from einshape import jax_einshape as einshape
from solver import compute_HJ_residual_EO_1d_general, compute_HJ_residual_EO_2d_general, compute_Dxx_fft_fv
import matplotlib.pyplot as plt


def PDHG_solver_oneiter(fn_update_primal, fn_update_dual, ndim, phi0, rho0, alp0, 
                   dt, dspatial, c_on_rho, fns_dict, x_arr, t_arr, epsl = 0.0,
                   N_maxiter = 1000000, print_freq = 1000, eps = 1e-6, stepsz_param=0.9):
  '''
  @ parameters:  #TODO: add a function for evaluating the error, TODO: rearrage the parameters in functions
    fn_update_primal: function to update primal variable, takes p, d, delta_p, and other parameters, 
                        output p_next, and other parameters
    fn_update_dual: function to update dual variable, takes d, p, delta_d, and other parameters
                        output d_next, and other parameters
    ndim: int, 1 or 2
    phi0: [nt, nx] for 1d, [nt, nx, ny] for 2d
    rho0: [nt-1, nx] for 1d, [nt-1, nx, ny] for 2d
    alp0: (alp1, alp2), where alp1 and alp2 are [nt-1, nx, dim_ctrl] for 1d
          (alp1_x, alp2_x, alp1_y, alp2_y), where alp1_x, alp2_x, alp1_y, alp2_y are [nt-1, nx, ny, dim_ctrl] for 2d
    dt, c_on_rho: scalar
    dspatial: (dx, dy) for 2d, (dx,) for 1d
    fns_dict: dict of functions, see the function set_up_example_fns in solver.py
    x_arr: array that can be broadcasted to [1, nx, 1] for 1d, [1, nx, ny, 2] for 2d
    t_arr: array that can be broadcasted to [nt-1, 1] for 1d, [nt-1, 1, 1] for 2d
    epsl: scalar, diffusion coefficient [TODO: not used yet]
    N_maxiter: int, maximum number of iterations
    print_gap: int, gap for printing and saving results
    eps: scalar, stopping criterion
    stepsz_param: scalar, step size parameter
  @ returns:  #TODO: remove None in results_all
    results_all: list of tuples, each tuple is (iter, alp_next, rho_next, None, phi_next)
    error_all: [N_maxiter, 3], primal error, dual error, equation error
  '''
  phi_prev = phi0
  rho_prev = rho0
  alp_prev = alp0

  scale = 1.5
  tau_phi = stepsz_param / scale
  tau_rho = stepsz_param * scale
  
  nspatial = jnp.shape(phi0)[1:]  # [nx] or [nx, ny]
  fv = compute_Dxx_fft_fv(ndim, nspatial, dspatial)

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
    if ndim == 1:
      HJ_residual = compute_HJ_residual_EO_1d_general(phi_next, dt, dspatial, fns_dict, epsl, x_arr, t_arr)
    elif ndim == 2:
      HJ_residual = compute_HJ_residual_EO_2d_general(phi_next, dt, dspatial, fns_dict, epsl, x_arr, t_arr)
    err3 = jnp.mean(jnp.abs(HJ_residual))
    
    error = jnp.array([err1, err2, err3])
    error_all.append(error)
    if error[2] < eps:
      print('PDHG converges at iter {}'.format(i), flush=True)
      break
    if jnp.isnan(error[0]) or jnp.isnan(error[1]):
      print("Nan error at iter {}".format(i))
      break
    if print_freq > 0 and i % print_freq == 0:
      results_all.append((i, alp_next, rho_prev, [], phi_prev))
      print('iteration {}, primal error {:.2E}, dual error {:.2E}, eqt error {:.2E}, min rho {:.2f}, max rho {:.2f}'.format(i, 
                  error[0],  error[1],  error[2], jnp.min(rho_next), jnp.max(rho_next)), flush = True)
    rho_prev = rho_next
    phi_prev = phi_next
    alp_prev = alp_next
  # print the final error
  print('iteration {}, primal error with prev step {:.2E}, dual error with prev step {:.2E}, eqt error {:.2E}'.format(i, error[0],  error[1],  error[2]), flush = True)
  results_all.append((i+1, alp_next, rho_next, None, phi_next))
  return results_all, jnp.array(error_all)


def PDHG_multi_step(fn_update_primal, fn_update_dual, fns_dict, x_arr, nt, nspatial, ndim,
                    g, dt, dspatial, c_on_rho, time_step_per_PDHG = 2,
                    N_maxiter = 1000000, print_freq = 1000, eps = 1e-6,
                    epsl = 0.0, stepsz_param=0.9, n_ctrl = None):
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
      results_all, errs = PDHG_solver_oneiter(fn_update_primal, fn_update_dual, ndim, phi0, rho0, alp0, 
                                    dt, dspatial, c_on_rho, fns_dict, x_arr, t_arr,
                                    N_maxiter = N_maxiter, print_freq = print_freq, eps = eps,
                                    epsl = epsl, stepsz_param=stepsz_param)
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
        pdhg_iters, alp_curr, rho_curr, _, phi_curr = results_all[-1]
        max_iters = jnp.maximum(max_iters, pdhg_iters)
        # save results
        if i < nt_PDHG-1:  # if not the last time block, exclude the last time step
          phi_all.append(phi_curr[:-1,:])
        else:
          phi_all.append(phi_curr)
        alp_all.append(jnp.stack(alp_curr, axis = -1))  # [time_step_per_PDHG-1, nx, ny, dim_ctrl, 4] for 2d and [time_step_per_PDHG-1, nx, dim_ctrl, 2] for 1d
        rho_all.append(rho_curr)
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
  results_out = [(max_iters, alp_out, rho_out, None, phi_out)]
  print('\n\n')
  print('===========================================')
  utils.timer.toc('all_time')
  if sol_nan:
    print('pdhg does not conv, please decrease stepsize to be less than {}'.format(stepsz_param), flush = True)
  else:
    print('pdhg conv. Max err is {:.2E}. Max iters is {}'.format(max_err, max_iters), flush = True)

  # plot solution 
  # T = dt * (nt-1)
  # phi_trans = einshape('ij->ji', phi_out)  # [nx, nt]
  # phi_trans = jnp.concatenate([phi_trans, phi_trans[:1,:]], axis = 0)  # [nx+1, nt]
  # dim1, dim2 = phi_trans.shape
  # x_arr = jnp.linspace(0.0, 2.0, num = nx + 1, endpoint = True)
  # t_arr = jnp.linspace(0.0, T, num = nt, endpoint = True)
  # t_mesh, x_mesh = jnp.meshgrid(t_arr, x_arr)
  # fig = plt.figure()
  # plt.contourf(x_mesh[:dim1, :dim2], t_mesh[:dim1, :dim2], phi_trans)
  # plt.colorbar()
  # plt.xlabel('x')
  # plt.ylabel('t')
  # plt.savefig('./fig_solution.png')
  # plt.close()

  # rho_trans = einshape('ij->ji', rho_out)  # [nx, nt-1]
  # rho_trans = jnp.concatenate([rho_trans, rho_trans[:1,:]], axis = 0)  # [nx+1, nt-1]
  # dim1, dim2 = rho_trans.shape
  # fig = plt.figure()
  # plt.contourf(x_mesh[:dim1, -dim2:], t_mesh[:dim1, -dim2:], rho_trans)
  # plt.colorbar()
  # plt.xlabel('x')
  # plt.ylabel('t')
  # plt.savefig('./fig_rho.png')
  # plt.close()

  # print('v_out shape: ', jnp.shape(v_out), flush = True)
  # v0_trans = einshape('ij->ji', v_out[...,0])  # [nx, nt-1]
  # v0_trans = jnp.concatenate([v0_trans, v0_trans[:1,:]], axis = 0)  # [nx+1, nt-1]
  # dim1, dim2 = v0_trans.shape
  # fig = plt.figure()
  # plt.contourf(x_mesh[:dim1, -dim2:], t_mesh[:dim1, -dim2:], v0_trans)
  # plt.colorbar()
  # plt.xlabel('x')
  # plt.ylabel('t')
  # plt.savefig('./fig_v0.png')
  # plt.close()

  # v1_trans = einshape('ij->ji', v_out[...,1])  # [nx, nt-1]
  # v1_trans = jnp.concatenate([v1_trans, v1_trans[:1,:]], axis = 0)  # [nx+1, nt-1]
  # dim1, dim2 = v1_trans.shape
  # fig = plt.figure()
  # plt.contourf(x_mesh[:dim1, -dim2:], t_mesh[:dim1, -dim2:], v1_trans)
  # plt.colorbar()
  # plt.xlabel('x')
  # plt.ylabel('t')
  # plt.savefig('./fig_v1.png')
  # plt.close()
  return results_out, None
