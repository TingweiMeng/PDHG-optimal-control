import jax
import jax.numpy as jnp
import utils
from einshape import jax_einshape as einshape
from solver import compute_HJ_residual_EO_1d_general, compute_HJ_residual_EO_2d_general
import matplotlib.pyplot as plt


def PDHG_solver_oneiter(fn_update_primal, fn_update_dual, ndim, phi0, rho0, v0, 
                   dt, dspatial, c_on_rho, fns_dict, x_arr, t_arr, epsl = 0.0,
                   N_maxiter = 1000000, print_freq = 1000, eps = 1e-6, stepsz_param=0.9):
  '''
  @ parameters:
    fn_update_primal: function to update primal variable, takes p, d, delta_p, and other parameters, 
                        output p_next, and other parameters
    fn_update_dual: function to update dual variable, takes d, p, delta_d, and other parameters
                        output d_next, and other parameters
    phi0: [nt, nx]
    rho0: [nt-1, nx]
    v0: [vp0, vm0], where vp0 and vm0 are [nt-1, nx] (if using m method, this is [m0,0], where m0 is [nt-1, nx])
    dx, dt, c_on_rho: scalar
    fns_dict: dict of functions, see the function set_up_example_fns in solver.py
    f_in_H: [1, nx] or [nt-1, nx]
    c_in_H: [1, nx] or [nt-1, nx]
    epsl: scalar, diffusion coefficient
    dy: scalar, placeholder for 2d case
    N_maxiter: int, maximum number of iterations
    print_gap: int, gap for printing and saving results
    eps: scalar, stopping criterion
    stepsz_param: scalar, step size parameter
  @ returns:
  '''
  phi_prev = phi0
  rho_prev = rho0
  v_prev = v0

  scale = 1.5
  tau_phi = stepsz_param / scale
  tau_rho = stepsz_param * scale
  
  # fft for preconditioning
  if ndim == 1:
    _,nx = jnp.shape(phi0)
    dx = dspatial[0]
    Lap_vec = jnp.array([-2/(dx*dx), 1/(dx*dx)] + [0.0] * (nx-3) + [1/(dx*dx)])
    fv = jnp.fft.fft(Lap_vec)  # [nx]
  elif ndim == 2:
    _,nx,ny = jnp.shape(phi0)
    dx, dy = dspatial[0], dspatial[1]
    Lap_mat = jnp.array([[-2/(dx*dx)-2/(dy*dy), 1/(dy*dy)] + [0.0] * (ny-3) + [1/(dy*dy)],
                        [1/(dx*dx)] + [0.0] * (ny -1)] + [[0.0]* ny] * (nx-3) + \
                        [[1/(dx*dx)] + [0.0] * (ny-1)])  # [nx, ny]
    fv = jnp.fft.fft2(Lap_mat)  # [nx, ny]
  else:
    raise NotImplementedError

  error_all = []
  results_all = []

  for i in range(N_maxiter):
    phi_next = fn_update_primal(phi_prev, rho_prev, c_on_rho, v_prev, tau_phi, dt, dspatial, fv, epsl)
    # extrapolation
    phi_bar = 2 * phi_next - phi_prev
    rho_next, v_next = fn_update_dual(phi_bar, rho_prev, c_on_rho, v_prev, tau_rho, dt, dspatial, epsl, 
                                      fns_dict, x_arr, t_arr, ndim)

    # primal error
    err1 = jnp.linalg.norm(phi_next - phi_prev) / jnp.maximum(jnp.linalg.norm(phi_prev), 1.0)
    # err2: dual error
    err2 = jnp.linalg.norm(rho_next - rho_prev) / jnp.maximum(jnp.linalg.norm(rho_prev), 1.0)
    for v0, v1 in zip(v_prev, v_next):
      err2 += jnp.linalg.norm(v1 - v0) / jnp.maximum(jnp.linalg.norm(v0), 1.0)
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
      results_all.append((i, v_next, rho_prev, [], phi_prev))
      print('iteration {}, primal error {:.2E}, dual error {:.2E}, eqt error {:.2E}, min rho {:.2f}, max rho {:.2f}'.format(i, 
                  error[0],  error[1],  error[2], jnp.min(rho_next), jnp.max(rho_next)), flush = True)
    rho_prev = rho_next
    phi_prev = phi_next
    v_prev = v_next
  # print the final error
  print('iteration {}, primal error with prev step {:.2E}, dual error with prev step {:.2E}, eqt error {:.2E}'.format(i, error[0],  error[1],  error[2]), flush = True)
  results_all.append((i+1, v_next, rho_next, None, phi_next))
  return results_all, jnp.array(error_all)


def PDHG_multi_step(fn_update_primal, fn_update_dual, fns_dict, x_arr, nt, nspatial, ndim,
                    g, dt, dspatial, c_on_rho, time_step_per_PDHG = 2,
                    N_maxiter = 1000000, print_freq = 1000, eps = 1e-6,
                    epsl = 0.0, stepsz_param=0.9):
  assert (nt-1) % (time_step_per_PDHG-1) == 0  # make sure nt-1 is divisible by time_step_per_PDHG
  nt_PDHG = (nt-1) // (time_step_per_PDHG-1)
  phi0 = einshape("i...->(ki)...", g, k=time_step_per_PDHG)  # repeat each row of g to nt times, [nt, nx] or [nt, nx, ny]
  if ndim == 1:
    nx = nspatial[0]
    rho0 = jnp.zeros([time_step_per_PDHG-1, nx]) + c_on_rho
    vp0 = jnp.zeros([time_step_per_PDHG-1, nx])
    vm0 = jnp.zeros([time_step_per_PDHG-1, nx])
    v0 = (vp0, vm0)
  else:
    nx, ny = nspatial[0], nspatial[1]
    rho0 = jnp.zeros([time_step_per_PDHG-1, nx, ny]) + c_on_rho
    vxp0 = jnp.zeros([time_step_per_PDHG-1, nx, ny])
    vxm0 = jnp.zeros([time_step_per_PDHG-1, nx, ny])
    vyp0 = jnp.zeros([time_step_per_PDHG-1, nx, ny])
    vym0 = jnp.zeros([time_step_per_PDHG-1, nx, ny])
    v0 = (vxp0, vxm0, vyp0, vym0)
    print('shape of phi0: ', jnp.shape(phi0), flush = True)
  
  phi_all = []
  v_all = []
  rho_all = []

  stepsz_param_min = stepsz_param / 10
  stepsz_param_delta = stepsz_param / 10
  sol_nan = False
  max_err = 0
  
  utils.timer.tic('all_time')
  for i in range(nt_PDHG):
    utils.timer.tic('pdhg_iter{}'.format(i))
    print('=================== nt_PDHG = {}, i = {} ==================='.format(nt_PDHG, i), flush=True)
    t_arr = jnp.linspace(i* dt* (time_step_per_PDHG-1), (i+1)* dt* (time_step_per_PDHG-1), num = time_step_per_PDHG)[1:]  # [time_step_per_PDHG-1]
    if ndim == 1:
      t_arr = t_arr[:,None]  # [time_step_per_PDHG-1, 1]
    else:
      t_arr = t_arr[:,None,None]  # [time_step_per_PDHG-1, 1, 1]
    while True:
      results_all, errs = PDHG_solver_oneiter(fn_update_primal, fn_update_dual, ndim, phi0, rho0, v0, 
                                    dt, dspatial, c_on_rho, fns_dict, x_arr, t_arr,
                                    N_maxiter = N_maxiter, print_freq = print_freq, eps = eps,
                                    epsl = epsl, stepsz_param=stepsz_param)
      if jnp.any(jnp.isnan(errs)):
        if stepsz_param > stepsz_param_min + stepsz_param_delta:
          stepsz_param -= stepsz_param_delta
          print('pdhg does not conv at t_ind = {}, decrease step size to {}'.format(i, stepsz_param), flush = True)
        else:
          print('pdhg does not conv at t_ind = {}, algorithm failed'.format(i), flush = True)
          sol_nan = True
          break
      else:
        max_err = jnp.maximum(max_err, errs[-1][-1])
        break
    _, v_curr, rho_curr, _, phi_curr = results_all[-1]
    utils.timer.toc('pdhg_iter{}'.format(i))
    utils.timer.toc('all_time')

    if i < nt_PDHG-1:
      phi_all.append(phi_curr[:-1,:])
    else:
      phi_all.append(phi_curr)
    v_all.append(jnp.stack(v_curr, axis = -1))  # [time_step_per_PDHG-1, nx, ny, 2**ndim] or [time_step_per_PDHG-1, nx, ny, 1]
    rho_all.append(rho_curr)
    g_diff = phi_curr[-1:,...] - phi0[0:1,...]
    print('g_diff err: ', jnp.linalg.norm(g_diff), flush = True)
    phi0 = phi0 + g_diff
    rho0 = rho_curr
    v0 = v_curr
    if sol_nan:
      break
  phi_out = jnp.concatenate(phi_all, axis = 0)
  v_out = jnp.concatenate(v_all, axis = 0)
  rho_out = jnp.concatenate(rho_all, axis = 0)
  results_out = [(0, v_out, rho_out, None, phi_out)]
  print('\n\n')
  print('===========================================')
  utils.timer.toc('all_time')
  if sol_nan:
    print('pdhg does not conv, please decrease stepsize to be less than {}'.format(stepsz_param), flush = True)
  else:
    print('pdhg conv. Max err is {:.2E}'.format(max_err), flush = True)

  # plot solution 
  T = dt * (nt-1)
  phi_trans = einshape('ij->ji', phi_out)  # [nx, nt]
  phi_trans = jnp.concatenate([phi_trans, phi_trans[:1,:]], axis = 0)  # [nx+1, nt]
  dim1, dim2 = phi_trans.shape
  x_arr = jnp.linspace(0.0, 2.0, num = nx + 1, endpoint = True)
  t_arr = jnp.linspace(0.0, T, num = nt, endpoint = True)
  t_mesh, x_mesh = jnp.meshgrid(t_arr, x_arr)
  fig = plt.figure()
  plt.contourf(x_mesh[:dim1, :dim2], t_mesh[:dim1, :dim2], phi_trans)
  plt.colorbar()
  plt.xlabel('x')
  plt.ylabel('t')
  plt.savefig('./fig_solution.png')
  plt.close()

  rho_trans = einshape('ij->ji', rho_out)  # [nx, nt-1]
  rho_trans = jnp.concatenate([rho_trans, rho_trans[:1,:]], axis = 0)  # [nx+1, nt-1]
  dim1, dim2 = rho_trans.shape
  fig = plt.figure()
  plt.contourf(x_mesh[:dim1, -dim2:], t_mesh[:dim1, -dim2:], rho_trans)
  plt.colorbar()
  plt.xlabel('x')
  plt.ylabel('t')
  plt.savefig('./fig_rho.png')
  plt.close()

  print('v_out shape: ', jnp.shape(v_out), flush = True)
  v0_trans = einshape('ij->ji', v_out[...,0])  # [nx, nt-1]
  v0_trans = jnp.concatenate([v0_trans, v0_trans[:1,:]], axis = 0)  # [nx+1, nt-1]
  dim1, dim2 = v0_trans.shape
  fig = plt.figure()
  plt.contourf(x_mesh[:dim1, -dim2:], t_mesh[:dim1, -dim2:], v0_trans)
  plt.colorbar()
  plt.xlabel('x')
  plt.ylabel('t')
  plt.savefig('./fig_v0.png')
  plt.close()

  v1_trans = einshape('ij->ji', v_out[...,1])  # [nx, nt-1]
  v1_trans = jnp.concatenate([v1_trans, v1_trans[:1,:]], axis = 0)  # [nx+1, nt-1]
  dim1, dim2 = v1_trans.shape
  fig = plt.figure()
  plt.contourf(x_mesh[:dim1, -dim2:], t_mesh[:dim1, -dim2:], v1_trans)
  plt.colorbar()
  plt.xlabel('x')
  plt.ylabel('t')
  plt.savefig('./fig_v1.png')
  plt.close()
  return results_out, None
