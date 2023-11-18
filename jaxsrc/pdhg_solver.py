import jax
import jax.numpy as jnp
import utils
from einshape import jax_einshape as einshape
from solver import compute_HJ_residual_EO_1d_general, compute_HJ_residual_EO_2d_general
from functools import partial
import matplotlib.pyplot as plt


def PDHG_solver_oneiter(fn_update_primal, fn_update_dual, ndim, phi0, rho0, v0, 
                   dt, dspatial, c_on_rho, fns_dict, x_arr, t_arr, fwd, epsl = 0.0,
                   N_maxiter = 1000000, print_freq = 1000, eps = 1e-6, sigma_hj=0.9, sigma_cont=0.9,
                   precond_hj = False, precond_cont = False, old_ver=False):
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
    fwd: whether use implicit or explicit scheme for HJ
  @ returns:
  '''
  if old_ver:
    from utils_pdhg_old import HJ_residual as HJ_residual_fn
    from utils_pdhg_old import cont_residual as cont_residual_fn
  else:
    from utils_pdhg import HJ_residual as HJ_residual_fn
    from utils_pdhg import cont_residual as cont_residual_fn

  phi_prev = phi0
  phi_bar = phi0
  rho_prev = rho0
  v_prev = v0

  tau_phi = sigma_hj
  tau_rho = sigma_cont
  print('tau_phi: ', tau_phi, 'tau_rho: ', tau_rho)
  
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
    rho_next, v_next = fn_update_dual(phi_bar, rho_prev, c_on_rho, v_prev, tau_rho, dt, dspatial, epsl, 
                                      fns_dict, x_arr, t_arr, ndim, fwd=fwd, precond=precond_cont, fv=fv)
    phi_next = fn_update_primal(phi_prev, rho_next, c_on_rho, v_next, tau_phi, dt, dspatial, fv, epsl, fwd=fwd, precond=precond_hj,
                                fns_dict=fns_dict, x_arr=x_arr, t_arr=t_arr)
    # extrapolation
    phi_bar = 2 * phi_next - phi_prev

    # print('rho_next: ', rho_next)
    # print('phi_next: ', phi_next)
    # print('v_next: ', v_next)

    # primal error
    err1 = jnp.linalg.norm(phi_next - phi_prev) / jnp.maximum(jnp.linalg.norm(phi_prev), 1.0)
    # err2: dual error
    err2 = jnp.linalg.norm(rho_next - rho_prev) / jnp.maximum(jnp.linalg.norm(rho_prev), 1.0)
    for v0, v1 in zip(v_prev, v_next):
      err2 += jnp.linalg.norm(v1 - v0) / jnp.maximum(jnp.linalg.norm(v0), 1.0)
    # err3: equation error
    if ndim == 1:
      HJ_residual = compute_HJ_residual_EO_1d_general(phi_next, dt, dspatial, fns_dict, epsl, x_arr, t_arr, fwd = fwd)
    elif ndim == 2:
      HJ_residual = compute_HJ_residual_EO_2d_general(phi_next, dt, dspatial, fns_dict, epsl, x_arr, t_arr, fwd = fwd)
    err3 = jnp.mean(jnp.abs(HJ_residual))
    
    error = jnp.array([err1, err2, err3])
    error_all.append(error)
    HJ_residual_pdhg = HJ_residual_fn(phi_next, v_next, dspatial, dt, epsl, fns_dict, x_arr, t_arr, fwd)
    cont_residual_pdhg = cont_residual_fn(rho_next, v_next, dspatial, dt, epsl, fns_dict, x_arr, t_arr, fwd, c_on_rho)
    stopping_criteria_hj = jnp.linalg.norm(HJ_residual_pdhg)
    stopping_criteria_cont = jnp.linalg.norm(cont_residual_pdhg)

    if i % 100 == 0:
      print('iteration {}, primal error {:.2E}, dual error {:.2E}, eqt error {:.2E}'.format(i,err1, err2, err3), flush = True)
      # print('rho_next: ', rho_next)
      # print('phi_next: ', phi_next)
      # print('v_next: ', v_next)
      print('max rho {:.2f}, min rho {:.2f}'.format(jnp.max(rho_next), jnp.min(rho_next)))
      print('err phi: ', jnp.linalg.norm(phi_next - phi_prev))
      print('err rho: ', jnp.linalg.norm(rho_next - rho_prev))
      print('err v: ', jnp.linalg.norm(v_next - v_prev))
      print('stopping_criteria_hj: ', stopping_criteria_hj)
      print('stopping_criteria_cont: ', stopping_criteria_cont)
      print('eps: ', eps)
    
    if stopping_criteria_hj < eps and stopping_criteria_cont < eps:
      print('PDHG converges at iter {}'.format(i), flush=True)
      break
    if jnp.isnan(error[0]) or jnp.isnan(error[1]):
      print("Nan error at iter {}".format(i))
      break
    if print_freq > 0 and i % print_freq == 0:
      results_all.append((i, v_next, rho_prev, phi_prev))
      print('iteration {}, primal error {:.2E}, dual error {:.2E}, eqt error {:.2E}, min rho {:.2f}, max rho {:.2f}'.format(i, 
                  error[0],  error[1],  error[2], jnp.min(rho_next), jnp.max(rho_next)), flush = True)
    rho_prev = rho_next
    phi_prev = phi_next
    v_prev = v_next
  # print the final error
  print('iteration {}, primal error with prev step {:.2E}, dual error with prev step {:.2E}, eqt error {:.2E}'.format(i, error[0],  error[1],  error[2]), flush = True)
  results_all.append((i+1, v_next, rho_next, phi_next))
  return results_all, jnp.array(error_all)


def PDHG_multi_step_inverse(fn_update_primal, fn_update_dual, fns_dict, x_arr, nt, nspatial, ndim,
                    g, dt, dspatial, c_on_rho, time_step_per_PDHG = 2,
                    N_maxiter = 1000000, print_freq = 1000, eps = 1e-6,
                    epsl = 0.0, fwd=False, sigma_hj=0.9, sigma_cont=0.9,
                    precond_hj=False, precond_cont=False,
                    old_ver=False):
  if old_ver:
    from utils_pdhg_old import HJ_residual as HJ_residual_fn
    from utils_pdhg_old import cont_residual as cont_residual_fn
  else:
    from utils_pdhg import HJ_residual as HJ_residual_fn
    from utils_pdhg import cont_residual as cont_residual_fn

  assert (nt-1) % (time_step_per_PDHG-1) == 0  # make sure nt-1 is divisible by time_step_per_PDHG
  nt_PDHG = (nt-1) // (time_step_per_PDHG-1)
  phi0 = einshape("i...->(ki)...", g, k=time_step_per_PDHG)  # repeat each row of g to nt times, [nt, nx] or [nt, nx, ny]
  # NOTE: use true solution here TODO: change this !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  # roll g to left by 1 and concate with g
  # phi0 = jnp.concatenate([jnp.roll(phi0[0:1,...], -1, axis=1), phi0[1:,...]], axis = 0)  # [2*nt-1, nx] or [2*nt-1, nx, ny]
  if ndim == 1:
    nx = nspatial[0]
    rho0 = jnp.zeros([time_step_per_PDHG-1, nx]) + c_on_rho
    v0 = jnp.zeros([time_step_per_PDHG-1, nx])
  else:
    raise NotImplementedError
  
  phi_all = []
  v_all = []
  rho_all = []

  sigma_hj_min = sigma_hj / 10
  sigma_hj_delta = sigma_hj / 10
  sigma_cont_min = sigma_cont / 10
  sigma_cont_delta = sigma_cont / 10
  sol_nan = False
  max_err = 0
  
  T = dt * (nt-1)

  utils.timer.tic('all_time')
  num_iters = 0
  for i in range(nt_PDHG):
    utils.timer.tic('pdhg_iter{}'.format(i))
    print('=================== nt_PDHG = {}, i = {} ==================='.format(nt_PDHG, i), flush=True)
    t_arr = jnp.linspace(T - (i+1)* dt* (time_step_per_PDHG-1), T - i* dt* (time_step_per_PDHG-1), num = time_step_per_PDHG)[1:]  # [time_step_per_PDHG-1]
    if ndim == 1:
      t_arr = t_arr[:,None]  # [time_step_per_PDHG-1, 1]
    else:
      t_arr = t_arr[:,None,None]  # [time_step_per_PDHG-1, 1, 1]
    while True:
      results_all, errs = PDHG_solver_oneiter(fn_update_primal, fn_update_dual, ndim, phi0, rho0, v0, 
                                    dt, dspatial, c_on_rho, fns_dict, x_arr, t_arr,
                                    N_maxiter = N_maxiter, print_freq = print_freq, eps = eps,
                                    epsl = epsl, fwd = fwd, sigma_hj=sigma_hj, sigma_cont=sigma_cont,
                                    precond_hj = precond_hj, precond_cont = precond_cont, 
                                    old_ver=old_ver)
      if jnp.any(jnp.isnan(errs)):
        if sigma_hj > sigma_hj_min + sigma_hj_delta and sigma_cont > sigma_cont_min + sigma_cont_delta:
          sigma_hj -= sigma_hj_delta
          sigma_cont -= sigma_cont_delta
          print('pdhg does not conv at t_ind = {}, decrease sigma_hj to {}, sigma_cont to {}'.format(i, sigma_hj, sigma_cont), flush = True)
        else:
          print('pdhg does not conv at t_ind = {}, algorithm failed'.format(i), flush = True)
          sol_nan = True
          break
      else:
        max_err = jnp.maximum(max_err, errs[-1][-1])
        break
    num_iter_curr, v_curr, rho_curr, phi_curr = results_all[-1]
    utils.timer.toc('pdhg_iter{}'.format(i))
    utils.timer.toc('all_time')
    # max of num_iter_curr
    num_iters = jnp.maximum(num_iters, num_iter_curr)

    if i > 0:
      phi_all.append(phi_curr[:-1,:])
    else:
      phi_all.append(phi_curr)
    # v_all.append(jnp.stack(v_curr, axis = -1))  # [time_step_per_PDHG-1, nx, ny, 2**ndim] or [time_step_per_PDHG-1, nx, ny, 1]
    v_all.append(v_curr)
    rho_all.append(rho_curr)
    # g_diff = phi_curr[:1,...] - phi0[-1:,...]  # make sure the initial phi gives the terminal cond of next step
    # print('g_diff err: ', jnp.linalg.norm(g_diff), flush = True)
    # phi0 = phi0 + g_diff
    # rho0 = rho_curr
    # v0 = v_curr
    phi0 = einshape("i...->(ki)...", g, k=time_step_per_PDHG)  # repeat each row of g to nt times, [nt, nx] or [nt, nx, ny]
    rho0 = jnp.zeros([time_step_per_PDHG-1, nx]) + c_on_rho
    v0 = jnp.zeros([time_step_per_PDHG-1, nx])
    if sol_nan:
      break
  phi_all.reverse()
  v_all.reverse()
  rho_all.reverse()

  phi_out = jnp.concatenate(phi_all, axis = 0)
  v_out = jnp.concatenate(v_all, axis = 0)
  rho_out = jnp.concatenate(rho_all, axis = 0)
  results_out = [(0, v_out, rho_out, phi_out)]
  print('\n\n')
  print('===========================================')
  utils.timer.toc('all_time')
  if sol_nan:
    print('pdhg does not conv, please decrease sigma_hj and sigma_cont to be less than {:.2E} and {:.2E}'.format(sigma_hj_min, sigma_cont_min), flush = True)
  else:
    print('pdhg conv. Max err is {:.2E}, max num_iters is {}'.format(max_err, num_iters), flush = True)

  # plot solution 
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

  alp_trans = einshape('ij->ji', v_out)  # [nx, nt-1]
  alp_trans = jnp.concatenate([alp_trans, alp_trans[:1,:]], axis = 0)  # [nx+1, nt-1]
  dim1, dim2 = alp_trans.shape
  fig = plt.figure()
  plt.contourf(x_mesh[:dim1, -dim2:], t_mesh[:dim1, -dim2:], alp_trans)
  plt.colorbar()
  plt.xlabel('x')
  plt.ylabel('t')
  plt.savefig('./fig_alp.png')
  plt.close()

  HJ_residual = HJ_residual_fn(phi_out, v_out, dspatial, dt, epsl, fns_dict, x_arr, t_arr, fwd)
  HJ_residual_trans = einshape('ij->ji', HJ_residual)
  HJ_residual_trans = jnp.concatenate([HJ_residual_trans, HJ_residual_trans[:1,:]], axis = 0)  # [nx+1, nt]
  dim1, dim2 = HJ_residual_trans.shape
  fig = plt.figure()
  plt.contourf(x_mesh[:dim1, -dim2:], t_mesh[:dim1, -dim2:], HJ_residual_trans)
  plt.colorbar()
  plt.xlabel('x')
  plt.ylabel('t')
  plt.savefig('./fig_HJ_error.png')
  plt.close()

  cont_residual = cont_residual_fn(rho_out, v_out, dspatial, dt, epsl, fns_dict, x_arr, t_arr, fwd, c_on_rho)
  cont_residual_trans = einshape('ij->ji', cont_residual)
  cont_residual_trans = jnp.concatenate([cont_residual_trans, cont_residual_trans[:1,:]], axis = 0)  # [nx+1, nt]
  dim1, dim2 = cont_residual_trans.shape
  print('cont_residual_trans shape, ', cont_residual_trans.shape)
  plt.figure()
  plt.contourf(x_mesh[:dim1, -dim2:], t_mesh[:dim1, -dim2:], cont_residual_trans)
  plt.colorbar()
  plt.xlabel('x')
  plt.ylabel('t')
  plt.savefig('./fig_cont_error.png')
  plt.close()

  return results_out
