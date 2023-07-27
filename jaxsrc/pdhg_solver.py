import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import utils
from einshape import jax_einshape as einshape
import os
import solver
from solver import interpolation_x, interpolation_t
import pickle
from save_analysis import compute_HJ_residual_EO_1d_general, compute_HJ_residual_EO_2d_general
import matplotlib.pyplot as plt

@jax.jit
def Dx_right_decreasedim(phi, dx):
  '''F phi = (phi_{k+1,i+1}-phi_{k+1,i})/dx
  phi_{k+1,i+1} is periodic in i+1. Can be also used for 2d spatial domain
  @ parameters:
    phi: [nt, nx] or [nt, nx, ny]
  @ return
    out: [nt-1, nx] or [nt-1, nx, ny]
  '''
  phi_ip1 = jnp.roll(phi, -1, axis=1)
  out = phi_ip1 - phi
  out = out[1:,...]/dx
  return out

@jax.jit
def Dx_right_increasedim(m, dx):
  '''F m = (-m[k-1,i] + m[k-1,i+1])/dx
  m[k,i+1] is periodic in i+1
  prepend 0 in axis-0
  @ parameters:
    m: [nt-1, nx] or [nt-1, nx, ny]
  @ return
    out: [nt, nx] or [nt, nx, ny]
  '''
  m_ip1 = jnp.roll(m, -1, axis=1)
  out = -m + m_ip1
  out = out/dx
  out = jnp.concatenate([jnp.zeros_like(out[0:1,...]), out], axis = 0) #prepend 0
  return out

@jax.jit
def Dx_left_decreasedim(phi, dx):
  '''F phi = (phi_{k+1,i}-phi_{k+1,i-1})/dx
  phi_{k+1,i-1} is periodic in i+1
  @ parameters:
    phi: [nt, nx] or [nt, nx, ny]
  @ return
    out: [nt-1, nx] or [nt-1, nx, ny]
  '''
  phi_im1 = jnp.roll(phi, 1, axis=1)
  out = phi - phi_im1
  out = out[1:,...]/dx
  return out

@jax.jit
def Dx_left_increasedim(m, dx):
  '''F m = (-m[k,i-1] + m[k,i])/dx
  m[k,i-1] is periodic in i-1
  prepend 0 in axis-0
  @ parameters:
    m: [nt-1, nx] or [nt-1, nx, ny]
  @ return
    out: [nt, nx] or [nt, nx, ny]
  '''
  m_im1 = jnp.roll(m, 1, axis=1)
  out = -m_im1 + m
  out = out/dx
  out = jnp.concatenate([jnp.zeros_like(out[0:1,...]), out], axis = 0) #prepend 0
  return out


@jax.jit
def Dy_right_decreasedim(phi, dy):
  '''F phi = (phi_{k+1,:,i+1}-phi_{k+1,:,i})/dy
  phi_{k+1,:,i+1} is periodic in i+1.
  @ parameters:
    phi: [nt, nx, ny]
  @ return
    out: [nt-1, nx, ny]
  '''
  phi_ip1 = jnp.roll(phi, -1, axis=2)
  out = phi_ip1 - phi
  out = out[1:,...]/dy
  return out

@jax.jit
def Dy_right_increasedim(m, dy):
  '''F m = (-m[k-1,:,i] + m[k-1,:,i+1])/dy
  m[k,:,i+1] is periodic in i+1
  prepend 0 in axis-0
  @ parameters:
    m: [nt-1, nx, ny]
  @ return
    out: [nt, nx, ny]
  '''
  m_ip1 = jnp.roll(m, -1, axis=2)
  out = -m + m_ip1
  out = out/dy
  out = jnp.concatenate([jnp.zeros_like(out[0:1,...]), out], axis = 0) #prepend 0
  return out

@jax.jit
def Dy_left_decreasedim(phi, dy):
  '''F phi = (phi_{k+1,:,i}-phi_{k+1,:,i-1})/dy
  phi_{k+1,:,i-1} is periodic in i+1
  @ parameters:
    phi: [nt, nx, ny]
  @ return
    out: [nt-1, nx, ny]
  '''
  phi_im1 = jnp.roll(phi, 1, axis=2)
  out = phi - phi_im1
  out = out[1:,...]/dy
  return out

@jax.jit
def Dy_left_increasedim(m, dy):
  '''F m = (-m[k,:,i-1] + m[k,:,i])/dy
  m[k,:,i-1] is periodic in i-1
  prepend 0 in axis-0
  @ parameters:
    m: [nt-1, nx, ny]
  @ return
    out: [nt, nx, ny]
  '''
  m_im1 = jnp.roll(m, 1, axis=2)
  out = -m_im1 + m
  out = out/dy
  out = jnp.concatenate([jnp.zeros_like(out[0:1,...]), out], axis = 0) #prepend 0
  return out


@jax.jit
def Dt_decreasedim(phi, dt):
  '''Dt phi = (phi_{k+1,...}-phi_{k,...})/dt
  phi_{k+1,...} is not periodic
  @ parameters:
    phi: [nt, nx] or [nt, nx, ny]
  @ return
    out: [nt-1, nx] or [nt-1, nx, ny]
  '''
  phi_kp1 = phi[1:,...]
  phi_k = phi[:-1,...]
  out = (phi_kp1 - phi_k) /dt
  return out

@jax.jit
def Dxx_decreasedim(phi, dx):
  '''Dxx phi = (phi_{k+1,i+1}+phi_{k+1,i-1}-2*phi_{k+1,i})/dx^2
  phi_{k+1,i} is periodic in i, but not in k
  @ parameters:
    phi: [nt, nx] or [nt, nx, ny]
  @ return
    out: [nt-1, nx] or [nt-1, nx, ny]
  '''
  phi_kp1 = phi[1:,:]
  phi_ip1 = jnp.roll(phi_kp1, -1, axis=1)
  phi_im1 = jnp.roll(phi_kp1, 1, axis=1)
  out = (phi_ip1 + phi_im1 - 2*phi_kp1)/dx**2
  return out

@jax.jit
def Dyy_decreasedim(phi, dy):
  '''Dxx phi = (phi_{k+1,:,i+1}+phi_{k+1,:,i-1}-2*phi_{k+1,:,i})/dy^2
  phi_{k+1,:,i} is periodic in i, but not in k
  @ parameters:
    phi: [nt, nx, ny]
  @ return
    out: [nt-1, nx, ny]
  '''
  phi_kp1 = phi[1:,...]
  phi_ip1 = jnp.roll(phi_kp1, -1, axis=2)
  phi_im1 = jnp.roll(phi_kp1, 1, axis=2)
  out = (phi_ip1 + phi_im1 - 2*phi_kp1)/dy**2
  return out


@jax.jit
def Dt_increasedim(rho, dt):
  '''Dt rho = (-rho[k-1,...] + rho[k,...])/dt
            #k = 0...(nt-1)
  rho[-1,:] = 0
  @ parameters:
    rho: [nt-1, nx] or [nt-1, nx, ny]
  @ return
    out: [nt, nx] or [nt, nx, ny]
  '''
  rho_km1 = jnp.concatenate([jnp.zeros_like(rho[0:1,...]), rho], axis = 0) #prepend 0
  rho_k = jnp.concatenate([rho, jnp.zeros_like(rho[0:1,...])], axis = 0) #append 0
  out = (-rho_km1 + rho_k)/dt
  return out

@jax.jit
def Dxx_increasedim(rho, dx):
  '''F rho = (rho[k-1,i+1]+rho[k-1,i-1]-2*rho[k-1,i])/dx^2
            #k = 0...(nt-1)
  rho[-1,:] = 0
  @ parameters:
    rho: [nt-1, nx] or [nt-1, nx, ny]
  @ return
    out: [nt, nx] or [nt, nx, ny]
  '''
  rho_km1 = jnp.concatenate([jnp.zeros_like(rho[0:1,...]), rho], axis = 0) #prepend 0
  rho_im1 = jnp.roll(rho_km1, 1, axis=1)
  rho_ip1 = jnp.roll(rho_km1, -1, axis=1)
  out = (rho_ip1 + rho_im1 - 2*rho_km1) /dx**2
  return out

@jax.jit
def Dyy_increasedim(rho, dy):
  '''F rho = (rho[k-1,:,i+1]+rho[k-1,:,i-1]-2*rho[k-1,:,i])/dy^2
            #k = 0...(nt-1)
  rho[-1,:] = 0
  @ parameters:
    rho: [nt-1, nx, ny]
  @ return
    out: [nt, nx, ny]
  '''
  rho_km1 = jnp.concatenate([jnp.zeros_like(rho[0:1,...]), rho], axis = 0) #prepend 0
  rho_im1 = jnp.roll(rho_km1, 1, axis=2)
  rho_ip1 = jnp.roll(rho_km1, -1, axis=2)
  out = (rho_ip1 + rho_im1 - 2*rho_km1) /dy**2
  return out

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
    fns_dict: dict of functions, if using vmethod, should contain Hstar_plus_prox_fn, Hstar_minus_prox_fn, Hstar_plus_fn, Hstar_minus_fn
                      in both cases, should contain H_plus_fn, H_minus_fn
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

  c_max = 50 * c_on_rho
  delta_c = 50

  scale = 1.0
  # scale = 1.5
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
                        [[1/(dx*dx)] + [0.0] * (ny-1)])
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
    err2 = jnp.linalg.norm(rho_next - rho_prev) / jnp.maximum(jnp.linalg.norm(rho_prev), 1.0) ** 2
    for v0, v1 in zip(v_prev, v_next):
      err2 += jnp.linalg.norm(v1 - v0) / jnp.maximum(jnp.linalg.norm(v0), 1.0) ** 2
    err2 = jnp.sqrt(err2)
    # err3: equation error
    if ndim == 1:
      HJ_residual = compute_HJ_residual_EO_1d_general(phi_next, dt, dspatial, fns_dict, epsl, x_arr, t_arr)
    elif ndim == 2:
      HJ_residual = compute_HJ_residual_EO_2d_general(phi_next, dt, dspatial, fns_dict, epsl, x_arr, t_arr)
    err3 = jnp.mean(jnp.abs(HJ_residual))
    
    error = jnp.array([err1, err2,err3])
    error_all.append(error)
    if error[2] < eps:
      print('PDHG converges at iter {}'.format(i), flush=True)
      break
    if jnp.isnan(error[0]) or jnp.isnan(error[1]):
      print("Nan error at iter {}".format(i))
      break
    # if jnp.min(rho_next) < -c_on_rho + eps and c_on_rho < c_max:
    #   print('increase c value from {} to {}'.format(c_on_rho, c_on_rho + delta_c), flush = True)
    #   c_on_rho += delta_c
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
    rho0 = jnp.zeros([time_step_per_PDHG-1, nx])
    vp0 = jnp.zeros([time_step_per_PDHG-1, nx])
    vm0 = jnp.zeros([time_step_per_PDHG-1, nx])
    v0 = (vp0, vm0)
  else:
    nx, ny = nspatial[0], nspatial[1]
    rho0 = jnp.zeros([time_step_per_PDHG-1, nx, ny])
    vxp0 = jnp.zeros([time_step_per_PDHG-1, nx, ny])
    vxm0 = jnp.zeros([time_step_per_PDHG-1, nx, ny])
    vyp0 = jnp.zeros([time_step_per_PDHG-1, nx, ny])
    vym0 = jnp.zeros([time_step_per_PDHG-1, nx, ny])
    v0 = (vxp0, vxm0, vyp0, vym0)
  
  phi_all = []
  # vp_all = []
  # vm_all = []
  # rho_all = []
  
  for i in range(nt_PDHG):
    # # note: check true solution IC, TODO: remove this !!!!!!!!!!!!!!!!!!!!!!!!!!
    # t_arr = jnp.array([i*dt, (i+1)*dt])[:,None]
    # phi0 = (x_arr[...,0]-1)**2/2 / (1 + t_arr)
    # vp0 = jnp.maximum(x_arr[...,0]-1, 0) / (1 + t_arr[1:,:])
    # vm0 = jnp.minimum(x_arr[...,0]-1, 0) / (1 + t_arr[1:,:])
    # v0 = (vp0, vm0)
    utils.timer.tic('pdhg_iter{}'.format(i))
    print('nt_PDHG = {}, i = {}'.format(nt_PDHG, i), flush=True)
    t_arr = jnp.linspace(i* dt* (time_step_per_PDHG-1), (i+1)* dt* (time_step_per_PDHG-1), num = time_step_per_PDHG)[1:]  # [time_step_per_PDHG-1]
    if ndim == 1:
      t_arr = t_arr[:,None]  # [time_step_per_PDHG-1, 1]
    else:
      t_arr = t_arr[:,None,None]  # [time_step_per_PDHG-1, 1, 1]
    results_all, _ = PDHG_solver_oneiter(fn_update_primal, fn_update_dual, ndim, phi0, rho0, v0, 
                                    dt, dspatial, c_on_rho, fns_dict, x_arr, t_arr,
                                    N_maxiter = N_maxiter, print_freq = print_freq, eps = eps,
                                    epsl = epsl, stepsz_param=stepsz_param)
    _, v_curr, rho_curr, _, phi_curr = results_all[-1]
    # print('phi_curr: ', phi_curr, flush=True)
    if i < nt_PDHG-1:
      phi_all.append(phi_curr[:-1,:])
    else:
      phi_all.append(phi_curr)
    # vp_all.append(v_curr[0])
    # vm_all.append(v_curr[1])
    # rho_all.append(rho_curr)
    g_diff = phi_curr[-1:,...] - phi0[0:1,...]
    phi0 = phi0 + g_diff
    rho0 = rho_curr
    v0 = v_curr
    utils.timer.toc('pdhg_iter{}'.format(i))
  phi_out = jnp.concatenate(phi_all, axis = 0)
  # vp_out = jnp.concatenate(vp_all, axis = 0)
  # vm_out = jnp.concatenate(vm_all, axis = 0)
  # rho_out = jnp.concatenate(rho_all, axis = 0)
  # results_out = [(0, (vp_out, vm_out), rho_out, None, phi_out)]
  results_out = [(0, None, None, None, phi_out)]
  return results_out, None

def main(argv):
  import pdhg1d_m_2var
  import pdhg1d_v_2var
  import pdhg1d_v_samealg
  import pdhg1d_v_diffalg
  from solver import set_up_example_fns
  import save_analysis

  epsl = FLAGS.epsl
  vmethod = FLAGS.vmethod
  stepsz_param = FLAGS.stepsz_param
  ndim = FLAGS.ndim
  nt = FLAGS.nt
  nx = FLAGS.nx
  egno = FLAGS.egno
  N_maxiter = FLAGS.N_maxiter
  theoretical_ver = FLAGS.theoretical_scheme

  time_step_per_PDHG = 2
  print_freq = 100

  eps = 1e-6
  T = 1
  x_period = 2
  c_on_rho = 10.0

  J, fns_dict = set_up_example_fns(egno, ndim, x_period, 2, theoretical_ver=theoretical_ver)

  dx = x_period / (nx)
  dt = T / (nt-1)
  if ndim == 1:
    x_arr = jnp.linspace(0.0, x_period - dx, num = nx)[None,:,None]  # [1, nx, 1]
  else:
    x_arr = jnp.linspace(0.0, x_period - dx, num = nx)  
    y_arr = jnp.linspace(0.0, x_period - dx, num = nx)
    x_mesh, y_mesh = jnp.meshgrid(x_arr, y_arr)  # [nx, ny]
    x_arr = jnp.stack([x_mesh, y_mesh], axis = -1)[None,...]  # [1, nx, ny, 2]
  g = J(x_arr)  # [1, nx]

  if vmethod == 1: # v method
    if ndim == 1:
      fn_update_primal = pdhg1d_v_2var.update_primal_1d
    else:
      fn_update_primal = pdhg1d_v_2var.update_primal_2d
    fn_update_dual = pdhg1d_v_2var.update_dual
    # stepsz_param = 0.1
  elif vmethod == 0: # m method
    fn_update_primal = pdhg1d_m_2var.update_primal_1d
    fn_update_dual = pdhg1d_m_2var.update_dual_1d
    # stepsz_param = 0.9
  elif vmethod == 2: # v method test on linear
    fn_update_primal = pdhg1d_v_samealg.update_primal_1d
    fn_update_dual = pdhg1d_v_samealg.update_dual
  else:
    fn_update_primal = pdhg1d_v_diffalg.update_primal_1d
    fn_update_dual = pdhg1d_v_diffalg.update_dual

  if ndim == 1:
    dspatial = [dx]
    nspatial = [nx]
  else:
    dspatial = [dx, dx]
    nspatial = [nx, nx]
  
  results, errs_none = PDHG_multi_step(fn_update_primal, fn_update_dual, fns_dict, x_arr, nt, nspatial, ndim,
                    g, dt, dspatial, c_on_rho, time_step_per_PDHG = time_step_per_PDHG,
                    N_maxiter = N_maxiter, print_freq = print_freq, eps = eps,
                    epsl = epsl, stepsz_param=stepsz_param)
  t_arr = jnp.linspace(0, T, num = nt)[:,None]
  # print('phi: ', results[-1][-1], flush=True)
  
  if egno == 10:
    phi_true = (x_arr[...,0]-1)**2/2/(1+ t_arr) + epsl /2 * jnp.log(1 + t_arr)
    err = jnp.linalg.norm(results[-1][-1] - phi_true) / jnp.maximum(jnp.linalg.norm(phi_true), 1)
    print('phi error: ', err, flush=True)
  elif egno == 21 and dx == dt:
    phi_true = [g]
    for i in range(nt-1):
      phi_true.append(jnp.roll(phi_true[-1], 1, axis=1))
    phi_true = jnp.concatenate(phi_true, axis = 0)
    err = jnp.linalg.norm(results[-1][-1] - phi_true) / jnp.maximum(jnp.linalg.norm(phi_true), 1)
    print('phi error: ', err, flush=True)

  # save_analysis.save('test', 'results_vmethod_{}'.format(vmethod), (results, errs_none))

if __name__ == '__main__':
  from absl import app, flags, logging
  FLAGS = flags.FLAGS
  flags.DEFINE_float('epsl', 0.0, 'diffusion coefficient')
  flags.DEFINE_float('stepsz_param', 0.9, 'stepsize in pdhg')
  flags.DEFINE_integer('vmethod', 0, '1 if using vmethod, 0 if using mmethod')
  flags.DEFINE_integer('ndim', 1, 'spatial dimension: 1 or 2')
  flags.DEFINE_integer('nt', 6, 'time discretization number')
  flags.DEFINE_integer('nx', 10, 'spatial discretization number')
  flags.DEFINE_integer('egno', 21, 'egno')
  flags.DEFINE_integer('N_maxiter', 2001, 'maximum number of iterations')
  flags.DEFINE_boolean('theoretical_scheme', True, 'true if aligned with theory')

  app.run(main)