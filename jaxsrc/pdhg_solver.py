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
from save_analysis import compute_HJ_residual_EO_1d_general
import matplotlib.pyplot as plt

@jax.jit
def Dx_right_decreasedim(phi, dx):
  '''F phi = (phi_{k+1,i+1}-phi_{k+1,i})/dx
  phi_{k+1,i+1} is periodic in i+1
  @ parameters:
    phi: [nt, nx]
  @ return
    out: [nt-1, nx]
  '''
  phi_ip1 = jnp.roll(phi, -1, axis=1)
  out = phi_ip1 - phi
  out = out[1:,:]/dx
  return out

@jax.jit
def Dx_right_increasedim(m, dx):
  '''F m = (-m[k-1,i] + m[k-1,i+1])/dx
  m[k,i+1] is periodic in i+1
  prepend 0 in axis-0
  @ parameters:
    m: [nt-1, nx]
  @ return
    out: [nt, nx]
  '''
  m_ip1 = jnp.roll(m, -1, axis=1)
  out = -m + m_ip1
  out = out/dx
  out = jnp.pad(out, ((1,0),(0,0)), mode = 'constant', constant_values=0.0) #prepend 0
  return out

@jax.jit
def Dx_left_decreasedim(phi, dx):
  '''F phi = (phi_{k+1,i}-phi_{k+1,i-1})/dx
  phi_{k+1,i-1} is periodic in i+1
  @ parameters:
    phi: [nt, nx]
  @ return
    out: [nt-1, nx]
  '''
  phi_im1 = jnp.roll(phi, 1, axis=1)
  out = phi - phi_im1
  out = out[1:,:]/dx
  return out

@jax.jit
def Dx_left_increasedim(m, dx):
  '''F m = (-m[k,i-1] + m[k,i])/dx
  m[k,i-1] is periodic in i-1
  prepend 0 in axis-0
  @ parameters:
    m: [nt-1, nx]
  @ return
    out: [nt, nx]
  '''
  m_im1 = jnp.roll(m, 1, axis=1)
  out = -m_im1 + m
  out = out/dx
  out = jnp.pad(out, ((1,0),(0,0)), mode = 'constant', constant_values=0.0) #prepend 0
  return out


@jax.jit
def Dt_decreasedim(phi, dt):
  '''Dt phi = (phi_{k+1,i}-phi_{k,i})/dt
  phi_{k+1,i} is not periodic
  @ parameters:
    phi: [nt, nx]
  @ return
    out: [nt-1, nx]
  '''
  phi_kp1 = phi[1:,:]
  phi_k = phi[:-1,:]
  out = (phi_kp1 - phi_k) /dt
  return out

@jax.jit
def Dxx_decreasedim(phi, dx):
  '''Dxx phi = (phi_{k+1,i+1}+phi_{k+1,i-1}-2*phi_{k+1,i})/dx^2
  phi_{k+1,i} is not periodic
  @ parameters:
    phi: [nt, nx]
  @ return
    out: [nt-1, nx]
  '''
  phi_kp1 = phi[1:,:]
  phi_ip1 = jnp.roll(phi_kp1, -1, axis=1)
  phi_im1 = jnp.roll(phi_kp1, 1, axis=1)
  out = (phi_ip1 + phi_im1 - 2*phi_kp1)/dx**2
  return out


@jax.jit
def Dt_increasedim(rho, dt):
  '''F rho = (-rho[k-1,i] + rho[k,i])/dt
            #k = 0...(nt-1)
  rho[-1,:] = 0
  @ parameters:
    rho: [nt-1, nx]
  @ return
    out: [nt, nx]
  '''
  rho_km1 = jnp.pad(rho, ((1,0),(0,0)), mode = 'constant', constant_values=0.0)
  rho_k = jnp.pad(rho, ((0,1),(0,0)),  mode = 'constant', constant_values=0.0)
  out = (-rho_km1 + rho_k)/dt
  return out

@jax.jit
def Dxx_increasedim(rho, dx):
  '''F rho = (rho[k-1,i+1]+rho[k-1,i-1]-2*rho[k-1,i])/dx^2
            #k = 0...(nt-1)
  rho[-1,:] = 0
  @ parameters:
    rho: [nt-1, nx]
  @ return
    out: [nt, nx]
  '''
  rho_km1 = jnp.pad(rho, ((1,0),(0,0)), mode = 'constant', constant_values=0.0)
  rho_im1 = jnp.roll(rho_km1, 1, axis=1)
  rho_ip1 = jnp.roll(rho_km1, -1, axis=1)
  out = (rho_ip1 + rho_im1 - 2*rho_km1) /dx**2
  return out

def PDHG_solver_1d(fn_update_primal, fn_update_dual, phi0, rho0, v0, 
                   dx, dt, c_on_rho, fns_dict,
                   N_maxiter = 1000000, print_freq = 1000, eps = 1e-6,
                   epsl = 0.0,
                   stepsz_param=0.9,
                   if_precondition=True, 
                   dy = 0.0):
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
                      if using mmethod, should contain vectors f_in_H: [1, nx], c_in_H: [1, nx]
                      in both cases, should contain H_plus_fn, H_minus_fn
    N_maxiter: int, maximum number of iterations
    print_gap: int, gap for printing and saving results
    eps: scalar, stopping criterion
    if_saving: bool, if save results
  @ returns:
  '''
  nt,nx = jnp.shape(phi0)
  phi_prev = phi0
  rho_prev = rho0
  vp_prev = v0[0]
  vm_prev = v0[1]

  if if_precondition:
    tau_phi = stepsz_param
  else:
    tau_phi = stepsz_param / (2/dx + 3/dt)

  tau_rho = tau_phi
  scale = 1.5
  tau_rho = tau_rho * scale
  tau_phi = tau_phi / scale

  # fft for preconditioning
  Lap_vec = jnp.array([-2/(dx*dx), 1/(dx*dx)] + [0.0] * (nx-3) + [1/(dx*dx)])
  fv = jnp.fft.fft(Lap_vec)  # [nx]

  H_plus_fn = fns_dict['H_plus_fn']
  H_minus_fn = fns_dict['H_minus_fn']
  error_all = []
  results_all = []

  for i in range(N_maxiter):
    # update p:  [Ndata, ndim]
    phi_next = fn_update_primal(phi_prev, rho_prev, c_on_rho, vp_prev, vm_prev, tau_phi, dt, dx, fv, epsl, if_precondition=if_precondition)
    # extrapolation
    phi_bar = 2 * phi_next - phi_prev
    # update u:  [Ndata, ndim]
    rho_next, vp_next, vm_next = fn_update_dual(phi_bar, rho_prev, c_on_rho, vp_prev, vm_prev, tau_rho, dt, dx, epsl, fns_dict)

    # primal error
    err1 = jnp.linalg.norm(phi_next - phi_prev) / jnp.maximum(jnp.linalg.norm(phi_prev), 1.0)
    # err2: dual error
    err2_rho = jnp.linalg.norm(rho_next - rho_prev) / jnp.maximum(jnp.linalg.norm(rho_prev), 1.0)
    err2_vp = jnp.linalg.norm(vp_next - vp_prev) / jnp.maximum(jnp.linalg.norm(vp_prev), 1.0)
    err2_vm = jnp.linalg.norm(vm_next - vm_prev) / jnp.maximum(jnp.linalg.norm(vm_prev), 1.0)
    err2 = jnp.sqrt(err2_rho*err2_rho + err2_vp*err2_vp + err2_vm*err2_vm)
    # err3: equation error
    HJ_residual = compute_HJ_residual_EO_1d_general(phi_next, dt, dx, H_plus_fn, H_minus_fn, epsl)
    err3 = jnp.mean(jnp.abs(HJ_residual))
    
    error = jnp.array([err1, err2,err3])
    error_all.append(error)
    if error[2] < eps:
      print('PDHG converges at iter {}'.format(i), flush=True)
      break
    if jnp.isnan(error[0]) or jnp.isnan(error[1]):
      print("Nan error at iter {}".format(i))
      break
    if print_freq > 0 and i % print_freq == 0:
      results_all.append((i, [vp_next, vm_next], rho_prev, [], phi_prev))
      print('iteration {}, primal error with prev step {:.2E}, dual error with prev step {:.2E}, eqt error {:.2E}, min rho {:.2f}'.format(i, 
                  error[0],  error[1],  error[2], jnp.min(rho_next)), flush = True)
      print('vm max {:.3E}, vm min {:.3E}, vp max {:.3E}, vp min {:.3E}'.format(jnp.max(vm_next), jnp.min(vm_next), jnp.max(vp_next), jnp.min(vp_next)), flush = True)
    rho_prev = rho_next
    phi_prev = phi_next
    vp_prev = vp_next
    vm_prev = vm_next
  # print the final error
  print('iteration {}, primal error with prev step {:.2E}, dual error with prev step {:.2E}, eqt error {:.2E}'.format(i, error[0],  error[1],  error[2]), flush = True)
  results_all.append((i+1, [vp_next, vm_next], rho_next, [], phi_next))
  return results_all, jnp.array(error_all)
        

def PDHG_multi_step(fn_update_primal, fn_update_dual, fns_dict, nt, nx, ndim,
                    g, dx, dt, c_on_rho, time_step_per_PDHG = 2,
                    N_maxiter = 1000000, print_freq = 1000, eps = 1e-6,
                    epsl = 0.0, stepsz_param=0.9, if_precondition=True, dy = 0.0):
  assert (nt-1) % (time_step_per_PDHG-1) == 0  # make sure nt-1 is divisible by time_step_per_PDHG
  nt_PDHG = (nt-1) // (time_step_per_PDHG-1)
  
  phi0 = einshape("i...->(ki)...", g, k=time_step_per_PDHG)  # repeat each row of g to nt times, [nt, nx] or [nt, nx, ny]
  rho0 = jnp.zeros([time_step_per_PDHG-1, nx])
  vp0 = jnp.zeros([time_step_per_PDHG-1, nx])
  vm0 = jnp.zeros([time_step_per_PDHG-1, nx])
  v0 = [vp0, vm0]
  
  phi_all = []
  if ndim==1:
    pdhg_fn = PDHG_solver_1d
  else:
    raise NotImplementedError
  
  for i in range(nt_PDHG):
    print('nt_PDHG = {}, i = {}'.format(nt_PDHG, i), flush=True)
    results_all, _ = pdhg_fn(fn_update_primal, fn_update_dual, phi0, rho0, v0, 
                                    dx, dt, c_on_rho, fns_dict,
                                    N_maxiter = N_maxiter, print_freq = print_freq, eps = eps,
                                    epsl = epsl, stepsz_param=stepsz_param, if_precondition=if_precondition, dy = dy)
    _, v_curr, rho_curr, _, phi_curr = results_all[-1]
    if i < nt_PDHG-1:
      phi_all.append(phi_curr[:-1,:])
    else:
      phi_all.append(phi_curr)
    g_diff = phi_curr[-1:,:] - phi0[0:1,:]
    phi0 = phi0 + g_diff
    rho0 = rho_curr
    v0 = v_curr
  phi_out = jnp.concatenate(phi_all, axis = 0)
  results_out = [(0, None, None, None, phi_out)]
  return results_out, None

def main(argv):
  import pdhg1d_m_2var
  import pdhg1d_v_2var

  epsl = FLAGS.epsl
  vmethod = FLAGS.vmethod

  nt = 7
  nx = 10
  time_step_per_PDHG = 3

  N_maxiter = 2001
  print_freq = 100

  eps = 1e-6
  T = 1
  x_period = 2
  c_on_rho = 10.0
  alpha = 2 * jnp.pi / x_period
  J = lambda x: jnp.sin(alpha * x)
  f_in_H_fn = lambda x: 0*x
  c_in_H_fn = lambda x: 0*x + 1

  dx = x_period / (nx)
  dt = T / (nt-1)
  x_arr = jnp.linspace(0.0, x_period - dx, num = nx)[None,:]  # [1, nx]
  g = J(x_arr)  # [1, nx]
  f_in_H = f_in_H_fn(x_arr)  # [1, nx]
  c_in_H = c_in_H_fn(x_arr)  # [1, nx]

  if vmethod == 1: # v method
    fn_update_primal = pdhg1d_v_2var.update_primal
    fn_update_dual = pdhg1d_v_2var.update_dual
    Hstar_minus_fn = lambda p: 0*p - f_in_H/2
    Hstar_plus_fn = lambda p: 0*p - f_in_H/2
    Hstar_minus_prox_fn = lambda p, t: jnp.maximum(jnp.minimum(p, 0.0), -c_in_H)
    Hstar_plus_prox_fn = lambda p, t: jnp.maximum(jnp.minimum(p, c_in_H), 0.0)
    fns_dict = {'Hstar_minus_fn': Hstar_minus_fn, 'Hstar_plus_fn': Hstar_plus_fn,
                'Hstar_minus_prox_fn': Hstar_minus_prox_fn, 'Hstar_plus_prox_fn': Hstar_plus_prox_fn}
    stepsz_param = 0.1
  else: # m method
    fn_update_primal = pdhg1d_m_2var.update_primal
    fn_update_dual = pdhg1d_m_2var.update_dual
    fns_dict = {'f_in_H': f_in_H, 'c_in_H': c_in_H}
    stepsz_param = 0.9

  H_plus_fn = lambda x: c_in_H * jnp.maximum(x, 0.0) + f_in_H
  H_minus_fn = lambda x: -c_in_H * jnp.minimum(x, 0.0) + f_in_H
  fns_dict['H_plus_fn'] = H_plus_fn
  fns_dict['H_minus_fn'] = H_minus_fn

  ndim = 1
  PDHG_multi_step(fn_update_primal, fn_update_dual, fns_dict, nt, nx, ndim,
                    g, dx, dt, c_on_rho, time_step_per_PDHG = time_step_per_PDHG,
                    N_maxiter = N_maxiter, print_freq = print_freq, eps = eps,
                    epsl = epsl, stepsz_param=stepsz_param, if_precondition=True, dy = 0.0)

if __name__ == '__main__':
  from absl import app, flags, logging
  FLAGS = flags.FLAGS
  flags.DEFINE_float('epsl', 0.0, 'diffusion coefficient')
  flags.DEFINE_integer('vmethod', 0, '1 if using vmethod, 0 if using mmethod')

  app.run(main)