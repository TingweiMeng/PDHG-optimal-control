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
from save_analysis import compute_HJ_residual_EO_1d_xdep
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

@jax.jit
def A1Mult_pos(phi, dt, dx):
  '''A1^+ phi = (-phi_{k+1,i+1}+phi_{k+1,i})*dt/dx
  phi_{k+1,i+1} is periodic in i+1
  @ parameters:
    phi: [nt, nx]
  @ return
    out: [nt-1, nx]
  '''
  phi_ip1 = jnp.roll(phi, -1, axis=1)
  out = -phi_ip1 + phi
  out = out[1:,:]*dt/dx
  return out

@jax.jit
def A1Mult_neg(phi, dt, dx):
  '''A1^- phi = (-phi_{k+1,i}+phi_{k+1,i-1})*dt/dx
  phi_{k+1,i-1} is periodic in i+1
  @ parameters:
    phi: [nt, nx]
  @ return
    out: [nt-1, nx]
  '''
  phi_im1 = jnp.roll(phi, 1, axis=1)
  out = -phi + phi_im1
  out = out[1:,:]*dt/dx
  return out


@jax.jit
def A1TransMult_pos(m, dt, dx):
  '''A1^+.T m = (-m[k-1,i-1] + m[k-1,i])*dt/dx
  m[k,i-1] is periodic in i-1
  prepend 0 in axis-0
  @ parameters:
    m: [nt-1, nx]
  @ return
    out: [nt, nx]
  '''
  m_im1 = jnp.roll(m, 1, axis=1)
  out = -m_im1 + m
  out = out*dt/dx
  out = jnp.pad(out, ((1,0),(0,0)), mode = 'constant', constant_values=0.0) #prepend 0
  return out

@jax.jit
def A1TransMult_neg(m, dt, dx):
  '''A1^-.T m = (-m[k-1,i] + m[k-1,i+1])*dt/dx
  m[k,i+1] is periodic in i+1
  prepend 0 in axis-0
  @ parameters:
    m: [nt-1, nx]
  @ return
    out: [nt, nx]
  '''
  m_ip1 = jnp.roll(m, -1, axis=1)
  out = -m + m_ip1
  out = out*dt/dx
  out = jnp.pad(out, ((1,0),(0,0)), mode = 'constant', constant_values=0.0) #prepend 0
  return out

@jax.jit
def A2Mult(phi, epsl, dt, dx):
  '''A2 phi = -phi_{k+1,i}+phi_{k,i} + eps*dt*(phi_{k+1,i+1}+phi_{k+1,i-1}-2*phi_{k+1,i})/dx^2
  phi_{k+1,i} is not periodic
  @ parameters:
    phi: [nt, nx]
  @ return
    out: [nt-1, nx]
  '''
  phi_kp1 = phi[1:,:]
  phi_k = phi[:-1,:]
  phi_ip1 = jnp.roll(phi_kp1, -1, axis=1)
  phi_im1 = jnp.roll(phi_kp1, 1, axis=1)
  out = -phi_kp1 + phi_k + (phi_ip1 + phi_im1 - 2*phi_kp1) * epsl * dt/dx**2
  return out


@jax.jit
def A2TransMult(rho, epsl, dt, dx):
  '''A2.T rho = (-rho[k-1,i] + rho[k,i]) +eps*dt*(rho[k-1,i+1]+rho[k-1,i-1]-2*rho[k-1,i])/dx^2
            #k = 0...(nt-1)
  rho[-1,:] = 0
  @ parameters:
    rho: [nt-1, nx]
  @ return
    out: [nt, nx]
  '''
  rho_km1 = jnp.pad(rho, ((1,0),(0,0)), mode = 'constant', constant_values=0.0)
  rho_k = jnp.pad(rho, ((0,1),(0,0)),  mode = 'constant', constant_values=0.0)
  rho_im1 = jnp.roll(rho_km1, 1, axis=1)
  rho_ip1 = jnp.roll(rho_km1, -1, axis=1)
  out = -rho_km1 + rho_k + (rho_ip1 + rho_im1 - 2*rho_km1) * epsl * dt/dx**2
  return out

@jax.jit
def update_phi_preconditioning(delta_phi, phi_prev, fv, dt):
  ''' this solves -(D_{tt} + D_{xx}) phi = delta_phi with zero Dirichlet at t=0 and 0 Neumann at t=T
  @parameters:
    delta_phi: [nt, nx]
    phi_prev: [nt, nx]
    fv: [nx], complex, this is FFT of neg Laplacian -Dxx
    dt: scalar
  @return:
    phi_next: [nt, nx]
  '''
  nt, nx = jnp.shape(delta_phi)
  # exclude the first row wrt t
  v_Fourier =  jnp.fft.fft(delta_phi[1:,:], axis = 1)  # [nt, nx]
  dl = jnp.pad(1/(dt*dt)*jnp.ones((nt-2,)), (1,0), mode = 'constant', constant_values=0.0).astype(jnp.complex128)
  du = jnp.pad(1/(dt*dt)*jnp.ones((nt-2,)), (0,1), mode = 'constant', constant_values=0.0).astype(jnp.complex128)
  neg_Lap_t_diag = jnp.array([-2/(dt*dt)] * (nt-2) + [-1/(dt*dt)])  # [nt-1]
  neg_Lap_t_diag_rep = einshape('n->nm', neg_Lap_t_diag, m = nx)  # [nt-1, nx]
  thomas_b = einshape('n->mn', fv, m = nt-1) + neg_Lap_t_diag_rep # [nt-1, nx]
  
  phi_fouir_part = solver.tridiagonal_solve_batch(dl, thomas_b, du, v_Fourier) # [nt-1, nx]
  F_phi_updates = jnp.fft.ifft(phi_fouir_part, axis = 1).real # [nt-1, nx]
  phi_next = phi_prev + jnp.concatenate([jnp.zeros((1,nx)), F_phi_updates], axis = 0) # [nt, nx]
  return phi_next

@partial(jax.jit, static_argnames=("if_precondition",))
def pdhg_1d_periodic_iter(f_in_H, c_in_H, tau, sigma, vp_prev, vm_prev, rho_prev, phi_prev,
                              g, dx, dt, c_on_rho, if_precondition, fv, epsl = 0.0):
  '''
  @ parameters
    f_in_H: [1, nx]
    c_in_H: [1, nx]
    tau: scalar
    sigma: scalar
    vp_prev: [nt-1, nx]
    vm_prev: [nt-1, nx]
    rho_prev: [nt-1, nx]
    phi_prev: [nt, nx]
    g: [1, nx]
    dx: scalar
    dt: scalar
    c_on_rho: scalar
    if_precondition: bool
    fv: [nx]
    epsl: scalar, diffusion coefficient

  @ return 
    rho_next: [nt-1, nx]
    phi_next: [nt, nx]
    vp_next: [nt-1, nx]
    vm_next: [nt-1, nx]
    err: jnp.array([err1, err2,err3])
  '''
  mp_prev = (rho_prev + c_on_rho) * vp_prev  # [nt-1, nx]
  mm_prev = (rho_prev + c_on_rho) * vm_prev  # [nt-1, nx]
  delta_phi_raw = - tau * (A1TransMult_pos(mp_prev, dt, dx) + A1TransMult_neg(mm_prev, dt, dx) + A2TransMult(rho_prev, epsl, dt, dx)) # [nt, nx]
  delta_phi = delta_phi_raw / dt # [nt, nx]

  if if_precondition:
    phi_next = update_phi_preconditioning(delta_phi, phi_prev, fv, dt)
  else:
    # no preconditioning
    phi_next = phi_prev - delta_phi

  # extrapolation
  phi_bar = 2 * phi_next - phi_prev

  # update rho
  vec = A1Mult_pos(phi_bar, dt, dx) * vp_prev + A1Mult_neg(phi_bar, dt, dx) * vm_prev + A2Mult(phi_bar, epsl, dt, dx) # [nt-1, nx]
  rho_next = jnp.maximum(rho_prev - sigma * vec / dt, -c_on_rho)  # [nt-1, nx]
  # update vp and vm
  vp_next_raw = vp_prev - sigma * (rho_next + c_on_rho) * A1Mult_pos(phi_bar, dt, dx) / dt  # [nt-1, nx]
  vp_next = jnp.maximum(jnp.minimum(vp_next_raw, 0.0), -c_in_H)  # [nt-1, nx]
  vm_next_raw = vm_prev - sigma * (rho_next + c_on_rho) * A1Mult_neg(phi_bar, dt, dx) / dt  # [nt-1, nx]
  vm_next = jnp.maximum(jnp.minimum(vm_next_raw, c_in_H), 0.0)  # [nt-1, nx]
  
  # primal error
  err1 = jnp.linalg.norm(phi_next - phi_prev)
  # err2: dual error
  err2_rho = jnp.linalg.norm(rho_next - rho_prev)
  err2_vp = jnp.linalg.norm(vp_next - vp_prev)
  err2_vm = jnp.linalg.norm(vm_next - vm_prev)
  err2 = jnp.sqrt(err2_rho*err2_rho + err2_vp*err2_vp + err2_vm*err2_vm)
  # err3: equation error
  HJ_residual = compute_HJ_residual_EO_1d_xdep(phi_next, dt, dx, f_in_H, c_in_H, epsl)
  err3 = jnp.mean(jnp.abs(HJ_residual))
  return rho_next, phi_next, vp_next, vm_next, jnp.array([err1, err2,err3])


def pdhg_1d_periodic_rho_m_EO_L1_xdep(f_in_H, c_in_H, phi0, rho0, v0, stepsz_param, 
                                          g, dx, dt, c_on_rho, if_precondition, 
                                          N_maxiter = 1000000, print_freq = 1000, eps = 1e-6,
                                          epsl = 0.0):
  '''
  @ parameters:
    f_in_H: [1, nx]
    c_in_H: [1, nx]
    phi0: [nt, nx]
    rho0: [nt-1, nx]
    v0: [vp0, vm0], where vp0 and vm0 are [nt-1, nx]
    stepsz_param: scalar
    g: [1, nx]
    dx: scalar
    dt: scalar
    c_on_rho: scalar
    if_precondition: bool
    N_maxiter: int
    eps: scalar
    epsl: scalar, diffusion coefficient

  @ return 
    results_all: list of (iter_no, [vp, vm], rho, [], phi)
    error_all: [#pdhg iter, 3]
  '''
  nt,nx = jnp.shape(phi0)
  phi_prev = phi0
  rho_prev = rho0
  vp_prev = v0[0]
  vm_prev = v0[1]

  print('epsl: {}'.format(epsl), flush=True)

  if if_precondition:
    tau = stepsz_param
  else:
    tau = stepsz_param / (2/dx + 3/dt)

  sigma = tau
  sigma_scale = 1.5
  sigma = sigma * sigma_scale
  tau = tau / sigma_scale


  if if_precondition:
    # fft for preconditioning
    Lap_vec = jnp.array([-2/(dx*dx), 1/(dx*dx)] + [0.0] * (nx-3) + [1/(dx*dx)])
    fv = jnp.fft.fft(Lap_vec)  # [nx]
  else:
    fv = None

  error_all = []

  results_all = []
  for i in range(N_maxiter):
    rho_next, phi_next, vp_next, vm_next, error = pdhg_1d_periodic_iter(f_in_H, c_in_H, tau, sigma, vp_prev, vm_prev, rho_prev, phi_prev,
                                                                           g, dx, dt, c_on_rho, if_precondition, fv, epsl)
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
    if i % 1 == 0:
      # plot phi, rho, vp, vm
      plt.figure()
      plt.contourf(phi_next)
      plt.colorbar()
      plt.savefig('phi_iter{}.png'.format(i))
      plt.close()
      plt.figure()
      plt.contourf(rho_next)
      plt.colorbar()
      plt.savefig('rho_iter{}.png'.format(i))
      plt.close()
      plt.figure()
      plt.contourf(vp_next)
      plt.colorbar()
      plt.savefig('vp_iter{}.png'.format(i))
      plt.close()
      plt.figure()
      plt.contourf(vm_next)
      plt.colorbar()
      plt.savefig('vm_iter{}.png'.format(i))
      plt.close()


    
    rho_prev = rho_next
    phi_prev = phi_next
    vp_prev = vp_next
    vm_prev = vm_next
  
  # print the final error
  print('iteration {}, primal error with prev step {}, dual error with prev step {}, eqt error {}'.format(i, error[0],  error[1],  error[2]), flush = True)
  results_all.append((i+1, [vp_next, vm_next], rho_next, [], phi_next))
  return results_all, jnp.array(error_all)
