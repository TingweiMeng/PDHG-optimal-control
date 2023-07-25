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


from pdhg1d_m_2var import A1Mult, get_minimizer_ind, A1TransMult, pdhg_1d_periodic_iter as pdhg_1d_periodic_iter_prev
from pdhg1d_m_2var import A2Mult as A2Mult_prev, A2TransMult as A2TransMult_prev

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

def updating_rho(rho_prev, phi, vp, vm, updating_method, sigma, dt, dx, epsl, c_on_rho, Hstar_plus_fn, Hstar_minus_fn):
  vec = A1Mult_pos(phi, dt, dx) * vp + A1Mult_neg(phi, dt, dx) * vm + A2Mult(phi, epsl, dt, dx) # [nt-1, nx]
  vec = vec + Hstar_plus_fn(vm) * dt + Hstar_minus_fn(vp) * dt
  if updating_method == 0:
    rho_next = jnp.maximum(rho_prev - sigma * vec / dt, -c_on_rho)  # [nt-1, nx]
  elif updating_method == 1:
    rho_next = (rho_prev + c_on_rho) * jnp.exp(-sigma * vec / dt) - c_on_rho  # [nt-1, nx]
  else:
    raise ValueError("rho updating method not implemented")
  return rho_next

def updating_v(vp_prev, vm_prev, phi, rho, updating_method, sigma, dt, dx, c_on_rho, 
               Hstar_plus_help_fn, Hstar_minus_help_fn, eps=1e-4):
  '''
  @ parameters:
    Hstar_plus_help_fn and Hstar_minus_help_fn are gradient function if updating_method == 3
        they are prox point operator otherwise
  '''
  if updating_method == 3:
    pp_next_raw = -A1Mult_pos(phi, dt, dx) / dt
    pm_next_raw = -A1Mult_neg(phi, dt, dx) / dt
    vp_next = Hstar_minus_help_fn(pp_next_raw, 0)  # [nt-1, nx]
    vm_next = Hstar_plus_help_fn(pm_next_raw, 0)  # [nt-1, nx]
  else: 
    if updating_method == 0:
      t = sigma * (rho + c_on_rho + eps)
    elif updating_method == 1:
      t = sigma
    elif updating_method == 2:
      t = sigma / (rho + c_on_rho + eps)
    else:
      raise ValueError("v updating method not implemented")
    vp_next_raw = vp_prev - t * A1Mult_pos(phi, dt, dx) / dt  # [nt-1, nx]
    vm_next_raw = vm_prev - t * A1Mult_neg(phi, dt, dx) / dt  # [nt-1, nx]
    vp_next = Hstar_minus_help_fn(vp_next_raw, t)  # [nt-1, nx]
    vm_next = Hstar_plus_help_fn(vm_next_raw, t)  # [nt-1, nx]  
  return vp_next, vm_next

  
@partial(jax.jit, static_argnames=("if_precondition","v_method", "rho_method", "updating_rho_first",
                                   "Hstar_plus_fn", "Hstar_minus_fn", "Hstar_plus_help_fn", "Hstar_minus_help_fn",
                                   "H_plus_fn", "H_minus_fn"))
def pdhg_1d_periodic_iter(v_method, rho_method, updating_rho_first,
                              Hstar_plus_fn, Hstar_minus_fn, Hstar_plus_help_fn, Hstar_minus_help_fn,
                              H_plus_fn, H_minus_fn,
                              f_in_H, c_in_H, tau, sigma, vp_prev, vm_prev, rho_prev, phi_prev,
                              g, dx, dt, c_on_rho, if_precondition, fv, epsl = 0.0):
  '''
  @ parameters
    v_method: method for updating v:
        0: pdhg with penalty |v-v^l|^2/2
        1: pdhg with penalty (rho+c)|v-v^l|^2/2
        2: pdhg with penalty (rho+c)^2|v-v^l|^2/2
        3: updating using v = nabla H(Dx phi)
    rho_method: method for updating rho:
        0: pdhg with penalty |rho-rho^l|^2/2
        1: pdhg with penalty KL(rho|rho^l)
    updating_rho_first: if 1, update rho first, otherwise update v first
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
  eps = 1e-4
  mp_prev = (rho_prev + c_on_rho + eps) * vp_prev  # [nt-1, nx]
  mm_prev = (rho_prev + c_on_rho + eps) * vm_prev  # [nt-1, nx]
  delta_phi_raw = - tau * (A1TransMult_pos(mp_prev, dt, dx) + A1TransMult_neg(mm_prev, dt, dx) + A2TransMult(rho_prev, epsl, dt, dx)) # [nt, nx]
  delta_phi = delta_phi_raw / dt # [nt, nx]

  if if_precondition:
    phi_next = phi_prev + solver.Poisson_eqt_solver(delta_phi, fv, dt, Neumann_cond = True)
  else:
    # no preconditioning
    phi_next = phi_prev - delta_phi

  # extrapolation
  phi_bar = 2 * phi_next - phi_prev

  rho_v_iters = 10

  rho_prev0 = rho_prev
  vp_prev0 = vp_prev
  vm_prev0 = vm_prev
  if updating_rho_first == 1: 
    for j in range(rho_v_iters):
      rho_next = updating_rho(rho_prev0, phi_bar, vp_prev0, vm_prev0, rho_method, sigma, dt, dx, epsl, c_on_rho,
                            Hstar_plus_fn, Hstar_minus_fn)  # [nt-1, nx]
      vp_next, vm_next = updating_v(vp_prev0, vm_prev0, phi_bar, rho_next, v_method, sigma, dt, dx, c_on_rho, 
                                  Hstar_plus_help_fn, Hstar_minus_help_fn)  # [nt-1, nx]
      rho_prev0 = rho_next
      vp_prev0 = vp_next
      vm_prev0 = vm_next
  else:
    for j in range(rho_v_iters):
      vp_next, vm_next = updating_v(vp_prev0, vm_prev0, phi_bar, rho_prev0, v_method, sigma, dt, dx, c_on_rho, 
                                  Hstar_plus_help_fn, Hstar_minus_help_fn)
      rho_next = updating_rho(rho_prev0, phi_bar, vp_next, vm_next, rho_method, sigma, dt, dx, epsl, c_on_rho,
                            Hstar_plus_fn, Hstar_minus_fn)
      rho_prev0 = rho_next
      vp_prev0 = vp_next
      vm_prev0 = vm_next
  
  # primal error
  err1 = jnp.linalg.norm(phi_next - phi_prev) / jnp.maximum(jnp.linalg.norm(phi_prev), 1.0)
  # err2: dual error
  err2_rho = jnp.linalg.norm(rho_next - rho_prev) / jnp.maximum(jnp.linalg.norm(rho_prev), 1.0)
  err2_vp = jnp.linalg.norm(vp_next - vp_prev) / jnp.maximum(jnp.linalg.norm(vp_prev), 1.0)
  err2_vm = jnp.linalg.norm(vm_next - vm_prev) / jnp.maximum(jnp.linalg.norm(vm_prev), 1.0)
  err2 = jnp.sqrt(err2_rho*err2_rho + err2_vp*err2_vp + err2_vm*err2_vm)
  # err3: equation error
  HJ_residual = compute_HJ_residual_EO_1d_general(phi_next, dt, dx, H_plus_fn, H_minus_fn, epsl)
  # HJ_residual = compute_HJ_residual_EO_1d_xdep(phi_next, dt, dx, f_in_H, c_in_H, epsl)
  err3 = jnp.mean(jnp.abs(HJ_residual))
  return rho_next, phi_next, vp_next, vm_next, jnp.array([err1, err2,err3])



def get_v_from_m(m, rho, c_on_rho, eps=1e-4):
  m_next_plus = jnp.maximum(m, 0.0)
  m_next_minus = jnp.minimum(m, 0.0)
  vp_next = m_next_plus / (rho + c_on_rho + eps)
  vm_next = jnp.roll(m_next_minus, 1, axis = 1) / (rho + c_on_rho + eps)
  return vp_next, vm_next

def get_m_from_v(vp, vm, rho, c_on_rho, eps=1e-4):
  m_prev_plus = (rho + c_on_rho + eps) * vp
  m_prev_minus = (rho + c_on_rho + eps) * vm
  m_prev = m_prev_plus + jnp.roll(m_prev_minus, -1, axis = 1) # [nt-1, nx]
  return m_prev

@partial(jax.jit, static_argnames=("if_precondition", "Hstar_plus_fn", "Hstar_minus_fn", "Hstar_plus_help_fn", "Hstar_minus_help_fn",
                                   "H_plus_fn", "H_minus_fn"))
def pdhg_1d_periodic_iter_prev2(Hstar_plus_fn, Hstar_minus_fn, Hstar_plus_help_fn, Hstar_minus_help_fn,
                              H_plus_fn, H_minus_fn,
                              f_in_H, c_in_H, tau, sigma, vp_prev, vm_prev, rho_prev, phi_prev,
                              g, dx, dt, c_on_rho, if_precondition, fv, epsl = 0.0):
  eps = 1e-4

  mp_prev = (rho_prev + c_on_rho + eps) * vp_prev  # [nt-1, nx]
  mm_prev = (rho_prev + c_on_rho + eps) * vm_prev  # [nt-1, nx]
  delta_phi_raw = - tau * (A1TransMult_pos(mp_prev, dt, dx) + A1TransMult_neg(mm_prev, dt, dx) + A2TransMult(rho_prev, epsl, dt, dx)) # [nt, nx]
  delta_phi = delta_phi_raw / dt # [nt, nx]

  if if_precondition:
    reg_param = 10
    reg_param2 = 0
    f = -2*reg_param *phi_prev[0:1,:]
    phi_next = solver.pdhg_phi_update(delta_phi, phi_prev, fv, dt, Neumann_cond = True, 
                                      reg_param = reg_param, reg_param2=reg_param2, f=f)
    
    # phi_next = phi_prev + solver.Poisson_eqt_solver(delta_phi, fv, dt, Neumann_cond = True)
    # phi_next = phi_prev + solver.pdhg_phi_update(delta_phi, phi_prev, fv, dt, Neumann_cond = True, reg_param = reg_param)
  else:
    # no preconditioning
    phi_next = phi_prev - delta_phi

  # extrapolation
  phi_bar = 2 * phi_next - phi_prev

  m_prev = get_m_from_v(vp_prev, vm_prev, rho_prev, c_on_rho, eps=eps)
  rho_candidates = []
  # the previous version: vec1 = m_prev - sigma * A1Mult(phi_bar, dt, dx)  # [nt-1, nx]
  z = m_prev - sigma * A1Mult(phi_bar, dt, dx) / dt  # [nt-1, nx]
  z_left = jnp.roll(z, 1, axis = 1) # [vec1(:,end), vec1(:,1:end-1)]
  # previous version: vec2 = rho_prev - sigma * A2Mult(phi_bar) + sigma * f_in_H * dt # [nt-1, nx]
  alp = rho_prev - sigma * A2Mult(phi_bar, epsl, dt, dx) / dt + sigma * f_in_H # [nt-1, nx]

  rho_candidates.append(-c_on_rho * jnp.ones_like(rho_prev))  # left bound
  # two possible quadratic terms on G, 4 combinations
  vec3 = -c_in_H * c_in_H * c_on_rho - z * c_in_H
  vec4 = -c_in_H * c_in_H * c_on_rho + z_left * c_in_H
  rho_candidates.append(jnp.maximum(alp, - c_on_rho))  # for rho large, G = 0
  rho_candidates.append(jnp.maximum((alp + vec3)/(1+ c_in_H*c_in_H), - c_on_rho))#  % if G_i = (rho_i + c)c(xi) + a_i
  rho_candidates.append(jnp.maximum((alp + vec4)/(1+ c_in_H*c_in_H), - c_on_rho))#  % if G_i = (rho_i + c)c(xi) - a_{i-1}
  rho_candidates.append(jnp.maximum((alp + vec3 + vec4)/(1+ 2*c_in_H*c_in_H), - c_on_rho)) # we have both terms above
  rho_candidates.append(jnp.maximum(-c_on_rho - z / c_in_H, - c_on_rho)) # boundary term 1
  rho_candidates.append(jnp.maximum(-c_on_rho + z_left / c_in_H, - c_on_rho)) # boundary term 2
  
  rho_candidates = jnp.array(rho_candidates) # [7, nt-1, nx]
  rho_next = get_minimizer_ind(rho_candidates, alp, c_on_rho, z, c_in_H)
  # m is truncation of vec1 into [-(rho_i+c)c(xi), (rho_{i+1}+c)c(x_{i+1})]
  # m_next = jnp.minimum(jnp.maximum(z, -(rho_next + c_on_rho) * c_in_H), 
  #                       (jnp.roll(rho_next, -1, axis = 1) + c_on_rho) * jnp.roll(c_in_H, -1, axis = 1))
  # vp_next, vm_next = get_v_from_m(m_next, rho_next, c_on_rho, eps=eps)
  vp_next, vm_next = updating_v(vp_prev, vm_prev, phi_bar, rho_prev, 2, sigma, dt, dx, c_on_rho, 
                                Hstar_plus_help_fn, Hstar_minus_help_fn)
  m_next = get_m_from_v(vp_next, vm_next, rho_next, c_on_rho, eps=eps)

  # primal error
  err1 = jnp.linalg.norm(phi_next - phi_prev) / jnp.maximum(jnp.linalg.norm(phi_prev), 1.0)
  # err2: dual error
  err2_rho = jnp.linalg.norm(rho_next - rho_prev) / jnp.maximum(jnp.linalg.norm(rho_prev), 1.0)
  err2_m = jnp.linalg.norm(m_next - m_prev) / jnp.maximum(jnp.linalg.norm(m_prev), 1.0)
  err2 = jnp.sqrt(err2_rho*err2_rho + err2_m*err2_m)
  # err3: equation error
  HJ_residual = compute_HJ_residual_EO_1d_general(phi_next, dt, dx, H_plus_fn, H_minus_fn, epsl)
  # HJ_residual = compute_HJ_residual_EO_1d_xdep(phi_next, dt, dx, f_in_H, c_in_H, epsl)
  err3 = jnp.mean(jnp.abs(HJ_residual))

  return rho_next, phi_next, vp_next, vm_next, jnp.array([err1, err2,err3])




def pdhg_1d_periodic_rho_m_EO_L1_xdep(v_method, rho_method, updating_rho_first,
                                          f_in_H, c_in_H, phi0, rho0, v0, stepsz_param, 
                                          g, dx, dt, c_on_rho, if_precondition, 
                                          N_maxiter = 1000000, print_freq = 1000, eps = 1e-6,
                                          epsl = 0.0, if_quad=False, if_prev_codes=False):
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

  if if_prev_codes and if_quad:
    raise ValueError("if_prev_codes and if_quad cannot be both true")

  # define H^*_-, H^*_+, and the help fns
  # here is the 1d case, i.e., t is either a scalar or with the same dim as x
  # for H^*_- and H^*_+, ignore the indicator functions (e.g., quad case, the indicator of (-\infty,0] or [0,+\infty))
  if if_quad:
    Hstar_minus_fn = lambda p: p **2/2
    Hstar_plus_fn = lambda p: p **2/2
    H_plus_fn = lambda x: jnp.maximum(x, 0.0)**2/2
    H_minus_fn = lambda x: jnp.minimum(x, 0.0)**2/2
    if v_method == 3:  # gradient of H_- and H_+
      Hstar_minus_help_fn = lambda x, t: jnp.minimum(x, 0.0)
      Hstar_plus_help_fn = lambda x, t: jnp.maximum(x, 0.0)
    else:  # prox pt: the general operator (x,t)\mapsto argmin_y f(y) + |x-y|^2/(2t)
      Hstar_minus_help_fn = lambda x, t: jnp.minimum(x/(t+1), 0.0)
      Hstar_plus_help_fn = lambda x, t: jnp.maximum(x/(t+1), 0.0)
  else:
    Hstar_minus_fn = lambda p: 0*p - f_in_H/2
    Hstar_plus_fn = lambda p: 0*p - f_in_H/2
    H_plus_fn = lambda x: c_in_H * jnp.maximum(x, 0.0) + f_in_H
    H_minus_fn = lambda x: -c_in_H * jnp.minimum(x, 0.0) + f_in_H
    Hstar_minus_help_fn = lambda p, t: jnp.maximum(jnp.minimum(p, 0.0), -c_in_H)
    Hstar_plus_help_fn = lambda p, t: jnp.maximum(jnp.minimum(p, c_in_H), 0.0)

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
    if if_prev_codes:
      rho_next, phi_next, vp_next, vm_next, error = pdhg_1d_periodic_iter_prev2(Hstar_plus_fn, Hstar_minus_fn, Hstar_plus_help_fn, Hstar_minus_help_fn, 
                                                                          H_plus_fn, H_minus_fn,
                                                                          f_in_H, c_in_H, tau, sigma, vp_prev, vm_prev, rho_prev, phi_prev,
                                                                          g, dx, dt, c_on_rho, if_precondition, fv, epsl)
    else:
      rho_next, phi_next, vp_next, vm_next, error = pdhg_1d_periodic_iter(v_method, rho_method, updating_rho_first,
                                                                        Hstar_plus_fn, Hstar_minus_fn, Hstar_plus_help_fn, Hstar_minus_help_fn,
                                                                        H_plus_fn, H_minus_fn,
                                                                        f_in_H, c_in_H, tau, sigma, vp_prev, vm_prev, rho_prev, phi_prev,
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
      print('vm max {:.3E}, vm min {:.3E}, vp max {:.3E}, vp min {:.3E}'.format(jnp.max(vm_next), jnp.min(vm_next), jnp.max(vp_next), jnp.min(vp_next)), flush = True)
    rho_prev = rho_next
    phi_prev = phi_next
    vp_prev = vp_next
    vm_prev = vm_next
  
  # print the final error
  print('iteration {}, primal error with prev step {:.2E}, dual error with prev step {:.2E}, eqt error {:.2E}'.format(i, error[0],  error[1],  error[2]), flush = True)
  results_all.append((i+1, [vp_next, vm_next], rho_next, [], phi_next))
  return results_all, jnp.array(error_all)


def PDHG_multi_step(egno, f_in_H, c_in_H, stepsz_param, nt, nx,
                    g, dx, dt, c_on_rho, epsl = 0.0, time_step_per_PDHG = 2):
  if_precondition = True
  ifsave = False
  rho_method = 0
  v_method = 2
  updating_rho_first = 0

  assert (nt-1) % (time_step_per_PDHG-1) == 0  # make sure nt-1 is divisible by time_step_per_PDHG
  nt_PDHG = (nt-1) // (time_step_per_PDHG-1)
  
  phi0 = einshape("i...->(ki)...", g, k=time_step_per_PDHG)  # repeat each row of g to nt times, [nt, nx] or [nt, nx, ny]
  rho0 = jnp.zeros([time_step_per_PDHG-1, nx])
  vp0 = jnp.zeros([time_step_per_PDHG-1, nx])
  vm0 = jnp.zeros([time_step_per_PDHG-1, nx])
  v0 = [vp0, vm0]

  N_maxiter = 10000
  print_freq = 100

  if egno == 10:
    if_quad = True
  else:
    if_quad = False

  phi_all = []
  rho_all = []
  vp_all = []
  vm_all = []
  for i in range(nt_PDHG):
    print('nt_PDHG = {}, i = {}'.format(nt_PDHG, i), flush=True)
    results_all, _ = pdhg_1d_periodic_rho_m_EO_L1_xdep(v_method, rho_method, updating_rho_first,
                                          f_in_H, c_in_H, phi0, rho0, v0, stepsz_param, 
                                          g, dx, dt, c_on_rho, if_precondition, 
                                          N_maxiter = N_maxiter, print_freq = print_freq, eps = 1e-6,
                                          epsl = epsl, if_quad=if_quad, if_prev_codes=False)
    _, v_curr, rho_curr, _, phi_curr = results_all[-1]
    if i < nt_PDHG-1:
      phi_all.append(phi_curr[:-1,:])
      rho_all.append(rho_curr[:-1,:])
      vp_all.append(v_curr[0][:-1,:])
      vm_all.append(v_curr[1][:-1,:])
    else:
      phi_all.append(phi_curr)
      rho_all.append(rho_curr)
      vp_all.append(v_curr[0])
      vm_all.append(v_curr[1])
    g_diff = phi_curr[-1:,:] - phi0[0:1,:]
    phi0 = phi0 + g_diff
    rho0 = rho_curr
    v0 = v_curr
  phi_all = jnp.concatenate(phi_all, axis = 0)
  rho_all = jnp.concatenate(rho_all, axis = 0)
  vp_all = jnp.concatenate(vp_all, axis = 0)
  vm_all = jnp.concatenate(vm_all, axis = 0)


def main(argv):
  from solver import set_up_example_fns
  egno = FLAGS.egno
  stepsz_param = FLAGS.stepsz_param
  c_on_rho = FLAGS.c_on_rho

  epsl = 0.1

  T = 1
  x_period, y_period = 2, 2
  nx = 10
  nt = 6

  J, f_in_H_fn, c_in_H_fn = set_up_example_fns(egno, 1, x_period, y_period)

  dx = x_period / (nx)
  dt = T / (nt-1)
  spatial_arr = jnp.linspace(0.0, x_period - dx, num = nx)[None,:,None]  # [1, nx, 1]
  g = J(spatial_arr)  # [1, nx]
  f_in_H = f_in_H_fn(spatial_arr)  # [1, nx]
  c_in_H = c_in_H_fn(spatial_arr)  # [1, nx]

  PDHG_multi_step(egno, f_in_H, c_in_H, stepsz_param, nt, nx,
                    g, dx, dt, c_on_rho, epsl = epsl, time_step_per_PDHG = 2)


if __name__ == '__main__':
  from absl import app, flags, logging
  FLAGS = flags.FLAGS
  flags.DEFINE_integer('egno', 0, 'index of example')
  flags.DEFINE_float('stepsz_param', 0.1, 'default step size constant')
  flags.DEFINE_float('c_on_rho', 10.0, 'the constant added on rho')

  app.run(main)