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
def A1Mult(phi, dt, dx):
  '''A1 phi = (-phi_{k+1,i+1}+phi_{k+1,i})*dt/dx
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
def A1TransMult(m, dt, dx):
  '''A1.T m = (-m[k,i-1] + m[k,i])*dt/dx
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


def get_Gsq_from_rho(rho_plus_c_mul_cinH, z):
  '''
  @parameters:
    rho_plus_c_mul_cinH: [5, nt-1, nx]
    z: [nt-1, nx]
  @return 
    fn_val: [5, nt-1, nx]
  '''
  n_can = jnp.shape(rho_plus_c_mul_cinH)[0]
  z_left = jnp.roll(z, 1, axis = 1)
  z_rep = einshape("ij->kij", z, k=n_can)
  z_left_rep = einshape("ij->kij", z_left, k=n_can)
  G1 = jnp.minimum(rho_plus_c_mul_cinH + z_rep, 0) # when z < 0
  G2 = jnp.minimum(rho_plus_c_mul_cinH - z_left_rep, 0) # when z_left >=0
  G = jnp.zeros_like(rho_plus_c_mul_cinH)
  G = jnp.where(z_rep < 0, G + G1 ** 2, G)
  G = jnp.where(z_left_rep >= 0, G + G2 ** 2, G)
  return G # [n_can, nt-1, nx]


def get_minimizer_ind(rho_candidates, shift_term, c, z, c_in_H):
  '''
  A2_mul_phi is of size ((nt-1)*nx, 1)
  for each (k,i) index, find min_r (r - shift_term)^2 + G(rho)_{k,i}^2 in candidates
  @ parameters:
    rho_candidates: [7, nt-1, nx]
    shift_term: [nt-1, nx]
    c: scalar
    z: [nt-1, nx]
    c_in_H: [1, nx]
  @ return: 
    rho_min: [nt-1, nx]
  '''
  fn_val = (rho_candidates - shift_term[None,:,:])**2 # [5, nt-1, nx]
  fn_val_p = fn_val + get_Gsq_from_rho((rho_candidates + c) * c_in_H[None,:,:], z)
  minindex = jnp.argmin(fn_val_p, axis=0, keepdims=True)
  rho_min = jnp.take_along_axis(rho_candidates, minindex, axis = 0)
  return rho_min[0,:,:]


def update_primal(phi_prev, rho_prev, c_on_rho, m_prev, dummy_prev, tau, dt, dx, fv, epsl, if_precondition):
  delta_phi_raw = - tau * (A1TransMult(m_prev, dt, dx) + A2TransMult(rho_prev, epsl, dt, dx)) # [nt, nx]
  delta_phi = delta_phi_raw / dt # [nt, nx]

  if if_precondition:
    # phi_next = phi_prev + solver.Poisson_eqt_solver(delta_phi, fv, dt, Neumann_cond = True)
    reg_param = 10
    reg_param2 = 1
    f = -2*reg_param *phi_prev[0:1,:]
    # phi_next = phi_prev + solver.pdhg_phi_update(delta_phi, phi_prev, fv, dt, Neumann_cond = True, reg_param = reg_param)
    phi_next = solver.pdhg_precondition_update(delta_phi, phi_prev, fv, dt, Neumann_cond = True, 
                                      reg_param = reg_param, reg_param2=reg_param2, f=f)
  else:
    # no preconditioning
    phi_next = phi_prev - delta_phi
  return phi_next

def update_dual(phi_bar, rho_prev, c_on_rho, m_prev, dummy_prev, sigma, dt, dx, epsl, 
                fns_dict):
  c_in_H = fns_dict['c_in_H']
  f_in_H = fns_dict['f_in_H']
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
  m_next = jnp.minimum(jnp.maximum(z, -(rho_next + c_on_rho) * c_in_H), 
                        (jnp.roll(rho_next, -1, axis = 1) + c_on_rho) * jnp.roll(c_in_H, -1, axis = 1))
  return rho_next, m_next, 0*m_next

# def update_phi_preconditioning(delta_phi, phi_prev, fv, dt):
#   ''' this solves -(D_{tt} + D_{xx}) phi = delta_phi with zero Dirichlet at t=0 and 0 Neumann at t=T
#   @parameters:
#     delta_phi: [nt, nx]
#     phi_prev: [nt, nx]
#     fv: [nx], complex, this is FFT of neg Laplacian -Dxx
#     dt: scalar
#   @return:
#     phi_next: [nt, nx]
#   '''
#   nt, nx = jnp.shape(delta_phi)
#   # exclude the first row wrt t
#   v_Fourier =  jnp.fft.fft(delta_phi[1:,:], axis = 1)  # [nt, nx]
#   dl = jnp.pad(1/(dt*dt)*jnp.ones((nt-2,)), (1,0), mode = 'constant', constant_values=0.0).astype(jnp.complex128)
#   du = jnp.pad(1/(dt*dt)*jnp.ones((nt-2,)), (0,1), mode = 'constant', constant_values=0.0).astype(jnp.complex128)
#   neg_Lap_t_diag = jnp.array([-2/(dt*dt)] * (nt-2) + [-1/(dt*dt)])  # [nt-1]
#   neg_Lap_t_diag_rep = einshape('n->nm', neg_Lap_t_diag, m = nx)  # [nt-1, nx]
#   thomas_b = einshape('n->mn', fv, m = nt-1) + neg_Lap_t_diag_rep # [nt-1, nx]
  
#   phi_fouir_part = solver.tridiagonal_solve_batch(dl, thomas_b, du, v_Fourier) # [nt-1, nx]
#   F_phi_updates = jnp.fft.ifft(phi_fouir_part, axis = 1).real # [nt-1, nx]
#   phi_next = phi_prev + jnp.concatenate([jnp.zeros((1,nx)), F_phi_updates], axis = 0) # [nt, nx]
#   return phi_next

@partial(jax.jit, static_argnames=("if_precondition",))
def pdhg_1d_periodic_iter(f_in_H, c_in_H, tau, sigma, m_prev, rho_prev, phi_prev,
                              g, dx, dt, c_on_rho, if_precondition, fv, epsl = 0.0):
  '''
  @ parameters
    f_in_H: [1, nx]
    c_in_H: [1, nx]
    tau: scalar
    sigma: scalar
    m_prev: [nt-1, nx]
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
    m_next: [nt-1, nx]
    err: jnp.array([err1, err2,err3])
  '''

  delta_phi_raw = - tau * (A1TransMult(m_prev, dt, dx) + A2TransMult(rho_prev, epsl, dt, dx)) # [nt, nx]
  delta_phi = delta_phi_raw / dt # [nt, nx]

  if if_precondition:
    # phi_next = phi_prev + solver.Poisson_eqt_solver(delta_phi, fv, dt, Neumann_cond = True)
    reg_param = 10
    reg_param2 = 1
    f = -2*reg_param *phi_prev[0:1,:]
    # phi_next = phi_prev + solver.pdhg_phi_update(delta_phi, phi_prev, fv, dt, Neumann_cond = True, reg_param = reg_param)
    phi_next = solver.pdhg_phi_update(delta_phi, phi_prev, fv, dt, Neumann_cond = True, 
                                      reg_param = reg_param, reg_param2=reg_param2, f=f)
  else:
    # no preconditioning
    phi_next = phi_prev - delta_phi

  # extrapolation
  phi_bar = 2 * phi_next - phi_prev

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
  m_next = jnp.minimum(jnp.maximum(z, -(rho_next + c_on_rho) * c_in_H), 
                        (jnp.roll(rho_next, -1, axis = 1) + c_on_rho) * jnp.roll(c_in_H, -1, axis = 1))

  # primal error
  err1 = jnp.linalg.norm(phi_next - phi_prev) / jnp.maximum(jnp.linalg.norm(phi_prev), 1.0)
  # err2: dual error
  err2_rho = jnp.linalg.norm(rho_next - rho_prev) / jnp.maximum(jnp.linalg.norm(rho_prev), 1.0)
  err2_m = jnp.linalg.norm(m_next - m_prev) / jnp.maximum(jnp.linalg.norm(m_prev), 1.0)
  err2 = jnp.sqrt(err2_rho*err2_rho + err2_m*err2_m)
  # err3: equation error
  HJ_residual = compute_HJ_residual_EO_1d_xdep(phi_next, dt, dx, f_in_H, c_in_H, epsl)
  err3 = jnp.mean(jnp.abs(HJ_residual))
  return rho_next, phi_next, m_next, jnp.array([err1, err2,err3])


def pdhg_1d_periodic_rho_m_EO_L1_xdep(f_in_H, c_in_H, phi0, rho0, m0, stepsz_param, 
                                          g, dx, dt, c_on_rho, if_precondition, 
                                          N_maxiter = 1000000, print_freq = 1000, eps = 1e-6,
                                          epsl = 0.0):
  '''
  @ parameters:
    f_in_H: [1, nx]
    c_in_H: [1, nx]
    phi0: [nt, nx]
    rho0: [nt-1, nx]
    m0: [nt-1, nx]
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
    results_all: list of (iter_no, m, rho, mu, phi)
    error_all: [#pdhg iter, 3]
  '''
  nt,nx = jnp.shape(phi0)
  phi_prev = phi0
  rho_prev = rho0
  m_prev = m0
  c_max = 50 * c_on_rho
  delta_c = 50

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
    rho_next, phi_next, m_next, error = pdhg_1d_periodic_iter(f_in_H, c_in_H, tau, sigma, m_prev, rho_prev, phi_prev,
                                                                           g, dx, dt, c_on_rho, if_precondition, fv, epsl)
    error_all.append(error)
    if error[2] < eps:
      print('PDHG converges at iter {}'.format(i), flush=True)
      break
    if jnp.isnan(error[0]) or jnp.isnan(error[1]):
      print("Nan error at iter {}".format(i))
      break

    if jnp.min(rho_next) < -c_on_rho + eps and c_on_rho < c_max:
      print('increase c value from {} to {}'.format(c_on_rho, c_on_rho + delta_c), flush = True)
      c_on_rho += delta_c
    if print_freq > 0 and i % print_freq == 0:
      results_all.append((i, m_prev, rho_prev, [], phi_prev))
      print('iteration {}, primal error with prev step {:.2E}, dual error with prev step {:.2E}, eqt error {:.2E}, min rho {:.2f}'.format(i, 
                  error[0],  error[1],  error[2], jnp.min(rho_next)), flush = True)
      # if error[0] < eps and error[1] < eps and error[2] > eps and c_on_rho < c_max:
      #   print('increase c value from {} to {}'.format(c_on_rho, c_on_rho + delta_c), flush = True)
      #   c_on_rho += delta_c
    
    rho_prev = rho_next
    phi_prev = phi_next
    m_prev = m_next
  
  # print the final error
  print('iteration {}, primal error with prev step {:.2E}, dual error with prev step {:.2E}, eqt error {:.2E}'.format(i, error[0],  error[1],  error[2]), flush = True)
  results_all.append((i+1, m_next, rho_next, [], phi_next))
  return results_all, jnp.array(error_all)


if __name__ == "__main__":
  time_step_per_PDHG = 2
  nt = 6
  nx = 10
  nt_PDHG = (nt-1) // (time_step_per_PDHG-1)

  epsl = 0.1

  if_precondition = True
  N_maxiter = 2001
  eps = 1e-6
  T = 1
  x_period = 2
  stepsz_param = 0.9
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
    
  phi0 = einshape("i...->(ki)...", g, k=time_step_per_PDHG)  # repeat each row of g to nt times, [nt, nx] or [nt, nx, ny]
  rho0 = jnp.zeros([time_step_per_PDHG-1, nx])
  m0 = jnp.zeros([time_step_per_PDHG-1, nx])

  phi_all = []
  rho_all = []
  m_all = []
  for i in range(nt_PDHG):
    print('nt_PDHG = {}, i = {}'.format(nt_PDHG, i), flush=True)
    output, error_all = pdhg_1d_periodic_rho_m_EO_L1_xdep(f_in_H, c_in_H, phi0, rho0, m0, stepsz_param, 
                          g, dx, dt, c_on_rho, if_precondition, N_maxiter = N_maxiter, print_freq = 100, eps = eps, epsl=epsl)
    _, m_curr, rho_curr, _, phi_curr = output[-1]
    if i < nt_PDHG-1:
      phi_all.append(phi_curr[:-1,:])
      rho_all.append(rho_curr[:-1,:])
      m_all.append(m_curr[:-1,:])
    else:
      phi_all.append(phi_curr)
      rho_all.append(rho_curr)
      m_all.append(m_curr)
    g_diff = phi_curr[-1:,:] - phi0[0:1,:]
    phi0 = phi0 + g_diff
    rho0 = rho_curr
    m0 = m_curr
  phi_all = jnp.concatenate(phi_all, axis = 0)
  rho_all = jnp.concatenate(rho_all, axis = 0)
  m_all = jnp.concatenate(m_all, axis = 0)


  output, error_all = pdhg_1d_periodic_rho_m_EO_L1_xdep(f_in_H, c_in_H, phi0, rho0, m0, stepsz_param, 
                                          g, dx, dt, c_on_rho, if_precondition, N_maxiter = N_maxiter, print_freq = 100, eps = eps, epsl=0.0)