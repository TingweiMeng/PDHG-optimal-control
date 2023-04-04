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
def A2Mult(phi):
  '''A2 phi = -phi_{k+1,i}+phi_{k,i}
  phi_{k+1,i} is not periodic
  @ parameters:
    phi: [nt, nx]
  @ return
    out: [nt-1, nx]
  '''
  phi_kp1 = phi[1:,:]
  phi_k = phi[:-1,:]
  out = -phi_kp1 + phi_k
  return out


@jax.jit
def A2TransMult(rho):
  '''A2.T rho = (-rho[k-1,i] + rho[k,i]) #k = 0...(nt-1)
  rho[-1,:] = 0
  @ parameters:
    rho: [nt-1, nx]
  @ return
    out: [nt, nx]
  '''
  rho_km1 = jnp.pad(rho, ((1,0),(0,0)), mode = 'constant', constant_values=0.0)
  rho_k = jnp.pad(rho, ((0,1),(0,0)),  mode = 'constant', constant_values=0.0)
  out = -rho_km1 + rho_k
  return out


def check_HJ_sol_usingEO_L1_1d_xdep(phi, dt, dx, f_in_H, c_in_H):
  '''
  check a solution is true or not
  H is L1, 1-dimensional
  @parameters:
    phi: [nt, nx]
    dt: scalar
    dx: scalar
    f_in_H: [1, nx]
    c_in_H: [1, nx]
  @ return:
    HJ_residual: [nt-1, nx]
  '''
  dphidx_left = (phi - jnp.roll(phi, 1, axis = 1))/dx
  dphidx_right = (jnp.roll(phi, -1, axis=1) - phi)/dx
  H_val = jnp.maximum(-dphidx_right, 0) + jnp.maximum(dphidx_left, 0)
  H_val = c_in_H * H_val + f_in_H
  HJ_residual = (phi[1:,:] - phi[:-1,:])/dt + H_val[1:,:]
  return HJ_residual


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
    rho_candidates: [5, nt-1, nx]
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


def update_phi_preconditioning(delta_phi, phi_prev, fv, dt):
  '''
  @parameters:
    delta_phi: [nt, nx]
    phi_prev: [nt, nx]
    fv: [nx], complex
    dt: scalar
  @return:
    phi_next: [nt, nx]
  '''
  nt, nx = jnp.shape(delta_phi)
  v_Fourier =  jnp.fft.fft(delta_phi, axis = 1)  # [nt, nx]
  dl = jnp.pad(1/(dt*dt)*jnp.ones((nt-1,)), (1,0), mode = 'constant', constant_values=0.0).astype(jnp.complex128)
  du = jnp.pad(1/(dt*dt)*jnp.ones((nt-1,)), (0,1), mode = 'constant', constant_values=0.0).astype(jnp.complex128)
  thomas_b = einshape('n->mn', fv - 2/(dt*dt), m = nt)  # [nt, nx]
  
  phi_fouir_part = solver.tridiagonal_solve_batch(dl, thomas_b, du, v_Fourier) #[nt, nx]
  F_phi_updates = jnp.fft.ifft(phi_fouir_part, axis = 1).real #[nt, nx]
  phi_next = phi_prev + F_phi_updates
  return phi_next

@partial(jax.jit, static_argnames=("if_precondition",))
def pdhg_1d_periodic_iter(f_in_H, c_in_H, tau, sigma, m_prev, rho_prev, mu_prev, phi_prev,
                              g, dx, dt, c_on_rho, if_precondition, fv):
  '''
  @ parameters
    f_in_H: [1, nx]
    c_in_H: [1, nx]
    tau: scalar
    sigma: scalar
    m_prev: [nt-1, nx]
    rho_prev: [nt-1, nx]
    mu_prev: [1, nx]
    phi_prev: [nt, nx]
    g: [1, nx]
    dx: scalar
    dt: scalar
    c_on_rho: scalar
    if_precondition: bool
    fv: [nx]

  @ return 
    rho_next: [nt-1, nx]
    phi_next: [nt, nx]
    m_next: [nt-1, nx]
    mu_next: [1, nx]
    err: jnp.array([err1, err2,err3])
  '''

  delta_phi_raw = - tau * (A1TransMult(m_prev, dt, dx) + A2TransMult(rho_prev)) #[nt, nx]
  # old version
  #delta_phi = jnp.concatenate([delta_phi_raw[0:1,:] + tau* mu_prev, delta_phi_raw[1:,:]], axis = 0) #[nt, nx]
  # new version:
  delta_phi_before_scaling = jnp.concatenate([delta_phi_raw[0:1,:] + tau* mu_prev, delta_phi_raw[1:,:]], axis = 0) # [nt, nx]
  delta_phi = delta_phi_before_scaling / dt

  if if_precondition:
    phi_next = update_phi_preconditioning(delta_phi, phi_prev, fv, dt)
  else:
    # no preconditioning
    phi_next = phi_prev - delta_phi

  # extrapolation
  phi_bar = 2 * phi_next - phi_prev
  
  # update mu
  # inf_{mu} sum_i mu_i *(g_i- phi_{1,i}) + |mu - mu^l|^2/(2*sigma)
  mu_next = mu_prev + sigma * (phi_bar[0:1,:] - g)

  rho_candidates = []
  # the previous version: vec1 = m_prev - sigma * A1Mult(phi_bar, dt, dx)  # [nt-1, nx]
  z = m_prev - sigma * A1Mult(phi_bar, dt, dx) / dt  # [nt-1, nx]
  z_left = jnp.roll(z, 1, axis = 1) # [vec1(:,end), vec1(:,1:end-1)]
  # previous version: vec2 = rho_prev - sigma * A2Mult(phi_bar) + sigma * f_in_H * dt # [nt-1, nx]
  alp = rho_prev - sigma * A2Mult(phi_bar) / dt + sigma * f_in_H # [nt-1, nx]

  rho_candidates.append(-c_on_rho * jnp.ones_like(rho_prev))  # left bound
  # two possible quadratic terms on G, 4 combinations
  vec3 = -c_in_H * c_in_H * c_on_rho - z * c_in_H
  vec4 = -c_in_H * c_in_H * c_on_rho + z_left * c_in_H
  rho_candidates.append(jnp.maximum(alp, - c_on_rho))  # for rho large, G = 0
  rho_candidates.append(jnp.maximum((alp + vec3)/(1+ c_in_H*c_in_H), - c_on_rho))#  % if G_i = (rho_i + c)c(xi) + a_i
  rho_candidates.append(jnp.maximum((alp + vec4)/(1+ c_in_H*c_in_H), - c_on_rho))#  % if G_i = (rho_i + c)c(xi) - a_{i-1}
  rho_candidates.append(jnp.maximum((alp + vec3 + vec4)/(1+ 2*c_in_H*c_in_H), - c_on_rho)) # we have both terms above
  
  rho_candidates = jnp.array(rho_candidates) # [5, nt-1, nx]
  rho_next = get_minimizer_ind(rho_candidates, alp, c_on_rho, z, c_in_H)
  # m is truncation of vec1 into [-(rho_i+c)c(xi), (rho_{i+1}+c)c(x_{i+1})]
  m_next = jnp.minimum(jnp.maximum(z, -(rho_next + c_on_rho) * c_in_H), 
                        (jnp.roll(rho_next, -1, axis = 1) + c_on_rho) * jnp.roll(c_in_H, -1, axis = 1))

  # primal error
  err1 = jnp.linalg.norm(phi_next - phi_prev)
  # err2: dual error
  err2_rho = jnp.linalg.norm(rho_next - rho_prev)
  err2_m = jnp.linalg.norm(m_next - m_prev)
  err2_mu = jnp.linalg.norm(mu_next - mu_prev)
  err2 = jnp.sqrt(err2_rho*err2_rho + err2_m*err2_m + err2_mu*err2_mu)
  # err3: equation error
  HJ_residual = check_HJ_sol_usingEO_L1_1d_xdep(phi_next, dt, dx, f_in_H, c_in_H)
  err3 = jnp.mean(jnp.abs(HJ_residual))
  return rho_next, phi_next, m_next, mu_next, jnp.array([err1, err2,err3])


def pdhg_1d_periodic_rho_m_EO_L1_xdep(f_in_H, c_in_H, phi0, rho0, m0, mu0, stepsz_param, 
                                          g, dx, dt, c_on_rho, if_precondition, 
                                          N_maxiter = 1000000, print_freq = 1000, eps = 1e-6):
  '''
  @ parameters:
    f_in_H: [1, nx]
    c_in_H: [1, nx]
    phi0: [nt, nx]
    rho0: [nt-1, nx]
    m0: [nt-1, nx]
    mu0: [1, nx]
    stepsz_param: scalar
    g: [1, nx]
    dx: scalar
    dt: scalar
    c_on_rho: scalar
    if_precondition: bool
    N_maxiter: int
    eps: scalar

  @ return 
    results_all: list of (iter_no, m, rho, mu, phi)
    error_all: [#pdhg iter, 3]
  '''
  nt,nx = jnp.shape(phi0)
  phi_prev = phi0
  rho_prev = rho0
  m_prev = m0
  mu_prev = mu0

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
    rho_next, phi_next, m_next, mu_next, error = pdhg_1d_periodic_iter(f_in_H, c_in_H, tau, sigma, m_prev, rho_prev, mu_prev, phi_prev,
                                                                           g, dx, dt, c_on_rho, if_precondition, fv)
    error_all.append(error)
    if error[2] < eps:
      break
    if jnp.isnan(error[0]) or jnp.isnan(error[1]):
      print("Nan error at iter {}".format(i))
      break
    if print_freq > 0 and i % print_freq == 0:
      results_all.append((i, m_prev, rho_prev, mu_prev, phi_prev))
      print('iteration {}, primal error with prev step {}, dual error with prev step {}, eqt error {}, min rho {}'.format(i, 
                  error[0],  error[1],  error[2], jnp.min(rho_next)), flush = True)
   
    rho_prev = rho_next
    phi_prev = phi_next
    m_prev = m_next
    mu_prev = mu_next
  
  # print the final error
  print('iteration {}, primal error with prev step {}, dual error with prev step {}, eqt error {}'.format(i, error[0],  error[1],  error[2]), flush = True)
  results_all.append((i+1, m_next, rho_next, mu_next, phi_next))
  return results_all, jnp.array(error_all)

def interpolation_tx(mat, scaling_num_t, scaling_num_x, if_include_terminal = False):
  '''
  @ parameters:
    mat: [nt, nx] or [nt-1, nx]
    scaling_num_t, scaling_num_x: integers
    if_include_terminal: bool (True for phi, False for rho, m)
  @ return:
    mat_dense: [nt_dense, nx_dense] or [nt_dense-1, nx_dense]
  '''
  mat_x_right = jnp.concatenate([mat[:,1:], mat[:,0:1]], axis = 1)
  mat_x_dense = interpolation_x(mat, mat_x_right, scaling_num_x)
  if if_include_terminal:
    mat_t_left = mat_x_dense[:-1,:]
    mat_t_right = mat_x_dense[1:,:]
  else:
    mat_t_left = mat_x_dense
    mat_t_right = jnp.concatenate([mat_x_dense[1:,:], mat_x_dense[-1:,:]], axis = 0)
  mat_xt_dense = interpolation_t(mat_t_left, mat_t_right, scaling_num_t)
  if if_include_terminal:
    mat_dense = jnp.concatenate([mat_xt_dense, mat_x_dense[-1:,:]], axis = 0)
  else:
    mat_dense = mat_xt_dense
  return mat_dense


def get_initialization_1d(filename, nt_coarse, nx_coarse, nt_fine, nx_fine):
  '''
  @ parameters:
    filename: string
    nt_coarse, nx_coarse, nt_fine, nx_fine: int
  @ return:
    phi0: [nt_fine, nx_fine]
    rho0: [nt_fine-1, nx_fine]
    m0: [nt_fine-1, nx_fine]
    mu0: [1, nx_fine]
  '''
  with open(filename, 'rb') as f:
    results_np, _ = pickle.load(f)
  results = jax.device_put(results_np)
  phi_coarse = jax.device_put(results[-1][-1])
  mu_coarse = results[-1][-2]
  rho_coarse = results[-1][-3]
  m_coarse = results[-1][-4]
  # interpolation
  scaling_num_x = int(jnp.floor(nx_fine / nx_coarse))
  scaling_num_t = int(jnp.floor((nt_fine -1) / (nt_coarse -1)))
  print("scaling_num_x = {}, scaling_num_t = {}".format(scaling_num_x, scaling_num_t))
  if scaling_num_x * nx_coarse != nx_fine:
    raise ValueError("scaling_num_x should be an integer")
  if scaling_num_t * (nt_coarse -1) != (nt_fine -1):
    raise ValueError("scaling_num_t should be an integer")
  phi0 = interpolation_tx(phi_coarse, scaling_num_t, scaling_num_x, if_include_terminal = True)
  rho0 = interpolation_tx(rho_coarse, scaling_num_t, scaling_num_x, if_include_terminal = False)
  m0 = interpolation_tx(m_coarse, scaling_num_t, scaling_num_x, if_include_terminal = False)
  mu_coarse_right = jnp.concatenate([mu_coarse[:,1:], mu_coarse[:,0:1]], axis = 1)
  mu0 = interpolation_x(mu_coarse, mu_coarse_right, scaling_num_x)
  return phi0, rho0, m0, mu0

  

if __name__ == "__main__":
  nt = 200
  nx = 101
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
  c_in_H_fn = lambda x: 1 + 3* jnp.exp(-4 * (x-1) * (x-1))

  dx = x_period / (nx)
  dt = T / (nt-1)
  x_arr = jnp.linspace(0.0, x_period - dx, num = nx)[None,:]  # [1, nx]
  g = J(x_arr)  # [1, nx]
  f_in_H = f_in_H_fn(x_arr)  # [1, nx]
  c_in_H = c_in_H_fn(x_arr)  # [1, nx]
    
  phi0 = einshape("ij->(ki)j", g, k=nt)  # repeat each row of g to nt times, [nt, nx]
  
  rho0 = jnp.zeros([nt-1, nx])
  m0 = jnp.zeros([nt-1, nx])
  mu0 = jnp.zeros([1, nx])

  #utils.timeit(pdhg_1d_periodic_rho_m_EO_L1_xdep)(...) to get time
  output, error_all = pdhg_1d_periodic_rho_m_EO_L1_xdep(f_in_H, c_in_H, phi0, rho0, m0, mu0, stepsz_param, 
                                          g, dx, dt, c_on_rho, if_precondition, N_maxiter = N_maxiter, print_freq = N_maxiter//5, eps = eps)