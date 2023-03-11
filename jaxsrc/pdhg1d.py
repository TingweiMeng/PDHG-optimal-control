import jax
import jax.numpy as jnp
from functools import partial
import utils
from einshape import jax_einshape as einshape
import os
import solver

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


def get_Gsq_from_rho(rho_plus_c_mul_cinH, a):
  '''
  @parameters:
    rho_plus_c_mul_cinH: [5, nt-1, nx]
    a: [nt-1, nx]
  @return 
    fn_val: [5, nt-1, nx]
  '''
  n_can = jnp.shape(rho_plus_c_mul_cinH)[0]
  a_left = jnp.roll(a, 1, axis = 1)
  a_rep = einshape("ij->kij", a, k=n_can)
  a_left_rep = einshape("ij->kij", a_left, k=n_can)
  G1 = jnp.minimum(rho_plus_c_mul_cinH + a_rep, 0) # when a < 0
  G2 = jnp.minimum(rho_plus_c_mul_cinH - a_left_rep, 0) # when a >=0
  G = jnp.zeros_like(rho_plus_c_mul_cinH)
  G = jnp.where(a_rep < 0, G1, G)
  G = jnp.where(a_left_rep >= 0, G + G2, G)
  return G * G #[n_can, nt-1, nx]


def get_minimizer_ind(rho_candidates, shift_term, c, a, c_in_H):
  '''
  A2_mul_phi is of size ((nt-1)*nx, 1)
  for each (k,i) index, find min_r (r - shift_term)^2 + G(rho)_{k,i}^2 in candidates
  @ parameters:
    rho_candidates: [5, nt-1, nx]
    shift_term: [nt-1, nx]
    c: scalar
    a: [nt-1, nx]
    c_in_H: [1, nx]
  @ return: 
    rho_min: [nt-1, nx]
  '''
  fn_val = (rho_candidates - shift_term[None,:,:])**2 # [5, nt-1, nx]
  fn_val_p = fn_val + get_Gsq_from_rho((rho_candidates + c) * c_in_H[None,:,:], a)
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
  v_Fourier =  jnp.fft.fft(delta_phi, axis = 1) #[nt, nx]
  dl = jnp.pad(1/(dt*dt)*jnp.ones((nt-1,)), (1,0), mode = 'constant', constant_values=0.0).astype(jnp.complex128)
  du = jnp.pad(1/(dt*dt)*jnp.ones((nt-1,)), (0,1), mode = 'constant', constant_values=0.0).astype(jnp.complex128)
  thomas_b = einshape('n->mn', fv - 2/(dt*dt), m = nt) #[nt, nx]
  
  phi_fouir_part = solver.tridiagonal_solve_batch(dl, thomas_b, du, v_Fourier) #[nt, nx]
  F_phi_updates = jnp.fft.ifft(phi_fouir_part, axis = 1).real #[nt, nx]
  phi_next = phi_prev + F_phi_updates
  return phi_next

get_stat = lambda x: jnp.sum(jnp.abs(x))

@partial(jax.jit, static_argnames=("if_precondition",))
def pdhg_onedim_periodic_iter(f_in_H, c_in_H, tau, sigma, m_prev, rho_prev, mu_prev, phi_prev,
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
    c_on_rho: scaler
    if_precondition: bool
    fv: [nx]
    
    Aprecond_func: function = lambda x: Aprecondition @ x

  @ return 
    rho_next: [nt-1, nx]
    phi_next: [nt, nx]
    m_next: [nt-1, nx]
    mu_next: [1, nx]
    err: jnp.array([err1, err2,err3])
  '''

  delta_phi_raw = - tau * (A1TransMult(m_prev, dt, dx) + A2TransMult(rho_prev)) #[nt, nx]
  delta_phi = jnp.concatenate([delta_phi_raw[0:1,:] + tau* mu_prev, delta_phi_raw[1:,:]], axis = 0) #[nt, nx]

  if if_precondition:
    phi_next = update_phi_preconditioning(delta_phi, phi_prev, fv, dt);
  else:
    # no preconditioning
    phi_next = phi_prev - delta_phi

  # extrapolation
  phi_bar = 2 * phi_next - phi_prev
  
  # update mu
  # inf_{mu} sum_i mu_i *(g_i- phi_{1,i}) + |mu - mu^l|^2/(2*sigma)
  mu_next = mu_prev + sigma * (phi_bar[0:1,:] - g)

  rho_candidates = []
  vec1 = m_prev - sigma * A1Mult(phi_bar, dt, dx)  # [nt-1, nx]
  vec1_left = jnp.roll(vec1, 1, axis = 1) # [vec1(:,end), vec1(:,1:end-1)]
  vec2 = rho_prev - sigma * A2Mult(phi_bar) + sigma * f_in_H * dt # [nt-1, nx]
  rho_candidates.append(-c_on_rho * jnp.ones_like(rho_prev))  # left bound

  # two possible quadratic terms on G, 4 combinations
  vec3 = -c_in_H * c_in_H * c_on_rho - vec1 * c_in_H
  vec4 = -c_in_H * c_in_H * c_on_rho + vec1_left * c_in_H
  rho_candidates.append(jnp.maximum(vec2, - c_on_rho))  # for rho large, G = 0
  rho_candidates.append(jnp.maximum((vec2 + vec3)/(1+ c_in_H*c_in_H), - c_on_rho))#  % if G_i = (rho_i + c)c(xi) + a_i
  rho_candidates.append(jnp.maximum((vec2 + vec4)/(1+ c_in_H*c_in_H), - c_on_rho))#  % if G_i = (rho_i + c)c(xi) - a_{i-1}
  rho_candidates.append(jnp.maximum((vec2 + vec3 + vec4)/(1+ 2*c_in_H*c_in_H), - c_on_rho)) # we have both terms above
  
  rho_candidates = jnp.array(rho_candidates) # [5, nt-1, nx]
  rho_next = get_minimizer_ind(rho_candidates, vec2, c_on_rho, vec1, c_in_H)
  # m is truncation of vec1 into [-(rho_i+c)c(xi), (rho_{i+1}+c)c(x_{i+1})]
  m_next = jnp.minimum(jnp.maximum(vec1, -(rho_next + c_on_rho) * c_in_H), 
                        (jnp.roll(rho_next, -1, axis = 1) + c_on_rho) * jnp.roll(c_in_H, -1, axis = 1))

  # primal error
  err1 = jnp.linalg.norm(phi_next - phi_prev)
  # err2: dual error
  err2_rho = jnp.linalg.norm(rho_next - rho_prev)
  err2_m = jnp.linalg.norm(m_next - m_prev)
  err2_mu = jnp.linalg.norm(mu_next - mu_prev)
  err2 = jnp.sqrt(err2_rho*err2_rho + err2_m*err2_m + err2_mu*err2_mu)
  # err3: equation error
  HJ_residual = check_HJ_sol_usingEO_L1_1d_xdep(phi_next, dt, dx, f_in_H, c_in_H);
  err3 = jnp.mean(jnp.abs(HJ_residual))
  
  # jax.debug.print("next {y}, {m}, {n}, {l}", y= get_stat(rho_next), 
  #                 m = get_stat(phi_next), n = get_stat(m_next), 
  #                 l = get_stat(mu_next))
  return rho_next, phi_next, m_next, mu_next, jnp.array([err1, err2,err3])


@utils.timeit
def pdhg_onedim_periodic_rho_m_EO_L1_xdep(f_in_H, c_in_H, phi0, rho0, m0, mu0, stepsz_param, 
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
    c_on_rho: scaler
    if_precondition: bool
    N_maxiter: int
    eps: scalar

  @ return 
    phi: [nt, nx]
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
    tau = stepsz_param / (2*dt/dx + 3)

  sigma = tau
  sigma_scale = 1.5
  sigma = sigma * sigma_scale
  tau = tau / sigma_scale


  if if_precondition:
    # fft for preconditioning
    Lap_vec = jnp.array([-2/(dx*dx), 1/(dx*dx)] + [0.0] * (nx-3) + [1/(dx*dx)])
    fv = jnp.fft.fft(Lap_vec); #[nx]
  else:
    fv = None

  error_all = []
  for i in range(N_maxiter):
    rho_next, phi_next, m_next, mu_next, error = pdhg_onedim_periodic_iter(f_in_H, c_in_H, tau, sigma, m_prev, rho_prev, mu_prev, phi_prev,
                                                                           g, dx, dt, c_on_rho, if_precondition, fv)
    error_all.append(error)
    if error[0] < eps and error[1] < eps:
      break
    if i % print_freq == 0:
      print('iteration {}, primal error with prev step {}, dual error with prev step {}, eqt error {}'.format(i, error[0],  error[1],  error[2]), flush = True)
   
    rho_prev = rho_next
    phi_prev = phi_next
    m_prev = m_next
    mu_prev = mu_next
  return rho_next, jnp.array(error_all)



if __name__ == "__main__":

  for nt, nx in [(200, 101), (200, 201), (400,201), (400, 401), (800,401), (800,801), (1600,801), (1600,1601)]:
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

    print("nt = {}, nx = {}".format(nt, nx))
    print("warm up run, 1000 steps:", flush = True) # for each (nt, nx), the first few iterations would be slower due to jit compiling
    phi_output, error_all = pdhg_onedim_periodic_rho_m_EO_L1_xdep(f_in_H, c_in_H, phi0, rho0, m0, mu0, stepsz_param, 
                                            g, dx, dt, c_on_rho, if_precondition, N_maxiter = 1000, print_freq = 10000, eps = eps)
    print("real run, {} steps:".format(N_maxiter), flush = True)
    phi_output, error_all = pdhg_onedim_periodic_rho_m_EO_L1_xdep(f_in_H, c_in_H, phi0, rho0, m0, mu0, stepsz_param, 
                                            g, dx, dt, c_on_rho, if_precondition, N_maxiter = N_maxiter, print_freq = N_maxiter//5, eps = eps)

