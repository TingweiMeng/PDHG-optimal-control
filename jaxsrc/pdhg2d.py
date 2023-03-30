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
def A1Mult_x(phi, dt, dx):
  '''A1_x phi = (-phi_{k+1,i+1,j}+phi_{k+1,i,j})*dt/dx
  phi_{k+1,i+1,j} is periodic in i+1
  @ parameters:
    phi: [nt, nx, ny]
  @ return
    out: [nt-1, nx, ny]
  '''
  phi_ip1 = jnp.roll(phi, -1, axis=1)
  out = -phi_ip1 + phi
  out = out[1:,:,:]*dt/dx
  return out

@jax.jit
def A1TransMult_x(m, dt, dx):
  '''A1_x.T m = (-m[k,i-1,j] + m[k,i,j])*dt/dx
  m[k,i-1,j] is periodic in i-1
  prepend 0 in axis-0
  @ parameters:
    m: [nt-1, nx, ny]
  @ return
    out: [nt, nx, ny]
  '''
  m_im1 = jnp.roll(m, 1, axis=1)
  out = -m_im1 + m
  out = out * dt/dx
  out = jnp.pad(out, ((1,0),(0,0),(0,0)), mode = 'constant', constant_values=0.0) #prepend 0
  return out

@jax.jit
def A1Mult_y(phi, dt, dy):
  '''A1_y phi = (-phi_{k+1,i,j+1}+phi_{k+1,i,j})*dt/dy
  phi_{k+1,i,j+1} is periodic in j+1
  @ parameters:
    phi: [nt, nx, ny]
  @ return
    out: [nt-1, nx, ny]
  '''
  phi_jp1 = jnp.roll(phi, -1, axis=2)
  out = -phi_jp1 + phi
  out = out[1:,:,:] * dt/dy
  return out

@jax.jit
def A1TransMult_y(m, dt, dy):
  '''A1_y.T m = (-m[k,i,j-1] + m[k,i,j])*dt/dy
  m[k,i,j-1] is periodic in j-1
  prepend 0 in axis-0
  @ parameters:
    m: [nt-1, nx, ny]
  @ return
    out: [nt, nx, ny]
  '''
  m_jm1 = jnp.roll(m, 1, axis=2)
  out = -m_jm1 + m
  out = out * dt/dy
  out = jnp.pad(out, ((1,0),(0,0),(0,0)), mode = 'constant', constant_values=0.0) #prepend 0
  return out

@jax.jit
def A2Mult(phi):
  '''A2 phi = -phi_{k+1,i,j}+phi_{k,i,j}
  phi_{k+1,i} is not periodic
  @ parameters:
    phi: [nt, nx, ny]
  @ return
    out: [nt-1, nx, ny]
  '''
  phi_kp1 = phi[1:,:,:]
  phi_k = phi[:-1,:,:]
  out = -phi_kp1 + phi_k
  return out

@jax.jit
def A2TransMult(rho):
  '''A2.T rho = (-rho[k-1,i,j] + rho[k,i,j]) #k = 0...(nt-1)
  rho[-1,:] = 0
  @ parameters:
    rho: [nt-1, nx]
  @ return
    out: [nt, nx]
  '''
  rho_km1 = jnp.pad(rho, ((1,0),(0,0),(0,0)), mode = 'constant', constant_values=0.0)
  rho_k = jnp.pad(rho, ((0,1),(0,0),(0,0)),  mode = 'constant', constant_values=0.0)
  out = -rho_km1 + rho_k
  return out

def check_HJ_sol_usingEO_L1_2d_xdep(phi, dt, dx, dy, f_in_H, c_in_H):
  '''
  check a solution is true or not. H is L1, 2-dimensional
  @parameters:
    phi: [nt, nx, ny]
    dt: scalar
    dx, dy: scalar
    f_in_H: [1, nx, ny]
    c_in_H: [1, nx, ny]
  @ return:
    HJ_residual: [nt-1, nx, ny]
  '''
  dphidx_left = (phi - jnp.roll(phi, 1, axis = 1))/dx
  dphidx_right = (jnp.roll(phi, -1, axis=1) - phi)/dx
  dphidy_left = (phi - jnp.roll(phi, 1, axis = 2))/dy
  dphidy_right = (jnp.roll(phi, -1, axis=2) - phi)/dy
  H1_val = jnp.maximum(-dphidx_right, 0) + jnp.maximum(dphidx_left, 0)
  H2_val = jnp.maximum(-dphidy_right, 0) + jnp.maximum(dphidy_left, 0)
  H_val = c_in_H * (H1_val + H2_val) + f_in_H
  HJ_residual = (phi[1:,...] - phi[:-1,...])/dt + H_val[1:,...]
  return HJ_residual

def get_Gsq_from_rho(rho_plus_c_mul_cinH, z1, z2):
  '''
  @parameters:
    rho_plus_c_mul_cinH: [17, nt-1, nx, ny]
    z1, z2: [nt-1, nx, ny]
  @return 
    fn_val: [17, nt-1, nx, ny]
  '''
  n_can = jnp.shape(rho_plus_c_mul_cinH)[0]
  z1_left = jnp.roll(z1, 1, axis = 1)
  z1_rep = einshape("ijl->kijl", z1, k=n_can)
  z1_left_rep = einshape("ijl->kijl", z1_left, k=n_can)
  z2_left = jnp.roll(z2, 1, axis = 2)
  z2_rep = einshape("ijl->kijl", z2, k=n_can)
  z2_left_rep = einshape("ijl->kijl", z2_left, k=n_can)
  G1_1 = jnp.minimum(rho_plus_c_mul_cinH + z1_rep, 0) # when z1 < 0
  G2_1 = jnp.minimum(rho_plus_c_mul_cinH - z1_left_rep, 0) # when z1_left >=0
  G1_2 = jnp.minimum(rho_plus_c_mul_cinH + z2_rep, 0) # when z2 < 0
  G2_2 = jnp.minimum(rho_plus_c_mul_cinH - z2_left_rep, 0) # when z2_left >=0
  G = jnp.zeros_like(rho_plus_c_mul_cinH)
  G = jnp.where(z1_rep < 0, G + G1_1 ** 2, G)
  G = jnp.where(z1_left_rep >= 0, G + G2_1 ** 2, G)
  G = jnp.where(z2_rep < 0, G + G1_2 ** 2, G)
  G = jnp.where(z2_left_rep >= 0, G + G2_2 ** 2, G)
  return G  # [n_can, nt-1, nx, ny]

def get_minimizer_ind(rho_candidates, alp, c, z1, z2, c_in_H):
  '''
  for each (k,i,j) index, find min_r (r - alp)^2 + G(rho)_{k,i,j}^2 in candidates
  @ parameters:
    rho_candidates: [17, nt-1, nx, ny]
    alp: [nt-1, nx, ny]
    c: scalar
    z1, z2: [nt-1, nx, ny]
    c_in_H: [1, nx, ny]
  @ return: 
    rho_min: [nt-1, nx, ny]
  '''
  fn_val = (rho_candidates - alp[None,...])**2 # [17, nt-1, nx, ny]
  fn_val_p = fn_val + get_Gsq_from_rho((rho_candidates + c) * c_in_H[None,...], z1, z2)
  minindex = jnp.argmin(fn_val_p, axis=0, keepdims=True)
  rho_min = jnp.take_along_axis(rho_candidates, minindex, axis = 0)
  return rho_min[0,...]

def update_phi_preconditioning(delta_phi, phi_prev, fv, dt):
  '''
  @parameters:
    delta_phi: [nt, nx, ny]
    phi_prev: [nt, nx, ny]
    fv: [nx, ny], complex
    dt: scalar
  @return:
    phi_next: [nt, nx, ny]
  '''
  nt, nx, ny = jnp.shape(delta_phi)
  v_Fourier =  jnp.fft.fft2(delta_phi, axes = (1,2)) # [nt, nx, ny]
  dl = jnp.pad(1/(dt*dt)*jnp.ones((nt-1,)), (1,0), mode = 'constant', constant_values=0.0).astype(jnp.complex128)
  du = jnp.pad(1/(dt*dt)*jnp.ones((nt-1,)), (0,1), mode = 'constant', constant_values=0.0).astype(jnp.complex128)
  thomas_b = einshape('nk->mnk', fv - 2/(dt*dt), m = nt) # [nt, nx, ny]
  phi_fouir_part = solver.tridiagonal_solve_batch_2d(dl, thomas_b, du, v_Fourier)  # [nt, nx, ny]
  F_phi_updates = jnp.fft.ifft2(phi_fouir_part, axes = (1,2)).real  # [nt, nx, ny]
  phi_next = phi_prev + F_phi_updates
  return phi_next

# @partial(jax.jit, static_argnames=("if_precondition",))
def pdhg_2d_periodic_iter(f_in_H, c_in_H, tau, sigma, m1_prev, m2_prev, rho_prev, mu_prev, phi_prev,
                              g, dx, dy, dt, c_on_rho, if_precondition, fv, mask_candidates):
  '''
  @ parameters
    f_in_H: [1, nx, ny]
    c_in_H: [1, nx, ny]
    tau: scalar
    sigma: scalar
    m1_prev: [nt-1, nx, ny]
    m2_prev: [nt-1, nx, ny]
    rho_prev: [nt-1, nx, ny]
    mu_prev: [1, nx, ny]
    phi_prev: [nt, nx, ny]
    g: [1, nx, ny]
    dx, dy, dt: scalar
    c_on_rho: scalar
    if_precondition: bool
    fv: [nx, ny]
    mask_candidates: [16, 4]  mask for whether z1, z2, z1_left, z2_left are in the set C

  @ return 
    rho_next: [nt-1, nx, ny]
    phi_next: [nt, nx, ny]
    m1_next, m2_next: [nt-1, nx, ny]
    mu_next: [1, nx, ny]
    err: jnp.array([err1, err2,err3])
  '''

  print("tau {}, sigma {}".format(tau, sigma))

  delta_phi_raw = - tau * (A1TransMult_x(m1_prev, dt, dx) + A1TransMult_y(m2_prev, dt, dy) + A2TransMult(rho_prev)) #[nt, nx, ny]
  delta_phi_before_scaling = jnp.concatenate([delta_phi_raw[0:1,:,:] + tau* mu_prev, delta_phi_raw[1:,:,:]], axis = 0) # [nt, nx, ny]
  delta_phi = delta_phi_before_scaling / dt

  print("delta phi {}".format(delta_phi))

  if if_precondition:
    phi_next = update_phi_preconditioning(delta_phi, phi_prev, fv, dt)
  else: # no preconditioning
    phi_next = phi_prev - delta_phi

  print("phi next {}".format(phi_next))

  # extrapolation
  phi_bar = 2 * phi_next - phi_prev

  print("phi bar {}".format(phi_bar))
  
  # update mu
  mu_next = mu_prev + sigma * (phi_bar[0:1,:,:] - g)  # [1, nx, ny]

  print("mu next {}".format(mu_next))

  rho_candidates = []
  z1 = m1_prev - sigma * A1Mult_x(phi_bar, dt, dx) / dt  # [nt-1, nx, ny]
  z2 = m2_prev - sigma * A1Mult_y(phi_bar, dt, dy) / dt  # [nt-1, nx, ny]
  z1_left = jnp.roll(z1, 1, axis = 1)
  z2_left = jnp.roll(z2, 1, axis = 2)
  alp = rho_prev - sigma * A2Mult(phi_bar) / dt + sigma * f_in_H # [nt-1, nx, ny]

  print("z1 {}".format(z1))
  print("z2 {}".format(z2))
  print("alp {}".format(alp))

  rho_candidates_1 = -c_on_rho * jnp.ones_like(rho_prev)[None,...]  # left bound, [1,nt-1, nx,ny]
  # 16 candidates using mask
  num_vec_in_C = jnp.sum(mask_candidates, axis = -1)[:,None,None,None]   # [16, 1, 1,1]
  sum_vec_in_C = (-z1_left)[None,...] * mask_candidates[:,0,None,None,None] \
              + z1[None,...] * mask_candidates[:,1,None,None,None] + z2[None,...] * mask_candidates[:,2,None,None,None] \
              + (-z2_left)[None,...] * mask_candidates[:,3,None,None,None]  # [16, nt-1, nx, ny]
  rho_candidates_16 = (alp[None,...] - num_vec_in_C * c_in_H[None,...] **2 * c_on_rho - c_in_H[None,...] *\
              sum_vec_in_C) / (1 + num_vec_in_C * c_in_H[None,...]**2)
  rho_candidates = jnp.concatenate([rho_candidates_1, rho_candidates_16], axis = 0) # [17, nt-1, nx, ny]  (16 candidates and lower bound)

  print("rho candidates {}".format(rho_candidates))

  rho_next = get_minimizer_ind(rho_candidates, alp, c_on_rho, z1, z2, c_in_H)

  print("rho next {}".format(rho_next))

  m1_next = jnp.minimum(jnp.maximum(z1, -(rho_next + c_on_rho) * c_in_H), 
                        (jnp.roll(rho_next, -1, axis = 1) + c_on_rho) * jnp.roll(c_in_H, -1, axis = 1))
  # m2 is truncation of z2 into [-(rho_{i,j}+c)c(xi,yj), (rho_{i,j+1}+c)c(xi, y_{j+1})]
  m2_next = jnp.minimum(jnp.maximum(z2, -(rho_next + c_on_rho) * c_in_H), 
                        (jnp.roll(rho_next, -1, axis = 2) + c_on_rho) * jnp.roll(c_in_H, -1, axis = 2))
  print("m1 next lb {}".format((jnp.roll(rho_next, -1, axis = 1) + c_on_rho) * jnp.roll(c_in_H, -1, axis = 1)))
  print("m1 next {}".format(m1_next))
  print("m2 next {}".format(m2_next))
  # primal error
  err1 = jnp.linalg.norm(phi_next - phi_prev)
  # err2: dual error
  err2_rho = jnp.linalg.norm(rho_next - rho_prev)
  err2_m1 = jnp.linalg.norm(m1_next - m1_prev)
  err2_m2 = jnp.linalg.norm(m2_next - m2_prev)
  err2_mu = jnp.linalg.norm(mu_next - mu_prev)
  err2 = jnp.sqrt(err2_rho*err2_rho + err2_m1 * err2_m1 + err2_m2 * err2_m2 + err2_mu*err2_mu)
  # err3: equation error
  HJ_residual = check_HJ_sol_usingEO_L1_2d_xdep(phi_next, dt, dx, dy, f_in_H, c_in_H)
  err3 = jnp.mean(jnp.abs(HJ_residual))
  return rho_next, phi_next, m1_next, m2_next, mu_next, jnp.array([err1, err2, err3])


def pdhg_2d_periodic_rho_m_EO_L1_xdep(f_in_H, c_in_H, phi0, rho0, m0_1, m0_2, mu0, stepsz_param, 
                                          g, dx, dy, dt, c_on_rho, if_precondition, 
                                          N_maxiter = 1000000, print_freq = 1000, eps = 1e-6):
  '''
  @ parameters:
    f_in_H: [1, nx, ny]
    c_in_H: [1, nx, ny]
    phi0: [nt, nx, ny]
    rho0: [nt-1, nx, ny]
    m0_1, m0_2: [nt-1, nx, ny]
    mu0: [1, nx, ny]
    stepsz_param: scalar
    g: [1, nx, ny]
    dx, dy: scalar
    dt: scalar
    c_on_rho: scalar
    if_precondition: bool
    N_maxiter: int
    eps: scalar

  @ return 
    results_all: list of (iter_no, m1, m2, rho, mu, phi)
    error_all: [#pdhg iter, 3]
  '''
  nt, nx, ny = jnp.shape(phi0)
  phi_prev = phi0
  rho_prev = rho0
  m1_prev = m0_1
  m2_prev = m0_2
  mu_prev = mu0

  if if_precondition:
    tau = stepsz_param
  else:
    tau = stepsz_param / (2*dt/dx + 2*dt/dy + 3)

  sigma = tau
  sigma_scale = 1.5
  sigma = sigma * sigma_scale
  tau = tau / sigma_scale

  if if_precondition:
    # fft for preconditioning
    Lap_mat = jnp.array([[-2/(dx*dx)-2/(dy*dy), 1/(dy*dy)] + [0.0] * (ny-3) + [1/(dy*dy)],
                        [1/(dx*dx)] + [0.0] * (ny -1)] + [[0.0]* ny] * (nx-3) + \
                        [[1/(dx*dx)] + [0.0] * (ny-1)])
    fv = jnp.fft.fft2(Lap_mat)  # [nx, ny]
  else:
    fv = None

  # mask
  mask = jnp.array([[0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1], [0,1,0,0], [0,1,0,1], [0,1,1,0], [0,1,1,1], 
                    [1,0,0,0], [1,0,0,1], [1,0,1,0], [1,0,1,1], [1,1,0,0], [1,1,0,1], [1,1,1,0], [1,1,1,1]])
  
  error_all = []

  results_all = []
  for i in range(N_maxiter):
    rho_next, phi_next, m1_next, m2_next, mu_next, error = pdhg_2d_periodic_iter(f_in_H, c_in_H, tau, sigma, 
              m1_prev, m2_prev, rho_prev, mu_prev, phi_prev, g, dx, dy, dt, c_on_rho, if_precondition, fv, mask)
    error_all.append(error)
    if error[2] < eps:
      break
    if jnp.isnan(error[0]) or jnp.isnan(error[1]):
      print("Nan error at iter {}".format(i))
      break
    if print_freq > 0 and i % print_freq == 0:
      results_all.append((i, m1_prev, m2_prev, rho_prev, mu_prev, phi_prev))
      print('iteration {}, primal error with prev step {}, dual error with prev step {}, eqt error {}'.format(i, error[0],  error[1],  error[2]), flush = True)
   
    rho_prev = rho_next
    phi_prev = phi_next
    m1_prev = m1_next
    m2_prev = m2_next
    mu_prev = mu_next
  
  # print the final error
  print('iteration {}, primal error with prev step {}, dual error with prev step {}, eqt error {}'.format(i, error[0],  error[1],  error[2]), flush = True)
  results_all.append((i+1, m1_next, m2_next, rho_next, mu_next, phi_next))
  return results_all, jnp.array(error_all)


if __name__ == "__main__":
  nt = 3
  nx = 4
  ny = 3
  if_precondition = False # True
  N_maxiter = 2 # 2000001
  eps = 1e-6
  T = 1
  x_period = 2
  y_period = 2
  stepsz_param = 0.9
  c_on_rho = 10.0
  alpha_x = 2 * jnp.pi / x_period
  alpha_y = 2 * jnp.pi / y_period
  J = lambda x, y: jnp.sin(alpha_x * x + alpha_y * y)
  f_in_H_fn = lambda x, y: 0*x
  c_in_H_fn = lambda x, y: 0*x + 3* jnp.exp(-4 * (x-1) * (x-1) -4 * (y-1) * (y-1))

  dx = x_period / (nx)
  dy = y_period / (ny)
  dt = T / (nt-1)

  print("nt {}, nx {}, ny {}, dt {}, dx {}, dy {}".format(nt, nx, ny, dt, dx, dy))
  
  x_arr = jnp.linspace(0.0, x_period - dx, num = nx)  # [nx]
  y_arr = jnp.linspace(0.0, y_period - dy, num = ny)  # [ny]
  x_arr_2 = einshape("i->kij", x_arr, k=1, j=ny)  # [1,nx,ny]
  y_arr_2 = einshape("j->kij", y_arr, k=1, i=nx)  # [1,nx,ny]
  g = J(x_arr_2, y_arr_2)  # [1, nx, ny]
  f_in_H = f_in_H_fn(x_arr_2, y_arr_2)  # [1, nx, ny]
  c_in_H = c_in_H_fn(x_arr_2, y_arr_2)  # [1, nx, ny]

  phi0 = einshape("ijl->(ki)jl", g, k=nt)  # repeat each row of g to nt times, [nt, nx, ny]
  
  rho0 = jnp.zeros([nt-1, nx, ny])
  m0_1 = jnp.zeros([nt-1, nx, ny])
  m0_2 = jnp.zeros([nt-1, nx, ny])
  mu0 = jnp.zeros([1, nx, ny])

  # FOR DEBUG!
  key = jax.random.PRNGKey(1)
  key, subkey = jax.random.split(key)
  phi0 = jax.random.normal(subkey, [nt, nx, ny])
  key, subkey = jax.random.split(key)
  g = jax.random.normal(subkey, [1, nx, ny])
  key, subkey = jax.random.split(key)
  f_in_H = jax.random.normal(subkey, [1, nx, ny])
  key, subkey = jax.random.split(key)
  c_in_H = jnp.abs(jax.random.normal(subkey, [1, nx, ny]))
  key, subkey = jax.random.split(key)
  rho0 = jax.random.normal(subkey, [nt-1, nx, ny])
  key, subkey = jax.random.split(key)
  m0_1 = jax.random.normal(subkey, [nt-1, nx, ny])
  key, subkey = jax.random.split(key)
  m0_2 = jax.random.normal(subkey, [nt-1, nx, ny])
  key, subkey = jax.random.split(key)
  mu0 = jax.random.normal(subkey, [1, nx, ny])

  print("g {}".format(g))
  print("f {}".format(f_in_H))
  print("c {}".format(c_in_H))
  print("phi0 {}".format(phi0))
  print("rho0 {}".format(rho0))
  print("m0 1 {}".format(m0_1))
  print("m0 2 {}".format(m0_2))
  print("mu0 {}".format(mu0))

  output, error_all = utils.timeit(pdhg_2d_periodic_rho_m_EO_L1_xdep)(f_in_H, c_in_H, phi0, rho0, m0_1, m0_2, mu0, stepsz_param, 
                    g, dx, dy, dt, c_on_rho, if_precondition, N_maxiter = N_maxiter, print_freq = N_maxiter//100, eps = eps)