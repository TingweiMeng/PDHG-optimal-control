import jax
import jax.numpy as jnp
from functools import partial
from einshape import jax_einshape as einshape
jax.config.update("jax_enable_x64", True)

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
    rho_plus_c_mul_cinH: [nt-1, nx, 5]
    a: [nt-1, nx]
  @return 
    fn_val: [nt-1, nx, 5]
  '''
  n_can = jnp.shape(rho_plus_c_mul_cinH)[-1]
  a_left = jnp.roll(a, 1, axis = 1)
  a_rep = einshape("ij->ijk", a, k=n_can)
  a_left_rep = einshape("ij->ijk", a_left, k=n_can)
  G1 = jnp.minimum(rho_plus_c_mul_cinH + a_rep, 0) # when a < 0
  G2 = jnp.minimum(rho_plus_c_mul_cinH - a_left_rep, 0) # when a >=0
  G = jnp.zeros_like(rho_plus_c_mul_cinH)
  G = jnp.where(a_rep < 0, G1, G)
  G = jnp.where(a_left_rep >= 0, G + G2, G)
  return G * G #[nt-1, nx, n_can]


def get_minimizer_ind(rho_candidates, shift_term, c, a, c_in_H):
  '''
  A2_mul_phi is of size ((nt-1)*nx, 1)
  for each (k,i) index, find min_r (r - shift_term)^2 + G(rho)_{k,i}^2 in candidates
  @ parameters:
    rho_candidates: [nt-1, nx, 5]
    shift_term: [nt-1, nx]
    c: scalar
    a: [nt-1, nx]
    c_in_H: [1, nx]
  @ return: 
    rho_min: [nt-1, nx]
  '''
  fn_val = (rho_candidates - shift_term[:,:,None])**2 # [nt-1, nx, 5]
  fn_val_p = fn_val + get_Gsq_from_rho((rho_candidates + c) * c_in_H[:,:,None], a)
  minindex = jnp.argmin(fn_val_p, axis=-1, keepdims=True)
  rho_min = jnp.take_along_axis(rho_candidates, minindex, axis = -1)
  return rho_min[:,:,0]


def pdhg_onedim_periodic_iter(f_in_H, c_in_H, tau, sigma, m_prev, rho_prev, mu_prev, phi_prev,
                              g, dx, dt, c_on_rho, if_precondition, Aprecond_func, nt, nx):
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
    Aprecond_func: function = lambda x: Aprecondition @ x
    nt, nx: scalar

  @ return 
    rho_next: [nt-1, nx]
    phi_next: [nt, nx]
    m_next: [nt-1, nx]
    mu_next: [1, nx]
    err: jnp.array([err1, err2,err3])
  '''

  delta_phi_raw = - tau * (A1TransMult(m_prev) + A2TransMult(rho_prev))
  delta_phi = jnp.concatenate([delta_phi_raw[0:1,:] + tau* mu_prev, delta_phi_raw[1:,:]], axis = 0)

  if if_precondition:
      # % preconditioning: D(phi - phi_prev) = -delta_phi, phi_1 = g
      # % [D; Abd] * phi = [D*phi_prev - delta_phi; g]

    b = [D* phi_prev(:) - delta_phi(:); g(:)]
    # phi_next = jaxopt.linear_solve.solve_bicgstab(Aprecond_func, b)
    phi_next = reshape(phi_next, [nt, nx]);
  else:
    # no preconditioning
    phi_next = phi_prev - delta_phi;

    # extrapolation
    phi_bar = 2 * phi_next - phi_prev
    
    # update mu
    # inf_{mu} sum_i mu_i *(g_i- phi_{1,i}) + |mu - mu^l|^2/(2*sigma)
    mu_next = mu_prev + sigma * (phi_bar[0:1,:] - g)

    rho_candidates = []
    vec1 = m_prev - sigma * A1Mult(phi_bar)  # [nt-1, nx]
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
    
    return rho_next, phi_next, m_next, mu_next, jnp.array([err1, err2,err3])



def pdhg_onedim_periodic_rho_m_EO_L1_xdep(f_in_H, c_in_H, phi0, rho0, m0, mu0, stepsz_param, 
                                          g, dx, dt, c_on_rho, if_precondition):
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

  @ return 
    phi: [nt, nx]
    error_all: [#pdhg iter, 3]
  '''
  N_maxiter = 1e7
  eps = 1e-6
  nt,nx = jnp.shape(phi0)
  phi_prev = phi0
  rho_prev = rho0
  m_prev = m0
  mu_prev = mu0
  pdhg_param = 1
  error_all = jnp.zeros((N_maxiter, 3));

  if if_precondition:
    tau = stepsz_param;
  else:
    tau = stepsz_param / (2*dt/dx + 3);

  sigma = tau
  sigma_scale = 1.5
  sigma = sigma * sigma_scale
  tau = tau / sigma_scale


  # Dx = -A1 / dt;
  # Dt = -A2 / dt;
  # % preconditioning: min <v, phi> + |(Dx; Dt)(phi - phi_prev)|^2/2tau
  # ind_bd1 = 1: nx;
  # ind_bd2 = sub2ind([nt, nx], ones(nx, 1), (1:nx)');  % index of initial condition (1,j)
  # Abd = sparse(ind_bd1, ind_bd2, ones(nx,1), nx, nt*nx);
  # % 0 = tau* v + [Dx;Dt]^T[Dx; Dt](phi - phi_prev)
  # % g = phi_1
  # D = [Dx; Dt]'*[Dx; Dt];
  # Aprecondition = [D; Abd];


  error_all = []
  for i in range(N_maxiter):
    rho_next, phi_next, m_next, mu_next, error = pdhg_onedim_periodic_iter(f_in_H, c_in_H, tau, sigma, m_prev, rho_prev, mu_prev, phi_prev,
                                                                           g, dx, dt, c_on_rho, if_precondition, nt, nx)
    error_all.append(error)
    if error[0] < eps and error[1] < eps:
      break
    if i % 10000 == 0:
      print('iteration {}, primal error with prev step {}, dual error with prev step {}, eqt error {}'.format(i, error[0],  error[1],  error[2]));
   
    rho_prev = rho_next
    phi_prev = phi_next
    m_prev = m_next
    mu_prev = mu_next
  return rho_next, jnp.array(error_all)







if __name__ == "__main__":
  # rho = jnp.ones((3,2))
  # rho_km1 = jnp.pad(rho, ((1,0),(0,0)), mode = 'constant', constant_values=0.0)
  # rho_k = jnp.pad(rho, ((0,1),(0,0)),  mode = 'constant', constant_values=0.0)
  # print(rho_km1)
  # print(rho_k)
  a = jnp.reshape(jnp.arange(8), (2,4))
  b = einshape("ij->(ij)k", a, k = 2)
  print(b)
  pass