from jax import lax
import jax.numpy as jnp
from functools import partial
from einshape import jax_einshape as einshape
import jax
jax.config.update("jax_enable_x64", True)


def tridiagonal_solve(dl, d, du, b): 
  """Pure JAX implementation of `tridiagonal_solve`.""" 
  prepend_zero = lambda x: jnp.append(jnp.zeros([1], dtype=x.dtype), x[:-1]) 
  fwd1 = lambda tu_, x: x[1] / (x[0] - x[2] * tu_) 
  fwd2 = lambda b_, x: (x[0] - x[3] * b_) / (x[1] - x[3] * x[2]) 
  bwd1 = lambda x_, x: x[0] - x[1] * x_ 
  double = lambda f, args: (f(*args), f(*args)) 

  # Forward pass. 
  _, tu_ = lax.scan(lambda tu_, x: double(fwd1, (tu_, x)), 
                    du[0] / d[0], 
                    (d, du, dl), 
                    unroll=32) 

  _, b_ = lax.scan(lambda b_, x: double(fwd2, (b_, x)), 
                  b[0] / d[0], 
                  (b, d, prepend_zero(tu_), dl), 
                  unroll=32) 

  # Backsubstitution. 
  _, x_ = lax.scan(lambda x_, x: double(bwd1, (x_, x)), 
                  b_[-1], 
                  (b_[::-1], tu_[::-1]), 
                  unroll=32) 

  return x_[::-1] 

# batch in axis 1
tridiagonal_solve_batch = jax.vmap(tridiagonal_solve, in_axes=(None, 1, None, 1), out_axes=1)
tridiagonal_solve_batch_2d = jax.vmap(jax.vmap(tridiagonal_solve, in_axes=(None, -1, None, -1), out_axes=(-1)), 
                            in_axes=(None, -1, None, -1), out_axes=(-1))


@partial(jax.jit, static_argnames=("Neumann_cond",))
def Poisson_eqt_solver(source_term, fv, dt, Neumann_cond = True):
  ''' this solves (D_{tt} + D_{xx}) (u-u_prev) = source_term
  if Neumann_cond is True, we have zero Neumann boundary condition at t=T; otherwise, we have zero Dirichlet boundary condition at t=T
  @parameters:
    source_term: [nt, nx]
    fv: [nx], complex, this is FFT of neg Laplacian -Dxx
    dt: scalar
    Neumann_cond: bool
  @return:
    phi_update: [nt, nx]
  '''
  nt, nx = jnp.shape(source_term)
  # exclude the first row wrt t
  v_Fourier =  jnp.fft.fft(source_term[1:,:], axis = 1)  # [nt-1, nx]
  dl = jnp.pad(1/(dt*dt)*jnp.ones((nt-2,)), (1,0), mode = 'constant', constant_values=0.0).astype(jnp.complex128)
  du = jnp.pad(1/(dt*dt)*jnp.ones((nt-2,)), (0,1), mode = 'constant', constant_values=0.0).astype(jnp.complex128)
  if Neumann_cond:
    Lap_t_diag = jnp.array([-2/(dt*dt)] * (nt-2) + [-1/(dt*dt)])  # [nt-1]
  else:
    Lap_t_diag = jnp.array([-2/(dt*dt)] * (nt-1))  # [nt-1]
  Lap_t_diag_rep = einshape('n->nm', Lap_t_diag, m = nx)  # [nt-1, nx]
  thomas_b = einshape('n->mn', fv, m = nt-1) + Lap_t_diag_rep # [nt-1, nx]
  
  phi_fouir_part = tridiagonal_solve_batch(dl, thomas_b, du, v_Fourier) # [nt-1, nx]
  F_phi_updates = jnp.fft.ifft(phi_fouir_part, axis = 1).real # [nt-1, nx]
  phi_update = jnp.concatenate([jnp.zeros((1,nx)), F_phi_updates], axis = 0) # [nt, nx]
  return phi_update


@partial(jax.jit, static_argnames=("Neumann_cond",))
def pdhg_phi_update(source_term, phi_prev, fv, dt, Neumann_cond = True, reg_param = 0.0, reg_param2 = 0.0, f=0.0):
  ''' this solves (D_{tt} + D_{xx}) (u-u_prev) - reg_param * u -f + reg_param2 * (D_{tt} + D_{xx}) * u = source_term
      i.e., ((1+reg_param2)(D_{tt} + D_{xx}) - reg_param) u = source_term + (D_{tt} + D_{xx}) * u_prev + f,
      where f can be a number or [nt-1, nx] array
  if Neumann_cond is True, we have zero Neumann boundary condition at t=T; otherwise, we have zero Dirichlet boundary condition at t=T
  @parameters:
    source_term: [nt, nx]
    phi_prev: [nt, nx]
    fv: [nx], complex, this is FFT of neg Laplacian -Dxx
    dt: scalar
    Neumann_cond: bool
    reg_param: scalar, regularization parameter
  @return:
    phi_update: [nt, nx]
  '''
  nt, nx = jnp.shape(source_term)
  if jnp.isscalar(f):
    f = jnp.ones((nt-1, nx)) * f
  elif jnp.shape(f) == (1,nx):
    f = jnp.ones((nt-1, nx)) * f
  f_Fourier = jnp.fft.fft(f, axis = 1)  # [nt-1, nx]
  # exclude the first row wrt t
  v_Fourier =  jnp.fft.fft(source_term[1:,:], axis = 1)  # [nt-1, nx]
  dl = (1+reg_param2)*jnp.pad(1/(dt*dt)*jnp.ones((nt-2,)), (1,0), mode = 'constant', constant_values=0.0).astype(jnp.complex128)
  du = (1+reg_param2)*jnp.pad(1/(dt*dt)*jnp.ones((nt-2,)), (0,1), mode = 'constant', constant_values=0.0).astype(jnp.complex128)
  if Neumann_cond:
    Lap_t_diag = jnp.array([-2/(dt*dt)] * (nt-2) + [-1/(dt*dt)])  # [nt-1]
  else:
    Lap_t_diag = jnp.array([-2/(dt*dt)] * (nt-1))  # [nt-1]
  Lap_t_diag_rep = einshape('n->nm', Lap_t_diag, m = nx)  # [nt-1, nx]
  thomas_b = (1+reg_param2) * (einshape('n->mn', fv, m = nt-1) + Lap_t_diag_rep) - reg_param # [nt-1, nx]
  
  phi_prev_Fourier = jnp.fft.fft(phi_prev[1:,:], axis = 1)  # [nt-1, nx]
  rhs = v_Fourier + fv[None,:] * phi_prev_Fourier + Lap_t_diag_rep * phi_prev_Fourier # [nt-1, nx]
  rhs1 = rhs + jnp.pad(phi_prev_Fourier[:-1,:] / (dt*dt), ((1,0), (0,0)))
  rhs2 = rhs1 + jnp.pad(phi_prev_Fourier[1:,:] / (dt*dt), ((0,1), (0,0)))
  phi_fouir_part = tridiagonal_solve_batch(dl, thomas_b, du, rhs2 + f_Fourier) # [nt-1, nx]
  F_phi_next = jnp.fft.ifft(phi_fouir_part, axis = 1).real # [nt-1, nx]
  phi_next = jnp.concatenate([phi_prev[0:1,:], F_phi_next], axis = 0)
  return phi_next

def interpolation_t(fn_left, fn_right, scaling_num):
  '''
  Note: the interpolated matrix does not include the last element of fn_right
  @ parameters:
    fn_left, fn_right: [n, ...]
    scaling_num: int
  @ return:
    fn_dense: [n * scaling_num, ...]
  '''
  fn_dense = einshape('k...->(kj)...', fn_left, j = scaling_num)
  incremental = (fn_right - fn_left) / scaling_num
  for i in range(scaling_num):
    fn_dense = fn_dense.at[i::scaling_num, ...].set(fn_left + i * incremental)
    # fn_dense[i::scaling_num, ...] = fn_left + i * incremental
  return fn_dense

def interpolation_x(fn_left, fn_right, scaling_num):
  '''
  Note: the interpolated matrix does not include the last element of fn_right
  @ parameters:
    fn_left, fn_right: [nt, n, ...]
    scaling_num: int
  @ return:
    fn_dense: [nt, n * scaling_num, ...]
  '''
  fn_dense = einshape('ik->i(jk)', fn_left, j = scaling_num)
  incremental = (fn_right - fn_left) / scaling_num
  for i in range(scaling_num):
    fn_dense = fn_dense.at[:, i::scaling_num, ...].set(fn_left + i * incremental)
    # fn_dense[:, i::scaling_num, ...] = fn_left + i * incremental
  return fn_dense

def interpolation_y(fn_left, fn_right, scaling_num):
  '''
  Note: the interpolated matrix does not include the last element of fn_right
        this function is only used for 2d cases
  @ parameters:
    fn_left, fn_right: [nt, nx, n]
    scaling_num: int
  @ return:
    fn_dense: [nt, nx, n * scaling_num]
  '''
  fn_dense = einshape('ilk->il(kj)', fn_left, j = scaling_num)
  incremental = (fn_right - fn_left) / scaling_num
  for i in range(scaling_num):
    fn_dense = fn_dense.at[..., i::scaling_num].set(fn_left + i * incremental)
    # fn_dense[..., i::scaling_num] = fn_left + i * incremental
  return fn_dense

def solve_HJ_EO_1d(J_on_grids, fn_H_plus, fn_H_minus, nt, dt, dx):
  '''
  @params:
    J_on_grids: [nx]
    fn_H_plus, fn_H_minus: functions taking [nx] and returning [nx]
  @return:
    phi: [nt, nx]
  '''
  phi_curr = J_on_grids
  phi_list = [phi_curr]
  for i in range(nt-1):
    dphidx_left = (phi_curr - jnp.roll(phi_curr, 1, axis=-1))/dx
    dphidx_right = (jnp.roll(phi_curr, -1, axis=-1) - phi_curr)/dx
    phi_next = phi_curr - dt * (fn_H_plus(dphidx_left) + fn_H_minus(dphidx_right))
    phi_list.append(phi_next)
    phi_curr = phi_next
  return jnp.stack(phi_list, axis = 0)


def set_up_example_fns(egno, ndim, x_period, y_period):
  '''
  @ parameters:
    egno, ndim: int
    x_period, y_period: scalars
  @ return:
    J, f_in_H_fn, c_in_H_fn: functions
  '''
  if ndim == 1:
    alpha = 2 * jnp.pi / x_period
  else:
    alpha = jnp.array([2 * jnp.pi / x_period, 2 * jnp.pi / y_period])
  
  if egno == 3:
    J = lambda x: jnp.sum(-(x-1)**2/2 + 2, axis = -1)
  else:
    J = lambda x: jnp.sum(jnp.sin(alpha * x), axis = -1)  # input [...,ndim] output [...]

  if egno == 1:
    # example 1
    f_in_H_fn = lambda x: jnp.zeros_like(x[...,0])
    c_in_H_fn = lambda x: 1 + 3* jnp.exp(-4 * jnp.sum((x-1) * (x-1), axis = -1))
  elif egno == 2:
    # example 2
    f_in_H_fn = lambda x: 1 + 3* jnp.exp(-4 * jnp.sum((x-1) * (x-1), axis = -1))
    c_in_H_fn = lambda x: jnp.zeros_like(x[...,0]) + 1
  elif egno == 0:  # x-indep case
    f_in_H_fn = lambda x: jnp.zeros_like(x[...,0])
    c_in_H_fn = lambda x: jnp.zeros_like(x[...,0]) + 1
  elif egno == 3:  # combine 1 and 2
    # f_in_H_fn = lambda x: 1 + 3* jnp.exp(-4 * jnp.sum((x-1) * (x-1), axis = -1))
    # f_in_H_fn = lambda x: jnp.sum((x-1)**2/2, axis = -1)
    f_in_H_fn = lambda x: jnp.sum(jnp.sin(alpha * x + 0.3), axis = -1) 
    c_in_H_fn = lambda x: 1 + 3* jnp.exp(-4 * jnp.sum((x-1) * (x-1), axis = -1))

  elif egno == 10:  # quad case, no f and c
    f_in_H_fn = lambda x: jnp.zeros_like(x[...,0])
    c_in_H_fn = lambda x: jnp.zeros_like(x[...,0])
  else:
    raise ValueError("egno {} not implemented".format(egno))
  return J, f_in_H_fn, c_in_H_fn



if __name__ == "__main__":
  # n = 20
  # dl = jnp.array([0.0] + [0.1] * (n-1)).astype(jnp.complex128)
  # du = jnp.array([0.1] * (n-1) + [0.0]).astype(jnp.complex128)
  # d = jnp.ones((n,)).astype(jnp.complex128)
  # b = 0.1 * jnp.arange(n).astype(jnp.complex128) 
  # out = tridiagonal_solve(dl, d, du, b)
  # print(out.shape, out)

  # print('=======')
  # bs = 3
  # d = jnp.ones((n, bs)).astype(jnp.complex128)
  # b = jax.random.uniform(jax.random.PRNGKey(1), shape = (n, bs)).astype(jnp.complex128)
  # out = tridiagonal_solve_batch(dl, d, du, b)
  # print(out.shape, out)
  # 

  import matplotlib.pyplot as plt
  import numpy as np
  from print_n_plot import compute_EO_forward_solution_1d

  x_period, y_period = 2, 2
  T = 1
  nx = 101
  nt = 201
  egno = 1
  x_arr = np.linspace(0.0, x_period, num = nx + 1, endpoint = True)  # [nx+1]
  t_arr = np.linspace(0.0, T, num = nt, endpoint = True)  # [nt]

  J, f_in_H_fn, c_in_H_fn = set_up_example_fns(egno, 1, x_period, y_period)

  dx = x_period / (nx)
  dt = T / (nt-1)
  x_input = x_arr[:-1, None]  # [nx, 1]
  g = J(x_input) # [nx]
  f_in_H = f_in_H_fn(x_input)  # [nx]
  c_in_H = c_in_H_fn(x_input)  # [nx]

  t_mesh, x_mesh = np.meshgrid(t_arr, x_arr)
  phi = compute_EO_forward_solution_1d(nt, dx, dt, f_in_H, c_in_H, g)
  print("shape x_arr {}, t_arr {}".format(np.shape(x_arr), np.shape(t_arr)))
  print("shape x_mesh {}, t_mesh {}".format(np.shape(x_mesh), np.shape(t_mesh)))
  print('shape phi {}'.format(np.shape(phi)))
  # plot solution
  phi_trans = einshape('ij->ji', phi)  # [nx, nt]
  phi_np = jax.device_get(jnp.concatenate([phi_trans, phi_trans[0:1,:]], axis = 0))  # [nx+1, nt]
  fig = plt.figure()
  plt.contourf(x_mesh, t_mesh, phi_np)
  plt.colorbar()
  plt.xlabel('x')
  plt.ylabel('t')
  plt.savefig('egno3_solution.png')  
