from jax import lax
import jax.numpy as jnp
from functools import partial
from einshape import jax_einshape as einshape
import jax
from collections import namedtuple
import os
import pickle
from utils import timer

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


def Poisson_eqt_solver(source_term, fv, dt):
  ''' this solves (D_{tt} + D_{xx}) phi_update = source_term
  if Neumann_tc is True, we have zero Neumann boundary condition at t=T; otherwise, we have zero Dirichlet boundary condition at t=T
  @parameters:
    source_term: [nt, nx]
    fv: [nx], complex, this is FFT of neg Laplacian -Dxx
    dt: scalar
  @return:
    phi_update: [nt, nx]
  '''
  nt, nx = jnp.shape(source_term)
  # exclude the first row wrt t
  v_Fourier =  jnp.fft.fft(source_term[1:,:], axis = 1)  # [nt-1, nx]
  dl = jnp.pad(1/(dt*dt)*jnp.ones((nt-2,)), (1,0), mode = 'constant', constant_values=0.0).astype(jnp.complex128)
  du = jnp.pad(1/(dt*dt)*jnp.ones((nt-2,)), (0,1), mode = 'constant', constant_values=0.0).astype(jnp.complex128)
  Lap_t_diag = jnp.array([-2/(dt*dt)] * (nt-2) + [-1/(dt*dt)])  # [nt-1]  # Neumann tc
  Lap_t_diag_rep = einshape('n->nm', Lap_t_diag, m = nx)  # [nt-1, nx]
  thomas_b = einshape('n->mn', fv, m = nt-1) + Lap_t_diag_rep # [nt-1, nx]  
  phi_fouir_part = tridiagonal_solve_batch(dl, thomas_b, du, v_Fourier) # [nt-1, nx]
  F_phi_updates = jnp.fft.ifft(phi_fouir_part, axis = 1).real # [nt-1, nx]
  phi_update = jnp.concatenate([jnp.zeros((1,nx)), F_phi_updates], axis = 0) # [nt, nx]
  return phi_update

def Poisson_eqt_solver_2d(source_term, fv, dt):
  nt, nx, ny = jnp.shape(source_term)
  v_Fourier =  jnp.fft.fft2(source_term[1:,...], axes = (1,2)) # [nt-1, nx, ny]
  dl = jnp.pad(1/(dt*dt)*jnp.ones((nt-2,)), (1,0), mode = 'constant', constant_values=0.0).astype(jnp.complex128)
  du = jnp.pad(1/(dt*dt)*jnp.ones((nt-2,)), (0,1), mode = 'constant', constant_values=0.0).astype(jnp.complex128)
  Lap_t_diag = jnp.array([-2/(dt*dt)] * (nt-2) + [-1/(dt*dt)])  # [nt-1]  # Neumann tc
  Lap_t_diag_rep = einshape('n->nmk', Lap_t_diag, m = nx, k = ny)  # [nt-1, nx, ny]
  thomas_b = einshape('nk->mnk', fv, m = nt-1) + Lap_t_diag_rep # [nt-1, nx, ny]
  phi_fouir_part = tridiagonal_solve_batch_2d(dl, thomas_b, du, v_Fourier)  # [nt-1, nx, ny]
  F_phi_updates = jnp.fft.ifft2(phi_fouir_part, axes = (1,2)).real  # [nt-1, nx, ny]
  phi_update = jnp.concatenate([jnp.zeros((1,nx,ny)), F_phi_updates], axis = 0) # [nt, nx, ny]
  return phi_update



def set_up_example_fns(egno, ndim, period_spatial):
  '''
  @ parameters:
    egno: int
    ndim: int
    period_spatial: list of length ndim
  @ return:
    J: initial condition, function
    fns_dict: named tuple of functions
  '''
  print('egno: ', egno, flush=True)

  if ndim == 1:
    x_period = period_spatial[0]
    alpha = 2 * jnp.pi / x_period
  else:
    x_period, y_period = period_spatial[0], period_spatial[1]
    alpha = jnp.array([2 * jnp.pi / x_period, 2 * jnp.pi / y_period])
  
  # NOTE: f_in_H_fn and c_in_H_fn are only used in this function and pdhg1d_m.py and pdhg2d_m.py
  if egno == 1:
    J = lambda x: jnp.sum((x - 1)**2/2, axis = -1)
    f_in_H_fn = lambda x, t: jnp.zeros_like(x[...,0])
    c_in_H_fn = lambda x, t: jnp.zeros_like(x[...,0]) + 1
  elif egno == 2:
    J = lambda x: jnp.sum(jnp.sin(alpha * x), axis = -1)  # input [...,ndim] output [...]
    f_in_H_fn = lambda x, t: jnp.zeros_like(x[...,0])
    c_in_H_fn = lambda x, t: jnp.zeros_like(x[...,0]) + 1
  elif egno == 3:
    J = lambda x: jnp.sum(jnp.sin(alpha * x), axis = -1)  # input [...,ndim] output [...]
    f_in_H_fn = lambda x, t: 1 + 3* jnp.exp(-4 * jnp.sum((x-1) * (x-1), axis = -1))
    c_in_H_fn = lambda x, t: jnp.zeros_like(x[...,0]) + 1
  elif egno == 4:
    J = lambda x: -jnp.sum((x-1)**2, axis=-1)/10
    if ndim == 1:
      f_in_H_fn = lambda x, t: -jnp.minimum(jnp.minimum((x[...,0] - t - 0.5)**2/2, (x[...,0]+x_period - t - 0.5)**2/2), 
                                          (x[...,0] -x_period - t - 0.5)**2/2)
    elif ndim == 2:
      f_in_H_fn = lambda x, t: -jnp.minimum(jnp.minimum((x[...,0] - t - 0.5)**2/2, (x[...,0]+x_period - t - 0.5)**2/2), 
                                          (x[...,0] -x_period - t - 0.5)**2/2) - (x[...,1] - 1)**2/4
    else:
      raise ValueError("ndim {} not implemented".format(ndim))
    c_in_H_fn = lambda x, t: jnp.zeros_like(x[...,0]) + 1
  else:
    raise ValueError("egno {} not implemented".format(egno))

  # omit the indicator function
  # note: dim of p is [nt-1, nx]
  # H_plus_fn, H_minus_fn, H_fn are only used in this function and compute_HJ_residual_EO_1d_general, compute_HJ_residual_EO_2d_general, compute_EO_forward_solution_1d_general, compute_EO_forward_solution_2d_general
  # Hstar_plus_fn, Hstar_minus_fn, Hstar_fn are only used in pdhg_v.py
  # Hstar_plus_prox_fn, Hstar_minus_prox_fn Hstar_prox_fn (or H_prox_fn) are only used in pdhg_v.py
  if egno == 2:  # c(x,t)|p|_1 + f(x,t)
    H_plus_fn = lambda p, x_arr, t_arr: c_in_H_fn(x_arr, t_arr) * jnp.maximum(p,0) + f_in_H_fn(x_arr, t_arr)/2/ndim
    H_minus_fn = lambda p, x_arr, t_arr: c_in_H_fn(x_arr, t_arr) * jnp.maximum(-p,0) + f_in_H_fn(x_arr, t_arr)/2/ndim
    Hstar_plus_fn = lambda p, x_arr, t_arr: jnp.zeros_like(p) -f_in_H_fn(x_arr, t_arr)/2/ndim # + indicator(p>=0)
    Hstar_minus_fn = lambda p, x_arr, t_arr: jnp.zeros_like(p) -f_in_H_fn(x_arr, t_arr)/2/ndim # + indicator(p<=0)
    Hstar_plus_prox_fn = lambda p, param, x_arr, t_arr: jnp.minimum(jnp.maximum(p, 0.0), c_in_H_fn(x_arr, t_arr))
    Hstar_minus_prox_fn = lambda p, param, x_arr, t_arr: jnp.maximum(jnp.minimum(p, 0.0), -c_in_H_fn(x_arr, t_arr))
  elif egno == 1 or egno == 4:  # c(x,t)|p|^2/2 + f(x,t)
    H_plus_fn = lambda p, x_arr, t_arr: c_in_H_fn(x_arr, t_arr) * jnp.maximum(p,0) **2/2 + f_in_H_fn(x_arr, t_arr)/2/ndim
    H_minus_fn = lambda p, x_arr, t_arr: c_in_H_fn(x_arr, t_arr) * jnp.minimum(p,0) **2/2 + f_in_H_fn(x_arr, t_arr)/2/ndim
    Hstar_plus_fn = lambda p, x_arr, t_arr: jnp.maximum(p, 0.0) **2/ c_in_H_fn(x_arr, t_arr)/2 - f_in_H_fn(x_arr, t_arr)/2/ndim
    Hstar_minus_fn = lambda p, x_arr, t_arr: jnp.minimum(p, 0.0) **2/ c_in_H_fn(x_arr, t_arr)/2 - f_in_H_fn(x_arr, t_arr)/2/ndim
    Hstar_plus_prox_fn = lambda p, param, x_arr, t_arr: jnp.maximum(p / (1+ param /c_in_H_fn(x_arr, t_arr)), 0.0)
    Hstar_minus_prox_fn = lambda p, param, x_arr, t_arr: jnp.minimum(p / (1+ param /c_in_H_fn(x_arr, t_arr)), 0.0)
  elif egno == 3:  # scheme for |p|_2 + f(x,t), non-seperable case, the indicator function is omitted
    if ndim == 1:
      H_fn = lambda p, x_arr, t_arr: jnp.sqrt(jnp.minimum(p[0],0)**2 + jnp.maximum(p[1],0)**2) + f_in_H_fn(x_arr, t_arr)  # p is [2,...] (xp,xm), x_arr and t_arr can be broadcasted to [...,ndim] and [...]
      Hstar_fn = lambda p, x_arr, t_arr: -f_in_H_fn(x_arr, t_arr)  # p is [2,...] (xp,xm), x_arr and t_arr can be broadcasted to [...]
      def Hstar_prox_fn(p, param, x_arr, t_arr):  # p is [2,...] (xp,xm), others can be broadcast to [...]
        uxp, uxm = p[0], p[1]
        uxp = jnp.minimum(uxp, 0)
        uxm = jnp.maximum(uxm, 0)
        norm = jnp.sqrt(uxp**2 + uxm**2)
        norm = jnp.maximum(norm, 1.0)
        return (uxp/norm, uxm/norm)  # TODO: check this and consistency for 1d case
    elif ndim == 2:
      H_fn = lambda p, x_arr, t_arr: jnp.sqrt(jnp.minimum(p[0],0)**2 + jnp.maximum(p[1],0)**2 + jnp.minimum(p[2],0)**2 \
                            + jnp.maximum(p[3],0)**2) + f_in_H_fn(x_arr, t_arr)  # p is [4,...] (xp,xm,yp,ym), x_arr and t_arr can be broadcasted to [...,ndim] and [...]
      Hstar_fn = lambda p, x_arr, t_arr: -f_in_H_fn(x_arr, t_arr)  # p is [4,...] (xp,xm,yp,ym), x_arr and t_arr can be broadcasted to [...]
      def Hstar_prox_fn(p, param, x_arr, t_arr):  # p is [4,...] (xp,xm,yp,ym), others can be broadcast to [...]
        uxp, uxm, uyp, uym = p[0], p[1], p[2], p[3]
        uxp = jnp.minimum(uxp, 0)
        uxm = jnp.maximum(uxm, 0)
        uyp = jnp.minimum(uyp, 0)
        uym = jnp.maximum(uym, 0)
        norm = jnp.sqrt(uxp**2 + uxm**2 + uyp**2 + uym**2)
        norm = jnp.maximum(norm, 1.0)
        return (uxp/norm, uxm/norm, uyp/norm, uym/norm)  # TODO: check this and consistency for 1d case
  else:
    raise ValueError("egno {} not implemented".format(egno))
  
  if ndim == 1 and egno != 3:  # separable case
    Functions = namedtuple('Functions', ['f_in_H_fn', 'c_in_H_fn', 
                                        'H_plus_fn', 'H_minus_fn', 'Hstar_plus_fn', 'Hstar_minus_fn',
                                        'Hstar_plus_prox_fn', 'Hstar_minus_prox_fn'])
    fns_dict = Functions(f_in_H_fn=f_in_H_fn, c_in_H_fn=c_in_H_fn, 
                        H_plus_fn=H_plus_fn, H_minus_fn=H_minus_fn,
                        Hstar_plus_fn=Hstar_plus_fn, Hstar_minus_fn=Hstar_minus_fn,
                        Hstar_plus_prox_fn=Hstar_plus_prox_fn, Hstar_minus_prox_fn=Hstar_minus_prox_fn)
  elif ndim == 2 and egno != 3:  # separable case
    Functions = namedtuple('Functions', ['f_in_H_fn', 'c_in_H_fn', 
                                        'Hx_plus_fn', 'Hx_minus_fn', 'Hy_plus_fn', 'Hy_minus_fn',
                                        'Hxstar_plus_fn', 'Hxstar_minus_fn', 'Hystar_plus_fn', 'Hystar_minus_fn',
                                        'Hxstar_plus_prox_fn', 'Hxstar_minus_prox_fn', 'Hystar_plus_prox_fn', 'Hystar_minus_prox_fn'])
    fns_dict = Functions(f_in_H_fn=f_in_H_fn, c_in_H_fn=c_in_H_fn, 
                        Hx_plus_fn=H_plus_fn, Hx_minus_fn=H_minus_fn, Hy_plus_fn=H_plus_fn, Hy_minus_fn=H_minus_fn,
                        Hxstar_plus_fn=Hstar_plus_fn, Hxstar_minus_fn=Hstar_minus_fn,
                        Hystar_plus_fn=Hstar_plus_fn, Hystar_minus_fn=Hstar_minus_fn,
                        Hxstar_plus_prox_fn=Hstar_plus_prox_fn, Hxstar_minus_prox_fn=Hstar_minus_prox_fn,
                        Hystar_plus_prox_fn=Hstar_plus_prox_fn, Hystar_minus_prox_fn=Hstar_minus_prox_fn)
  elif egno == 3:  # non-separable case
    Functions = namedtuple('Functions', ['f_in_H_fn', 'c_in_H_fn', 'H_fn', 'Hstar_fn', 'Hstar_prox_fn'])
    fns_dict = Functions(f_in_H_fn=f_in_H_fn, c_in_H_fn=c_in_H_fn, 
                        H_fn=H_fn, Hstar_fn=Hstar_fn, Hstar_prox_fn=Hstar_prox_fn)
  else:
    raise ValueError("ndim {} not implemented".format(ndim))
  return J, fns_dict


def compute_HJ_residual_EO_1d_general(phi, dt, dspatial, fns_dict, epsl, x_arr, t_arr):
  '''
  @parameters:
    phi: [nt, nx]
    dt: scalar
    dspatial: list of scalars
    fns_dict: see set_up_example_fns
    epsl: scalar, diffusion coefficient
    x_arr, t_arr: [1, nx], [nt-1, 1]
  @ return:
    HJ_residual: [nt-1, nx]
  '''
  dx = dspatial[0]
  dphidx_left = (phi - jnp.roll(phi, 1, axis = 1))/dx
  dphidx_right = (jnp.roll(phi, -1, axis=1) - phi)/dx
  if 'H_plus_fn' in fns_dict._fields and 'H_minus_fn' in fns_dict._fields:
    H_plus_fn = fns_dict.H_plus_fn
    H_minus_fn = fns_dict.H_minus_fn
    H_val = H_plus_fn(dphidx_left[1:,:], x_arr, t_arr) + H_minus_fn(dphidx_right[1:,:], x_arr, t_arr)  # [nt-1, nx]
  elif 'H_fn' in fns_dict._fields:
    H_val = fns_dict.H_fn(jnp.stack([dphidx_right, dphidx_left], axis = 0)[:,1:,:], x_arr, t_arr)  # [nt-1, nx]
  Lap = (dphidx_right - dphidx_left)/dx
  HJ_residual = (phi[1:,:] - phi[:-1,:])/dt + H_val - epsl * Lap[1:,:]
  return HJ_residual

def compute_HJ_residual_EO_2d_general(phi, dt, dspatial, fns_dict, epsl, x_arr, t_arr):
  '''
  H is c*|p| + f, 1-dimensional
  @parameters:
    phi: [nt, nx]
    dt: scalar
    dx: scalar
    H_plus, H_minus: functions taking [nt, nx] and returning [nt, nx]
    epsl: scalar, diffusion coefficient
    x_arr, t_arr: [1, nx], [nt-1, 1]
  @ return:
    HJ_residual: [nt-1, nx]
  '''
  dx, dy = dspatial[0], dspatial[1]
  dphidx_left = (phi - jnp.roll(phi, 1, axis = 1))/dx  # [nt, nx, ny]
  dphidx_right = (jnp.roll(phi, -1, axis=1) - phi)/dx  # [nt, nx, ny]
  dphidy_left = (phi - jnp.roll(phi, 1, axis = 2))/dy  # [nt, nx, ny]
  dphidy_right = (jnp.roll(phi, -1, axis=2) - phi)/dy  # [nt, nx, ny]
  if 'Hx_plus_fn' in fns_dict._fields and 'Hy_plus_fn' in fns_dict._fields \
      and 'Hx_minus_fn' in fns_dict._fields and 'Hy_minus_fn' in fns_dict._fields:
    # seperable case
    Hx_plus_fn = fns_dict.Hx_plus_fn
    Hx_minus_fn = fns_dict.Hx_minus_fn
    Hy_plus_fn = fns_dict.Hy_plus_fn
    Hy_minus_fn = fns_dict.Hy_minus_fn
    H_val = Hx_plus_fn(dphidx_left[1:,...], x_arr, t_arr) + Hx_minus_fn(dphidx_right[1:,...], x_arr, t_arr)  # [nt-1, nx, ny]
    H_val += Hy_plus_fn(dphidy_left[1:,...], x_arr, t_arr) + Hy_minus_fn(dphidy_right[1:,...], x_arr, t_arr)  # [nt-1, nx, ny]
  elif 'H_fn' in fns_dict._fields:  # non-seperable case
    dphi = jnp.stack([dphidx_right, dphidx_left, dphidy_right, dphidy_left], axis = 0)  # [4, nt, nx, ny]
    H_val = fns_dict.H_fn(dphi[:,1:,...], x_arr, t_arr)  # [nt-1, nx, ny]
  else:
    raise ValueError("fns_dict must contain Hx_plus_fn, Hx_minus_fn, Hy_plus_fn, Hy_minus_fn or H_fn")
  Lap = (dphidx_right - dphidx_left)/dx + (dphidy_right - dphidy_left)/dy
  HJ_residual = (phi[1:,...] - phi[:-1,...])/dt + H_val - epsl * Lap[1:,...]
  return HJ_residual


def save(save_dir, filename, results):
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  filename_full = save_dir + '/{}.pickle'.format(filename)
  with open(filename_full, 'wb') as file:
    pickle.dump(results, file)
    print('saved to {}'.format(file), flush = True)

def save_raw(filename, results):
  foldername = os.path.dirname(filename)
  if not os.path.exists(foldername):
    os.makedirs(foldername)
  with open(filename, 'wb') as file:
    pickle.dump(results, file)
    print('saved to {}'.format(file), flush = True)

def compute_xarr(ndim, n_spatial, period_spatial):
  '''
    @params:
      ndim: integer
      n_spatial: list of integers, containing nx and maybe ny
      period_spatial: list of floats, containing x_period and maybe y_period
    @return:
      x_arr: [nx, 1] or [nx, ny, 2]
  '''
  if ndim == 1:
    nx = n_spatial[0]
    x_period = period_spatial[0]
    out = jnp.linspace(0.0, x_period, num = nx, endpoint = False)[:,None]  # [nx, 1]
  elif ndim == 2:
    nx, ny = n_spatial[0], n_spatial[1]
    x_period, y_period = period_spatial[0], period_spatial[1]
    x_arr = jnp.linspace(0.0, x_period, num = nx, endpoint = False)  # [nx]
    y_arr = jnp.linspace(0.0, y_period, num = ny, endpoint = False)  # [ny]
    x_mesh, y_mesh = jnp.meshgrid(x_arr, y_arr, indexing='ij')  # [nx, ny]
    out = jnp.stack([x_mesh, y_mesh], axis = -1)  # [nx, ny, 2]
  else:
    raise ValueError("ndim must be 1 or 2")
  return out

def H_L1_true_sol_1d(x, t, fn_J):
  '''
  @params:
    x: [1]
    t: scalar
    fn_J: function taking [...,1] and returning [...]
  @return:
    phi: scalar
  '''
  # phi(x,t) = min_{ |u-x| <= t } J(u)
  n_grids = 101
  x_tests = jnp.linspace(x[0]-t, x[0]+t, num = n_grids)[...,None]  # [n_grids, 1]
  J_tests = fn_J(x_tests)  # [n_grids]
  return jnp.min(J_tests)


def H_L1_true_sol_2d(x, t, fn_J):
  '''
  @params:
    x: [2]
    t: scalar
    fn_J: function taking [...,2] and returning [...]
  @return:
    phi: scalar
  '''
  # phi(x,t) = min_{ |u1-x1| <= t, |u2-x2| <= t } J(u)
  n_grids = 21
  x_tests = jnp.linspace(x[0]-t, x[0]+t, num = n_grids)  # [n_grids]
  y_tests = jnp.linspace(x[0]-t, x[0]+t, num = n_grids)  # [n_grids]
  x_mesh, y_mesh = jnp.meshgrid(x_tests, y_tests, indexing='ij')  # [n_grids, n_grids]
  x_arr2 = jnp.stack([x_mesh, y_mesh], axis = -1)  # [n_grids, n_grids, 2]
  x_arr2 = einshape('ijk->(ij)k', x_arr2)  # [n_grids**2, 2]
  J_tests = fn_J(x_tests)  # [n_grids**2]
  return jnp.min(J_tests)

H_L1_true_sol_1d_batch = partial(jax.jit, static_argnums=(2))(jax.vmap(H_L1_true_sol_1d, in_axes=(0, 0, None), out_axes=0))
H_L1_true_sol_1d_batch2 = jax.vmap(H_L1_true_sol_1d_batch, in_axes=(0, 0, None), out_axes=0)

H_L1_true_sol_2d_batch = partial(jax.jit, static_argnums=(2))(jax.vmap(H_L1_true_sol_1d, in_axes=(0, 0, None), out_axes=0))
H_L1_true_sol_2d_batch2 = jax.vmap(H_L1_true_sol_2d_batch, in_axes=(0, 0, None), out_axes=0)


def compute_EO_forward_solution_1d_general(nt, dt, dspatial, fns_dict, g, x_arr, epsl = 0.0):
  '''
  @parameters:
    nt: integers
    dt: floats
    dspatial: list of scalars
    fns_dict: see set_up_example_fns
    g: [nx]
    x_arr: [nx, 1]
    epsl: float, diffusion coefficient
  @return:
    phi: [nt, nx]
  '''
  if 'H_plus_fn' in fns_dict._fields and 'H_minus_fn' in fns_dict._fields:
    seperable = True
    H_plus_fn = fns_dict.H_plus_fn
    H_minus_fn = fns_dict.H_minus_fn
  elif 'H_fn' in fns_dict._fields:
    seperable = False
    H_fn = fns_dict.H_fn
  else:
    raise ValueError("fns_dict must contain Hx_plus_fn, Hx_minus_fn, Hy_plus_fn, Hy_minus_fn or H_fn")
  def compute_H_val(dphidx_right, dphidx_left, x_arr, t_val):
    if seperable:
      H_val = H_plus_fn(dphidx_left, x_arr, t_val) + H_minus_fn(dphidx_right, x_arr, t_val)
    else:
      dphi = jnp.stack([dphidx_right, dphidx_left], axis = 0)  # [2, nx, ny]
      H_val = H_fn(dphi, x_arr, t_val)  # [nx, ny]
    return H_val
  dx = dspatial[0]
  H_plus_fn = fns_dict.H_plus_fn
  H_minus_fn = fns_dict.H_minus_fn
  phi = []
  phi.append(g)
  for i in range(nt-1):
    dphidx_left = (phi[i] - jnp.roll(phi[i], 1))/dx
    dphidx_right = (jnp.roll(phi[i], -1) - phi[i])/dx
    H_val = compute_H_val(dphidx_right, dphidx_left, x_arr, i*dt)
    diffusion = epsl * (jnp.roll(phi[i], -1) - 2 * phi[i] + jnp.roll(phi[i], 1)) / (dx**2)
    phi_new = phi[i] - dt * H_val + dt * diffusion
    phi.append(phi_new)
    if jnp.any(jnp.isnan(phi_new)):
      break
  phi_arr = jnp.stack(phi, axis = 0)
  return phi_arr
    
def compute_EO_forward_solution_2d_general(nt, dt, dspatial, fns_dict, g, x_arr, epsl = 0.0):
  '''
  @parameters:
    nt: integers
    dx: floats
    dspatial: list of scalars
    fns_dict: see set_up_example_fns
    g: [nx, ny]
    x_arr: [nx, ny, 2]
    epsl: float, diffusion coefficient
  @return:
    phi: [nt, nx, ny]
  '''
  if 'Hx_plus_fn' in fns_dict._fields and 'Hy_plus_fn' in fns_dict._fields and \
    'Hx_minus_fn' in fns_dict._fields and 'Hy_minus_fn' in fns_dict._fields:
    seperable = True
    Hx_plus_fn = fns_dict.Hx_plus_fn
    Hx_minus_fn = fns_dict.Hx_minus_fn
    Hy_plus_fn = fns_dict.Hy_plus_fn
    Hy_minus_fn = fns_dict.Hy_minus_fn
  elif 'H_fn' in fns_dict._fields:
    seperable = False
    H_fn = fns_dict.H_fn
  else:
    raise ValueError("fns_dict must contain Hx_plus_fn, Hx_minus_fn, Hy_plus_fn, Hy_minus_fn or H_fn")
  def compute_H_val(dphidx_right, dphidx_left, dphidy_right, dphidy_left, x_arr, t_val):
    if seperable:
      H_val = Hx_plus_fn(dphidx_left, x_arr, t_val) + Hx_minus_fn(dphidx_right, x_arr, t_val)
      H_val += Hy_plus_fn(dphidy_left, x_arr, t_val) + Hy_minus_fn(dphidy_right, x_arr, t_val)
    else:
      dphi = jnp.stack([dphidx_right, dphidx_left, dphidy_right, dphidy_left], axis = 0)  # [4, nx, ny]
      H_val = H_fn(dphi, x_arr, t_val)  # [nx, ny]
    return H_val
  dx, dy = dspatial[0], dspatial[1]
  phi = []
  phi.append(g)
  for i in range(nt-1):
    dphidx_left = (phi[i] - jnp.roll(phi[i], 1, axis = 0))/dx
    dphidx_right = (jnp.roll(phi[i], -1, axis = 0) - phi[i])/dx
    dphidy_left = (phi[i] - jnp.roll(phi[i], 1, axis = 1))/dy
    dphidy_right = (jnp.roll(phi[i], -1, axis = 1) - phi[i])/dy
    H_val = compute_H_val(dphidx_right, dphidx_left, dphidy_right, dphidy_left, x_arr, i*dt)
    diffusion = epsl * (jnp.roll(phi[i], -1, axis = 0) - 2 * phi[i] + jnp.roll(phi[i], 1, axis = 0)) / (dx**2) \
                + epsl * (jnp.roll(phi[i], -1, axis = 1) - 2 * phi[i] + jnp.roll(phi[i], 1, axis = 1)) / (dy**2)
    phi_new = phi[i] - dt * H_val + dt * diffusion
    phi.append(phi_new)
    if jnp.any(jnp.isnan(phi_new)):
        break
  phi_arr = jnp.stack(phi, axis = 0)
  print("phi dimension {}".format(jnp.shape(phi_arr)))
  return phi_arr

def read_solution(filename):
  with open(filename, 'rb') as f:
    results, errors = pickle.load(f)
  # compute errors
  phi = results[-1][-1]
  return phi

def read_raw_file(filename):
  with open(filename, 'rb') as f:
    result = pickle.load(f)
  return result

def get_sol_on_coarse_grid_1d(sol, coarse_nt, coarse_nx):
  nt, nx = jnp.shape(sol)
  if (nt - 1) % (coarse_nt - 1) != 0 or nx % coarse_nx != 0:
    raise ValueError("nx and nt-1 must be divisible by coarse_nx and coarse_nt-1")
  else:
    nt_factor = (nt-1) // (coarse_nt-1)
    nx_factor = nx // coarse_nx
    sol_coarse = sol[::nt_factor, ::nx_factor]
  return sol_coarse

def get_sol_on_coarse_grid_2d(sol, coarse_nt, coarse_nx, coarse_ny):
  nt, nx, ny = jnp.shape(sol)
  if (nt - 1) % (coarse_nt - 1) != 0 or nx % coarse_nx != 0 or ny % coarse_ny != 0:
    raise ValueError("nx, ny and nt-1 must be divisible by coarse_nx, coarse_ny and coarse_nt-1")
  else:
    nt_factor = (nt-1) // (coarse_nt-1)
    nx_factor = nx // coarse_nx
    ny_factor = ny // coarse_ny
    sol_coarse = sol[::nt_factor, ::nx_factor, ::ny_factor]
  return sol_coarse

def compute_err_1d(phi, true_soln):
  '''
  @parameters:
    phi: [nt, nx]
    true_soln: [nt_dense, nx_dense]
  @return:
    err_l1, err_l1_rel: scalar
    error: [nt, nx]
  '''
  nt, nx = jnp.shape(phi)
  phi_true = get_sol_on_coarse_grid_1d(true_soln, nt, nx)
  error = phi - phi_true
  err_l1 = jnp.mean(jnp.abs(error))
  err_l1_rel = err_l1/ jnp.mean(jnp.abs(phi_true))
  return err_l1, err_l1_rel, error

def compute_err_2d(phi, true_soln):
  '''
  @parameters:
    phi: [nt, nx, ny]
    true_soln: [nt_dense, nx_dense, ny_dense]
  @return:
    err_l1, err_l1_rel: scalar
    error: [nt, nx, ny]
  '''
  nt, nx, ny = jnp.shape(phi)
  phi_true = get_sol_on_coarse_grid_2d(true_soln, nt, nx, ny)
  error = phi - phi_true
  err_l1 = jnp.mean(jnp.abs(error))
  err_l1_rel = err_l1/ jnp.mean(jnp.abs(phi_true))
  return err_l1, err_l1_rel, error


def compute_true_soln_eg1(nt, n_spatial, T, period_spatial, J):
  ''' compute the true solution of HJ PDE with no diffusion and L1 Hamiltonian using LO formula
  '''
  x_arr = compute_xarr(1, n_spatial, period_spatial)
  t_arr = jnp.linspace(0.0, T, num = nt)  # [nt]
  phi_dense_list = []
  for i in range(nt):
    phi_dense_list.append(H_L1_true_sol_1d_batch(x_arr, t_arr[i] + jnp.zeros_like(x_arr[...,0]), J))
  phi_dense = jnp.stack(phi_dense_list, axis = 0)
  return phi_dense

def explicit_EO_general(egno, ndim, nt_dense, n_spatial, period_spatial, T, epsl, trial_num = 10):
  print('compute true soln using general solver')
  if ndim == 1:
    compute_true_soln_fn = compute_EO_forward_solution_1d_general
  else:
    compute_true_soln_fn = compute_EO_forward_solution_2d_general
  J, fns_dict = set_up_example_fns(egno, ndim, period_spatial)
  dt_dense = T / (nt_dense - 1)
  dspatial_dense = [period_spatial[i] / n_spatial[i] for i in range(ndim)]
  x_arr = compute_xarr(ndim, n_spatial, period_spatial)
  g = J(x_arr)  # [nx_dense, ny_dense]
  for j in range(trial_num):
    print('trial {}: nt_dense {}, nspatial_dense {}'.format(j, nt_dense, n_spatial))
    phi_dense = compute_true_soln_fn(nt_dense, dt_dense, dspatial_dense, fns_dict, g, x_arr, epsl=epsl)
    if not jnp.any(jnp.isnan(phi_dense)):
      break
    else:
      print('nan error: nt = {}, nspatial = {}'.format(nt_dense, n_spatial))
    nt_dense = 2 * (nt_dense - 1) + 1
    dt_dense = T / (nt_dense - 1)
  return phi_dense


def compute_ground_truth(egno, nt_dense, n_spatial, ndim, T, period_spatial, epsl = 0.0):
  print('computing ground truth... nt {}, n_spatial {}'.format(nt_dense, n_spatial))
  timer.tic('computing ground truth')
  if egno == 1 and epsl == 0.0 and ndim == 1:
    print('compute true soln using eg1 1d epsl=0.0 solver')
    J, fns_dict = set_up_example_fns(egno, ndim, period_spatial)
    phi_dense = compute_true_soln_eg1(nt_dense, n_spatial, T, period_spatial, J)
  else:
    phi_dense = explicit_EO_general(egno, ndim, nt_dense, n_spatial, period_spatial, T, epsl, trial_num = 10)
  timer.toc('computing ground truth')
  print('finished computing, shape phi_dense {}'.format(jnp.shape(phi_dense)))
  return phi_dense