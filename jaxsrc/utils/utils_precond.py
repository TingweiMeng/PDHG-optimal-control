from jax import lax
import jax.numpy as jnp
from einshape import jax_einshape as einshape
import jax

'''
bc: 0 for periodic, 1 for neumann, 2 for dirichlet
'''

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

def compute_Dxx_fft_fv(ndim, nspatial, dspatial, bc):
  # fv used in fft for preconditioning (see Poisson_eqt_solver_1d or Poisson_eqt_solver_2d)
  # computes the FFT of Laplacian operator
  if ndim == 1:
    dx = dspatial[0]
    nx = nspatial[0]
    Lap_vec = jnp.array([-2/(dx*dx), 1/(dx*dx)] + [0.0] * (nx-3) + [1/(dx*dx)])
    if bc == 0:
      fv = jnp.fft.fft(Lap_vec)  # [nx]
    elif bc == 1:
      pass
    else:
      raise NotImplementedError
  elif ndim == 2:
    dx, dy = dspatial
    nx, ny = nspatial
    bc_x, bc_y = bc
    Lap_mat = jnp.array([[-2/(dx*dx)-2/(dy*dy), 1/(dy*dy)] + [0.0] * (ny-3) + [1/(dy*dy)],
                        [1/(dx*dx)] + [0.0] * (ny -1)] + [[0.0]* ny] * (nx-3) + \
                        [[1/(dx*dx)] + [0.0] * (ny-1)])  # [nx, ny]
    if bc_x == 0 and bc_y == 0:
      fv = jnp.fft.fft2(Lap_mat)  # [nx, ny]
    elif bc_x == 1 and bc_y == 0:
      pass
    else:
      raise NotImplementedError
  else:
    raise NotImplementedError
  return fv

def H1_precond_1d(source_term, fv, dt, bc, C = 1.0, pow = 1, Ct = 1):
  ''' this solves (C - D_{xx})^pow * phi_update - Ct * D_{tt} phi_update = source_term
  tc is a mixed Neumann-Dirichlet boundary condition (we tested this does not influence much)
  ic is Dirichlet boundary condition zero (this is related to problem nature)
  @parameters:
    source_term: [nt, nx]
    fv: [nx], complex, this is FFT of Laplacian Dxx
    dt: scalar
    bc: 0, 1, 2
    C: postive number (it seems C = 0 and 1 are both fine. Did not test other values)
  @return:
    phi_update: [nt, nx] with ic = 0
  '''
  nt, nx = jnp.shape(source_term)
  # exclude the first row wrt t
  if bc == 0:
    v_Fourier =  jnp.fft.fft(source_term[1:,:], axis = 1)  # [nt-1, nx]
  elif bc == 1:
    pass
  else:
    raise NotImplementedError
  thomas_b = einshape('n->mn', -fv, m = nt-1) + C # [nt-1, nx]
  thomas_b = thomas_b ** pow
  if Ct != 0:
    dl = -jnp.pad(1/(dt*dt)*jnp.ones((nt-2,)), (1,0), mode = 'constant', constant_values=0.0).astype(jnp.complex128) * Ct
    du = -jnp.pad(1/(dt*dt)*jnp.ones((nt-2,)), (0,1), mode = 'constant', constant_values=0.0).astype(jnp.complex128) * Ct
    Lap_t_diag = -jnp.array([-2/(dt*dt)] * (nt-2) + [-1/(dt*dt)])  # [nt-1]  # Neumann tc
    Lap_t_diag_rep = einshape('n->nm', Lap_t_diag, m = nx) * Ct  # [nt-1, nx]
    phi_fouir_part = tridiagonal_solve_batch(dl, thomas_b + Lap_t_diag_rep, du, v_Fourier) # [nt-1, nx]
  else:
    phi_fouir_part = v_Fourier / thomas_b
  if bc == 0:
    F_phi_updates = jnp.fft.ifft(phi_fouir_part, axis = 1).real # [nt-1, nx]
  elif bc == 1:
    pass
  else:
    raise NotImplementedError
  phi_update = jnp.concatenate([jnp.zeros((1,nx)), F_phi_updates], axis = 0) # [nt, nx]
  return phi_update

def H1_precond_2d(source_term, fv, dt, bc, C = 1.0):
  ''' this solves C * phi_update -(D_{tt} + D_{xx} + D_{yy}) phi_update = source_term
  tc is a mixed Neumann-Dirichlet boundary condition (we tested this does not influence much)
  ic is Dirichlet boundary condition zero (this is related to problem nature)
  @parameters:
    source_term: [nt, nx, ny]
    fv: [nx, ny], complex, this is FFT of Laplacian Dxx + Dyy
    dt: scalar
    bc: tuples of 0, 1, 2
    C: postive number (it seems C = 0 and 1 are both fine. Did not test other values)
  @return:
    phi_update: [nt, nx, ny] with ic = 0
  '''
  nt, nx, ny = jnp.shape(source_term)
  bc_x, bc_y = bc
  if bc_x == 0 and bc_y == 0:
    v_Fourier =  jnp.fft.fft2(source_term[1:,...], axes = (1,2)) # [nt-1, nx, ny]
  elif bc_x == 1 and bc_y == 0:
    pass
  else:
    raise NotImplementedError
  dl = -jnp.pad(1/(dt*dt)*jnp.ones((nt-2,)), (1,0), mode = 'constant', constant_values=0.0).astype(jnp.complex128)
  du = -jnp.pad(1/(dt*dt)*jnp.ones((nt-2,)), (0,1), mode = 'constant', constant_values=0.0).astype(jnp.complex128)
  Lap_t_diag = -jnp.array([-2/(dt*dt)] * (nt-2) + [-1/(dt*dt)])  # [nt-1]  # Neumann tc
  Lap_t_diag_rep = einshape('n->nmk', Lap_t_diag, m = nx, k = ny)  # [nt-1, nx, ny]
  thomas_b = einshape('nk->mnk', -fv, m = nt-1) + Lap_t_diag_rep + C # [nt-1, nx, ny]
  phi_fouir_part = tridiagonal_solve_batch_2d(dl, thomas_b, du, v_Fourier)  # [nt-1, nx, ny]
  if bc_x == 0 and bc_y == 0:
    F_phi_updates = jnp.fft.ifft2(phi_fouir_part, axes = (1,2)).real  # [nt-1, nx, ny]
  elif bc_x == 1 and bc_y == 0:
    pass
  else:
    raise NotImplementedError
  phi_update = jnp.concatenate([jnp.zeros((1,nx,ny)), F_phi_updates], axis = 0) # [nt, nx, ny]
  return phi_update