import jax
import jax.numpy as jnp
from einshape import jax_einshape as einshape
import utils
from pdhg1d import pdhg_1d_periodic_rho_m_EO_L1_xdep
from pdhg2d import pdhg_2d_periodic_rho_m_EO_L1_xdep
import numpy as np
from absl import app, flags, logging
import pickle


def main(argv):
  for key, value in FLAGS.__flags.items():
    print(value.name, ": ", value._value, flush=True)

  nt = FLAGS.nt
  nx = FLAGS.nx
  ny = FLAGS.ny
  ndim = FLAGS.ndim
  egno = FLAGS.egno
  ifsave = FLAGS.ifsave

  if ndim > 2 or egno > 2:
    raise ValueError("ndim and egno must be 1 or 2")
  
  if_precondition = True
  N_maxiter = 1000000
  rept_num = 10  # repeat running iterations this many times
  eps = 1e-6
  T = 1
  x_period, y_period = 2, 2
  stepsz_param = 0.9
  c_on_rho = 10.0

  if ndim == 1:
    alpha = 2 * jnp.pi / x_period
    filename = './jaxsrc/eg{}_1d/nt{}_nx{}'.format(egno, nt, nx)
  else:
    alpha = jnp.array([2 * jnp.pi / x_period, 2 * jnp.pi / y_period])
    filename = './jaxsrc/eg{}_2d/nt{}_nx{}_ny{}'.format(egno, nt, nx, ny)
  
  J = lambda x: jnp.sum(jnp.sin(alpha * x), axis = -1)  # input [...,ndim] output [...]

  if egno == 1:
    # example 1
    f_in_H_fn = lambda x: jnp.zeros_like(x[...,0])
    c_in_H_fn = lambda x: 1 + 3* jnp.exp(-4 * jnp.sum((x-1) * (x-1), axis = -1))
  else:
    # example 2
    f_in_H_fn = lambda x: 1 + 3* jnp.exp(-4 * jnp.sum((x-1) * (x-1), axis = -1))
    c_in_H_fn = lambda x: jnp.zeros_like(x[...,0]) + 1

  dx = x_period / (nx)
  dy = y_period / (ny)
  dt = T / (nt-1)

  if ndim == 1:
    spatial_arr = jnp.linspace(0.0, x_period - dx, num = nx)[None,:,None]  # [1, nx, 1]
    rho0 = jnp.zeros([nt-1, nx])
    m0 = jnp.zeros([nt-1, nx])
    mu0 = jnp.zeros([1, nx])
  else:
    x_arr = jnp.linspace(0.0, x_period - dx, num = nx)  # [nx]
    y_arr = jnp.linspace(0.0, y_period - dy, num = ny)  # [ny]
    x_arr_2 = einshape("i->kij", x_arr, k=1, j=ny)[...,None]  # [1,nx,ny, 1]
    y_arr_2 = einshape("j->kij", y_arr, k=1, i=nx)[...,None]  # [1,nx,ny, 1]
    spatial_arr = jnp.concatenate([x_arr_2, y_arr_2], axis = -1)  # [1,nx,ny, 2]
    rho0 = jnp.zeros([nt-1, nx, ny])
    m0_1 = jnp.zeros([nt-1, nx, ny])
    m0_2 = jnp.zeros([nt-1, nx, ny])
    mu0 = jnp.zeros([1, nx, ny])

  g = J(spatial_arr)  # [1, nx] or [1, nx, ny]
  f_in_H = f_in_H_fn(spatial_arr)  # [1, nx] or [1, nx, ny]
  c_in_H = c_in_H_fn(spatial_arr)  # [1, nx] or [1, nx, ny]
  
  phi0 = einshape("i...->(ki)...", g, k=nt)  # repeat each row of g to nt times, [nt, nx] or [nt, nx, ny]
  
  print("nt = {}, nx = {}, ny = {}".format(nt, nx, ny))
  print("shape g {}, f {}, c {}".format(jnp.shape(g), jnp.shape(f_in_H), jnp.shape(c_in_H)))

  iter_no = 0
  for i in range(rept_num):
    if ndim == 1:
      results, errors = utils.timeit(pdhg_1d_periodic_rho_m_EO_L1_xdep)(f_in_H, c_in_H, phi0, rho0, m0, mu0, stepsz_param, 
                                          g, dx, dt, c_on_rho, if_precondition, N_maxiter = N_maxiter, print_freq = 10000, eps = eps)
    else:
      results, errors = utils.timeit(pdhg_2d_periodic_rho_m_EO_L1_xdep)(f_in_H, c_in_H, phi0, rho0, m0_1, m0_2, mu0, stepsz_param, 
                    g, dx, dy, dt, c_on_rho, if_precondition, N_maxiter = N_maxiter, print_freq = 10000, eps = eps)
    iter_no += results[-1][0]
    if ifsave:
      with open(filename + '_iter{}.pickle'.format(iter_no), 'wb') as file:
        pickle.dump((results, errors), file)
        print('saved to {}'.format(file), flush = True)
    if results[-1][0] < N_maxiter:
      break
    if ndim == 1:
      m0 = results[-1][1]
    else:
      m0_1 = results[-1][1]
      m0_2 = results[-1][2]
    rho0 = results[-1][-3]
    mu0 = results[-1][-2]
    phi0 = results[-1][-1]


if __name__ == '__main__':
  FLAGS = flags.FLAGS
  flags.DEFINE_integer('nt', 11, 'size of t grids')
  flags.DEFINE_integer('nx', 20, 'size of x grids')
  flags.DEFINE_integer('ny', 20, 'size of y grids')
  flags.DEFINE_integer('ndim', 1, 'dimensionality')
  flags.DEFINE_integer('egno', 1, 'index of example')
  flags.DEFINE_boolean('ifsave', True, 'if save to pickle')
  
  app.run(main)