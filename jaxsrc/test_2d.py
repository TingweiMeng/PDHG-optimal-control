import jax
import jax.numpy as jnp
from einshape import jax_einshape as einshape
import utils
from pdhg1d import pdhg_1d_periodic_rho_m_EO_L1_xdep, get_initialization_1d
from pdhg2d import pdhg_2d_periodic_rho_m_EO_L1_xdep, get_initialization_2d
import numpy as np
from absl import app, flags, logging
import pickle
from solver import set_up_example


def main(argv):
  for key, value in FLAGS.__flags.items():
    print(value.name, ": ", value._value, flush=True)

  nt = FLAGS.nt
  nx = FLAGS.nx
  ny = FLAGS.ny
  ndim = FLAGS.ndim
  egno = FLAGS.egno
  ifsave = FLAGS.ifsave
  stepsz_param = FLAGS.stepsz_param
  if_precondition = FLAGS.if_precondition
  c_on_rho = FLAGS.c_on_rho
  iterno_init = FLAGS.iterno
  coarse_nx = FLAGS.coarse_nx
  coarse_ny = FLAGS.coarse_ny
  coarse_nt = FLAGS.coarse_nt

  if ndim > 2 or egno > 2:
    raise ValueError("ndim and egno must be 1 or 2")
  
  N_maxiter = 1000000
  rept_num = 10  # repeat running iterations this many times
  eps = 1e-6
  T = 1
  x_period, y_period = 2, 2

  J, f_in_H_fn, c_in_H_fn, filename = set_up_example(egno, ndim, nt, nx, ny, x_period, y_period)

  dx = x_period / (nx)
  dy = y_period / (ny)
  dt = T / (nt-1)
  if ndim == 1:
    spatial_arr = jnp.linspace(0.0, x_period - dx, num = nx)[None,:,None]  # [1, nx, 1]
  else:
    x_arr = jnp.linspace(0.0, x_period - dx, num = nx)  # [nx]
    y_arr = jnp.linspace(0.0, y_period - dy, num = ny)  # [ny]
    x_arr_2 = einshape("i->kij", x_arr, k=1, j=ny)[...,None]  # [1,nx,ny, 1]
    y_arr_2 = einshape("j->kij", y_arr, k=1, i=nx)[...,None]  # [1,nx,ny, 1]
    spatial_arr = jnp.concatenate([x_arr_2, y_arr_2], axis = -1)  # [1,nx,ny, 2]
  g = J(spatial_arr)  # [1, nx] or [1, nx, ny]
  f_in_H = f_in_H_fn(spatial_arr)  # [1, nx] or [1, nx, ny]
  c_in_H = c_in_H_fn(spatial_arr)  # [1, nx] or [1, nx, ny]

  # initialization
  if coarse_nx != 0:
    if ndim == 1:
      initial_filename = './jaxsrc/eg{}_1d/nt{}_nx{}_iter{}.pickle'.format(egno, coarse_nt, coarse_nx, iterno_init)
      phi0, rho0, m0, mu0 = get_initialization_1d(initial_filename, coarse_nt, coarse_nx, nt, nx)
      print("init phi0 shape {}".format(jnp.shape(phi0)))
    else:
      initial_filename = './jaxsrc/eg{}_2d/nt{}_nx{}_ny{}_iter{}.pickle'.format(egno, coarse_nt, coarse_nx, coarse_ny, iterno_init)
      phi0, rho0, m1_0, m2_0, mu0 = get_initialization_2d(initial_filename, coarse_nt, coarse_nx, coarse_ny, nt, nx, ny)
  else:
    phi0 = einshape("i...->(ki)...", g, k=nt)  # repeat each row of g to nt times, [nt, nx] or [nt, nx, ny]
    if ndim == 1:
      rho0 = jnp.zeros([nt-1, nx])
      m0 = jnp.zeros([nt-1, nx])
      mu0 = jnp.zeros([1, nx])
    else:
      rho0 = jnp.zeros([nt-1, nx, ny])
      m1_0 = jnp.zeros([nt-1, nx, ny])
      m2_0 = jnp.zeros([nt-1, nx, ny])
      mu0 = jnp.zeros([1, nx, ny])

  
  print("nt = {}, nx = {}, ny = {}".format(nt, nx, ny))
  print("shape g {}, f {}, c {}".format(jnp.shape(g), jnp.shape(f_in_H), jnp.shape(c_in_H)))

  iter_no = 0
  for i in range(rept_num):
    if ndim == 1:
      results, errors = utils.timeit(pdhg_1d_periodic_rho_m_EO_L1_xdep)(f_in_H, c_in_H, phi0, rho0, m0, mu0, stepsz_param, 
                                          g, dx, dt, c_on_rho, if_precondition, N_maxiter = N_maxiter, print_freq = 10000, eps = eps)
    else:
      results, errors = utils.timeit(pdhg_2d_periodic_rho_m_EO_L1_xdep)(f_in_H, c_in_H, phi0, rho0, m1_0, m2_0, mu0, stepsz_param, 
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
      m1_0 = results[-1][1]
      m2_0 = results[-1][2]
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
  flags.DEFINE_float('stepsz_param', 0.9, 'default step size constant')
  flags.DEFINE_boolean('if_precondition', True, 'if use preconditioning')
  flags.DEFINE_float('c_on_rho', 10.0, 'the constant added on rho')
  flags.DEFINE_integer('iterno', 0, 'iteration number for the initialization filename')
  flags.DEFINE_integer('coarse_nx', 0, 'size of coarse x grids in initialization')
  flags.DEFINE_integer('coarse_ny', 0, 'size of coarse y grids in initialization')
  flags.DEFINE_integer('coarse_nt', 0, 'size of coarse t grids in initialization')
  
  app.run(main)