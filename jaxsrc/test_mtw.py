import jax
import jax.numpy as jnp
from einshape import jax_einshape as einshape
import utils
from pdhg1d import pdhg_onedim_periodic_rho_m_EO_L1_xdep
import numpy as np
from absl import app, flags, logging


def main(argv):

  for key, value in FLAGS.__flags.items():
    print(value.name, ": ", value._value, flush=True)

  nt = FLAGS.nt
  nx = FLAGS.nx
  if_precondition = True
  N_maxiter = 10000000
  eps = 1e-6
  T = 1
  x_period = 2
  stepsz_param = 0.9
  c_on_rho = 10.0
  alpha = 2 * jnp.pi / x_period
  J = lambda x: jnp.sin(alpha * x)
  f_in_H_fn = lambda x: 1 + 3* jnp.exp(-4 * (x-1) * (x-1))
  c_in_H_fn = lambda x: 0*x + 1

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
  results, errors = utils.timeit(pdhg_onedim_periodic_rho_m_EO_L1_xdep)(f_in_H, c_in_H, phi0, rho0, m0, mu0, stepsz_param, 
                                          g, dx, dt, c_on_rho, if_precondition, N_maxiter = N_maxiter, print_freq = 10000, eps = eps)

  np.savez('results_nt{}_nx{}.npz'.format(nt, nx), results = results, errors = errors)
  

if __name__ == '__main__':

  FLAGS = flags.FLAGS
  flags.DEFINE_integer('nt', 100, 'size of t grids')
  flags.DEFINE_integer('nx', 101, 'size of x grids')

  app.run(main)