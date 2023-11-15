import jax.numpy as jnp
from absl import app, flags, logging
from solver import set_up_example_fns
import pytz
from datetime import datetime
from pdhg_solver import PDHG_multi_step_inverse
from solver import save
import utils_pdhg

def main(argv):
  for key, value in FLAGS.__flags.items():
    print(value.name, ": ", value._value, flush=True)

  nt = FLAGS.nt
  nx = FLAGS.nx
  ny = FLAGS.ny
  ndim = FLAGS.ndim
  egno = FLAGS.egno
  ifsave = FLAGS.ifsave
  c_on_rho = FLAGS.c_on_rho
  epsl = FLAGS.epsl
  time_step_per_PDHG = FLAGS.time_step_per_PDHG
  eps = FLAGS.eps
  T = FLAGS.T
  fwd = FLAGS.fwd
  sigma_hj = FLAGS.pdhg_step_hj
  sigma_cont = FLAGS.pdhg_step_cont

  print('nx: ', nx)
  print('ny: ', ny)

  N_maxiter = FLAGS.N_maxiter
  print_freq = 10000
  x_period, y_period = 2, 2

  time_stamp = datetime.now(pytz.timezone('America/Los_Angeles')).strftime("%Y%m%d-%H%M%S")
  logging.info("current time: " + datetime.now(pytz.timezone('America/Los_Angeles')).strftime("%Y%m%d-%H%M%S"))
  save_dir = './check_points/{}'.format(time_stamp) + '/eg{}_{}d'.format(egno, ndim)
  if ndim == 1:
    filename_prefix = 'nt{}_nx{}'.format(nt, nx)
  elif ndim == 2:
    filename_prefix = 'nt{}_nx{}_ny{}'.format(nt, nx, ny)

  dx = x_period / (nx)
  dy = y_period / (ny)
  dt = T / (nt-1)
  x_arr = jnp.linspace(0.0, x_period - dx, num = nx)[None,:,None]  # [1, nx, 1]

  if ndim == 1:
    period_spatial = [x_period]
  else:
    period_spatial = [x_period, y_period]
  
  J, fns_dict = set_up_example_fns(egno, ndim, period_spatial)
  
  if ndim == 1:
    x_arr = jnp.linspace(0.0, x_period - dx, num = nx)[None,:,None]  # [1, nx, 1]
  else:
    x_arr = jnp.linspace(0.0, x_period - dx, num = nx)  
    y_arr = jnp.linspace(0.0, x_period - dy, num = ny)
    x_mesh, y_mesh = jnp.meshgrid(x_arr, y_arr, indexing='ij')  # [nx, ny]
    x_arr = jnp.stack([x_mesh, y_mesh], axis = -1)[None,...]  # [1, nx, ny, 2]
  g = J(x_arr)  # [1, nx] or [1, nx, ny]
  print('shape of g: ', g.shape)

  if ndim == 1:
    fn_update_primal = utils_pdhg.update_primal_1d
  else:
    raise NotImplementedError
  fn_update_dual = utils_pdhg.update_dual

  if ndim == 1:
    dspatial = [dx]
    nspatial = [nx]
  else:
    dspatial = [dx, dy]
    nspatial = [nx, ny]
    print('dspatial: ', dspatial)
    print('nspatial: ', nspatial)
  
  results = PDHG_multi_step_inverse(fn_update_primal, fn_update_dual, fns_dict, x_arr, nt, nspatial, ndim,
                    g, dt, dspatial, c_on_rho, time_step_per_PDHG = time_step_per_PDHG,
                    N_maxiter = N_maxiter, print_freq = print_freq, eps = eps,
                    epsl = epsl, fwd = fwd, sigma_hj = sigma_hj, sigma_cont = sigma_cont,
                    hj_precond = FLAGS.hj_precond, cont_precond = FLAGS.cont_precond)
  if ifsave:
    save(save_dir, filename_prefix, results)
  print('phi: ', results[0][-1])





if __name__ == '__main__':
  FLAGS = flags.FLAGS
  flags.DEFINE_integer('nt', 11, 'size of t grids')
  flags.DEFINE_integer('nx', 20, 'size of x grids')
  flags.DEFINE_integer('ny', 20, 'size of y grids')
  flags.DEFINE_integer('ndim', 1, 'spatial dimension')
  flags.DEFINE_integer('egno', 1, 'index of example')
  flags.DEFINE_boolean('ifsave', True, 'if save to pickle')
  flags.DEFINE_float('c_on_rho', 10.0, 'the constant added on rho')
  flags.DEFINE_float('epsl', 0.0, 'diffusion coefficient')
  flags.DEFINE_float('T', 1.0, 'final time')
  flags.DEFINE_integer('time_step_per_PDHG', 2, 'number of time discretization per PDHG iteration')
  flags.DEFINE_integer('N_maxiter', 1000000, 'maximum number of iterations')
  flags.DEFINE_float('eps', 1e-6, 'the error threshold')

  flags.DEFINE_boolean('fwd', False, 'implicit or explicit in HJ')
  flags.DEFINE_boolean('hj_precond', False, 'preconditioning for HJ')
  flags.DEFINE_boolean('cont_precond', False, 'preconditioning for continuity')
  flags.DEFINE_float('pdhg_step_hj', 0.9, 'step size for HJ')
  flags.DEFINE_float('pdhg_step_cont', 0.9, 'step size for continuity')

  app.run(main)
