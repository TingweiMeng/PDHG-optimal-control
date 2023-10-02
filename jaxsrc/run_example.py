import jax.numpy as jnp
from absl import app, flags, logging
from solver import set_up_example_fns
import pytz
from datetime import datetime
from pdhg_solver import PDHG_multi_step
from solver import save
import pdhg1d_m as pdhg1d_m
import pdhg_v as pdhg_v
import pdhg2d_m as pdhg2d_m

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
  c_on_rho = FLAGS.c_on_rho
  epsl = FLAGS.epsl
  time_step_per_PDHG = FLAGS.time_step_per_PDHG
  eps = FLAGS.eps
  T = FLAGS.T

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

  if egno == 2:
    if ndim == 1:
      fn_update_primal = pdhg1d_m.update_primal_1d
      fn_update_dual = pdhg1d_m.update_dual_1d
    else:
      fn_update_primal = pdhg2d_m.update_primal
      fn_update_dual = pdhg2d_m.update_dual
  else:
    if ndim == 1:
      fn_update_primal = pdhg_v.update_primal_1d
    else:
      fn_update_primal = pdhg_v.update_primal_2d
    fn_update_dual = pdhg_v.update_dual

  if ndim == 1:
    dspatial = [dx]
    nspatial = [nx]
  else:
    dspatial = [dx, dy]
    nspatial = [nx, ny]
    print('dspatial: ', dspatial)
    print('nspatial: ', nspatial)
  
  results, errs_none = PDHG_multi_step(fn_update_primal, fn_update_dual, fns_dict, x_arr, nt, nspatial, ndim,
                    g, dt, dspatial, c_on_rho, time_step_per_PDHG = time_step_per_PDHG,
                    N_maxiter = N_maxiter, print_freq = print_freq, eps = eps,
                    epsl = epsl, stepsz_param=stepsz_param)
  if ifsave:
    save(save_dir, filename_prefix, (results, errs_none))
  print('phi: ', results[0][-1])





if __name__ == '__main__':
  FLAGS = flags.FLAGS
  flags.DEFINE_integer('nt', 11, 'size of t grids')
  flags.DEFINE_integer('nx', 20, 'size of x grids')
  flags.DEFINE_integer('ny', 20, 'size of y grids')
  flags.DEFINE_integer('ndim', 1, 'spatial dimension')
  flags.DEFINE_integer('egno', 1, 'index of example')
  flags.DEFINE_boolean('ifsave', True, 'if save to pickle')
  flags.DEFINE_float('stepsz_param', 0.1, 'default step size constant')
  flags.DEFINE_float('c_on_rho', 10.0, 'the constant added on rho')
  flags.DEFINE_float('epsl', 0.0, 'diffusion coefficient')
  flags.DEFINE_float('T', 1.0, 'final time')
  flags.DEFINE_integer('time_step_per_PDHG', 2, 'number of time discretization per PDHG iteration')
  flags.DEFINE_integer('N_maxiter', 1000000, 'maximum number of iterations')

  flags.DEFINE_float('eps', 1e-6, 'the error threshold')
  
  app.run(main)
