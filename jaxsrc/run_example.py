import jax
import jax.numpy as jnp
from einshape import jax_einshape as einshape
import utils
import numpy as np
from absl import app, flags, logging
import pickle
from solver import set_up_example_fns
import pytz
from datetime import datetime
import os
import save_analysis
from print_n_plot import get_save_dir, get_sol_on_coarse_grid_1d
from pdhg_solver import PDHG_multi_step
import matplotlib.pyplot as plt


def main(argv):
  import pdhg1d_m_2var
  import pdhg1d_v_2var
  import pdhg1d_m_2var_test

  for key, value in FLAGS.__flags.items():
    print(value.name, ": ", value._value, flush=True)

  nt = FLAGS.nt
  nx = FLAGS.nx
  egno = FLAGS.egno
  ifsave = FLAGS.ifsave
  stepsz_param = FLAGS.stepsz_param
  c_on_rho = FLAGS.c_on_rho
  epsl = FLAGS.epsl
  time_step_per_PDHG = FLAGS.time_step_per_PDHG

  N_maxiter = 10000000
  print_freq = 10000
  eps = 1e-6
  T = 1
  x_period, y_period = 2, 2
  ndim = 1

  time_stamp = datetime.now(pytz.timezone('America/Los_Angeles')).strftime("%Y%m%d-%H%M%S")
  logging.info("current time: " + datetime.now(pytz.timezone('America/Los_Angeles')).strftime("%Y%m%d-%H%M%S"))
  save_dir, filename_prefix = get_save_dir(time_stamp, egno, 1, nt, nx, 0)

  dx = x_period / (nx)
  dt = T / (nt-1)
  x_arr = jnp.linspace(0.0, x_period - dx, num = nx)[None,:,None]  # [1, nx, 1]
  
  J, fns_dict = set_up_example_fns(egno, ndim, x_period, y_period, theoretical_ver=FLAGS.theoretical_scheme)

  if egno < 10:
    fn_update_primal = pdhg1d_m_2var.update_primal_1d
    fn_update_dual = pdhg1d_m_2var.update_dual_1d
    # fn_update_primal = pdhg1d_m_2var_test.update_primal_1d
    # fn_update_dual = pdhg1d_m_2var_test.update_dual_1d
  else:
    fn_update_primal = pdhg1d_v_2var.update_primal_1d
    fn_update_dual = pdhg1d_v_2var.update_dual

  g = J(x_arr)  # [1, nx]

  nspatial = [nx]
  dspatial = [dx]
  results, errs_none = PDHG_multi_step(fn_update_primal, fn_update_dual, fns_dict, x_arr,
                    nt, nspatial, ndim, g, dspatial, dt, c_on_rho, time_step_per_PDHG = time_step_per_PDHG,
                    N_maxiter = N_maxiter, print_freq = print_freq, eps = eps,
                    epsl = epsl, stepsz_param=stepsz_param)
  if ifsave:
    save_analysis.save(save_dir, filename_prefix, (results, errs_none))

  print('phi: ', results[0][-1])





if __name__ == '__main__':
  FLAGS = flags.FLAGS
  flags.DEFINE_integer('nt', 11, 'size of t grids')
  flags.DEFINE_integer('nx', 20, 'size of x grids')
  flags.DEFINE_integer('egno', 11, 'index of example')
  flags.DEFINE_boolean('ifsave', True, 'if save to pickle')
  flags.DEFINE_float('stepsz_param', 0.1, 'default step size constant')
  flags.DEFINE_float('c_on_rho', 10.0, 'the constant added on rho')
  flags.DEFINE_float('epsl', 0.0, 'diffusion coefficient')
  flags.DEFINE_integer('time_step_per_PDHG', 3, 'number of time discretization per PDHG iteration')
  flags.DEFINE_boolean('theoretical_scheme', True, 'true if aligned with theory')
  
  app.run(main)
