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

  time_stamp = datetime.now(pytz.timezone('America/Los_Angeles')).strftime("%Y%m%d-%H%M%S")
  logging.info("current time: " + datetime.now(pytz.timezone('America/Los_Angeles')).strftime("%Y%m%d-%H%M%S"))
  save_dir, filename_prefix = get_save_dir(time_stamp, egno, 1, nt, nx, 0)

  dx = x_period / (nx)
  dt = T / (nt-1)
  x_arr = jnp.linspace(0.0, x_period - dx, num = nx)[None,:]  # [1, nx]
  # t_arr = jnp.linspace(dt, T, num = nt-1)[:,None]  # [nt-1, 1]

  # alpha = 2 * jnp.pi / x_period
  # J = lambda x: jnp.sin(alpha * x)  # input [nx] output [nx]
  # f_in_H_fn = lambda p: jnp.zeros_like(p)
  # c_in_H_fn = lambda p: jnp.zeros_like(p) + 1
  if egno == 11 or egno == 12: # v method 
    if egno == 11:
      J = lambda x: 0 * x  
    else:
      J = lambda x: -x**2/10
    f_in_H_fn = lambda x, t: -jnp.minimum(jnp.minimum((x - t - 0.5)**2/2, (x+2 - t - 0.5)**2/2), (x-2 - t - 0.5)**2/2)
    c_in_H_fn = lambda x: jnp.zeros_like(x) + 1

    g = J(x_arr)  # [1, nx]
    c_in_H = c_in_H_fn(x_arr)  # [1, nx]

    fn_update_primal = pdhg1d_v_2var.update_primal_1d
    fn_update_dual = pdhg1d_v_2var.update_dual_1d
    Hstar_minus_fn_general = lambda p, t_arr: jnp.minimum(p, 0.0) **2/ c_in_H/2 - f_in_H_fn(x_arr, t_arr)/2
    Hstar_plus_fn_general = lambda p, t_arr: jnp.maximum(p, 0.0) **2/ c_in_H/2 - f_in_H_fn(x_arr, t_arr)/2
    Hstar_minus_prox_fn = lambda p, t: jnp.minimum(p / (1+t/c_in_H), 0.0)
    Hstar_plus_prox_fn = lambda p, t: jnp.maximum(p / (1+t/c_in_H), 0.0)
    fns_dict = {'Hstar_minus_fn_general': Hstar_minus_fn_general, 'Hstar_plus_fn_general': Hstar_plus_fn_general,
                'Hstar_minus_prox_fn': Hstar_minus_prox_fn, 'Hstar_plus_prox_fn': Hstar_plus_prox_fn}
  else: # m method
    raise NotImplementedError

  H_plus_fn = lambda p, t_arr: c_in_H * jnp.maximum(p,0) **2/2 + f_in_H_fn(x_arr, t_arr)/2
  H_minus_fn = lambda p, t_arr: c_in_H * jnp.minimum(p,0) **2/2 + f_in_H_fn(x_arr, t_arr)/2
  fns_dict['H_plus_fn_general'] = H_plus_fn
  fns_dict['H_minus_fn_general'] = H_minus_fn

  ndim = 1
  results, errs_none = PDHG_multi_step(fn_update_primal, fn_update_dual, fns_dict, nt, nx, ndim,
                    g, dx, dt, c_on_rho, time_step_per_PDHG = time_step_per_PDHG,
                    N_maxiter = N_maxiter, print_freq = print_freq, eps = eps,
                    epsl = epsl, stepsz_param=stepsz_param, dy = 0.0)
  if ifsave:
    save_analysis.save(save_dir, filename_prefix, (results, errs_none))





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
  
  app.run(main)
