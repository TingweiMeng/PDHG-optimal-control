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
from print_n_plot import compute_ground_truth, compute_xarr
from pdhg_solver import PDHG_multi_step
import matplotlib.pyplot as plt
import pdhg1d_v_2var

def main(argv):
  for key, value in FLAGS.__flags.items():
    print(value.name, ": ", value._value, flush=True)

  nt = FLAGS.nt
  nx = FLAGS.nx
  ndim = FLAGS.ndim
  egno = FLAGS.egno
  stepsz_param = FLAGS.stepsz_param
  c_on_rho = FLAGS.c_on_rho
  epsl = FLAGS.epsl
  time_step_per_PDHG = FLAGS.time_step_per_PDHG
  theoretical_ver = False
  eps = FLAGS.eps
  if_implicit = FLAGS.if_implicit

  print('nx: ', nx)
  ny = nx

  N_maxiter = 1000000000
  print_freq = 10000
  T = 1
  x_period, y_period = 2, 2

  dx = x_period / (nx)
  dy = y_period / (ny)
  dt = T / (nt-1)
  x_arr = jnp.linspace(0.0, x_period - dx, num = nx)[None,:,None]  # [1, nx, 1]

  period_spatial = [x_period]
  
  J, fns_dict = set_up_example_fns(egno, ndim, period_spatial, theoretical_ver=theoretical_ver)

  x_arr = jnp.linspace(0.0, x_period - dx, num = nx)[None,:,None]  # [1, nx, 1]
  g = J(x_arr)  # [1, nx] or [1, nx, ny]
  print('shape of g: ', g.shape)

  fn_update_primal = pdhg1d_v_2var.update_primal_1d
  fn_update_dual = pdhg1d_v_2var.update_dual

  dspatial = [dx]
  nspatial = [nx]
  
  if if_implicit:
    results, errs_none = PDHG_multi_step(fn_update_primal, fn_update_dual, fns_dict, x_arr, nt, nspatial, ndim,
                    g, dt, dspatial, c_on_rho, time_step_per_PDHG = time_step_per_PDHG,
                    N_maxiter = N_maxiter, print_freq = print_freq, eps = eps,
                    epsl = epsl, stepsz_param=stepsz_param)
  else:
    # use explicit scheme
    nt_dense = nt
    phi_dense = compute_ground_truth(egno, nt_dense, nspatial, ndim, T, period_spatial, epsl = epsl)
    print(phi_dense)

    # compute max of p
    p = (jnp.roll(phi_dense, -1, axis = 1) - phi_dense) / dx
    max_p = jnp.max(p)
    min_p = jnp.min(p)
    print('max_p: ', max_p)
    print('min_p: ', min_p)
  
  




if __name__ == '__main__':
  FLAGS = flags.FLAGS
  flags.DEFINE_integer('nt', 2, 'size of t grids')
  flags.DEFINE_integer('nx', 80, 'size of x grids')
  flags.DEFINE_integer('ndim', 1, 'spatial dimension')
  flags.DEFINE_integer('egno', 12, 'index of example')
  flags.DEFINE_float('stepsz_param', 0.1, 'default step size constant')
  flags.DEFINE_float('c_on_rho', 70.0, 'the constant added on rho')
  flags.DEFINE_float('epsl', 0.1, 'diffusion coefficient')
  flags.DEFINE_integer('time_step_per_PDHG', 2, 'number of time discretization per PDHG iteration')
  flags.DEFINE_boolean('theoretical_scheme', False, 'true if aligned with theory')

  flags.DEFINE_float('eps', 1e-6, 'the error threshold')
  flags.DEFINE_boolean('if_implicit', False, 'true if implicit scheme is used')
  
  app.run(main)
