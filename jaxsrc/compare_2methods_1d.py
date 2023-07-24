import jax
import jax.numpy as jnp
from einshape import jax_einshape as einshape
import utils
from pdhg1d_m_3var import pdhg_1d_periodic_rho_m_EO_L1_xdep as pdhg_method1_3var
from pdhg1d_m_2var import pdhg_1d_periodic_rho_m_EO_L1_xdep as pdhg_method1_2var
from pdhg1d_v_2var import pdhg_1d_periodic_rho_m_EO_L1_xdep as pdhg_method2_2var
import numpy as np
from absl import app, flags, logging
import pickle
from solver import set_up_example_fns
import pytz
from datetime import datetime
import os
import save_analysis
from print_n_plot import get_save_dir, get_sol_on_coarse_grid_1d, compute_ground_truth


def method1_3var(rept_num, eps, epsl, N_maxiter, ifsave, if_precondition, c_on_rho, stepsz_param, 
                    f_in_H, c_in_H, phi0, rho0, m0, mu0, g, dx, dt, save_dir, filename_prefix):
  save_dir = save_dir + '/method1_3var'
  utils.timer.tic('method 1 with 3 var')
  iter_no = 0
  for i in range(rept_num):
    results, errors = pdhg_method1_3var(f_in_H, c_in_H, phi0, rho0, m0, mu0, stepsz_param, 
                                        g, dx, dt, c_on_rho, if_precondition, N_maxiter = N_maxiter, print_freq = 10000, eps = eps,
                                        epsl = epsl)
    iter_no += results[-1][0]
    if ifsave:
      filename = filename_prefix + '_iter{}'.format(iter_no)
      save_analysis.save(save_dir, filename, (results, errors))
    if results[-1][0] < N_maxiter:
      break
    m0 = results[-1][1]
    rho0 = results[-1][-3]
    mu0 = results[-1][-2]
    phi0 = results[-1][-1]
  utils.timer.toc('method 1 with 3 var')

def method1_2var(rept_num, eps, epsl, N_maxiter, ifsave, if_precondition, c_on_rho, stepsz_param, 
                    f_in_H, c_in_H, phi0, rho0, m0, g, dx, dt, save_dir, filename_prefix):
  save_dir = save_dir + '/method1_2var'
  utils.timer.tic('method 1 with 2 var')
  iter_no = 0
  for i in range(rept_num):
    results, errors = pdhg_method1_2var(f_in_H, c_in_H, phi0, rho0, m0, stepsz_param, 
                                        g, dx, dt, c_on_rho, if_precondition, N_maxiter = N_maxiter, print_freq = 10000, eps = eps,
                                        epsl = epsl)
    iter_no += results[-1][0]
    if ifsave:
      filename = filename_prefix + '_iter{}'.format(iter_no)
      save_analysis.save(save_dir, filename, (results, errors))
    if results[-1][0] < N_maxiter:
      break
    m0 = results[-1][1]
    rho0 = results[-1][-3]
    phi0 = results[-1][-1]
  utils.timer.toc('method 1 with 2 var')
  m = results[-1][1]
  rho = results[-1][-3]
  phi = results[-1][-1]
  return phi, rho, m


def method2_2var(v_method, rho_method, updating_rho_first,
                  rept_num, eps, epsl, N_maxiter, ifsave, if_precondition, c_on_rho, stepsz_param, 
                  f_in_H, c_in_H, phi0, rho0, v0, g, dx, dt, save_dir, filename_prefix):
  save_dir = save_dir + '/method2_2var'
  utils.timer.tic('method 2 with 2 var')
  iter_no = 0
  for i in range(rept_num):
    results, errors = pdhg_method2_2var(v_method, rho_method, updating_rho_first,
                                        f_in_H, c_in_H, phi0, rho0, v0, stepsz_param, 
                                        g, dx, dt, c_on_rho, if_precondition, N_maxiter = N_maxiter, 
                                        print_freq = 10, eps = eps, epsl = epsl)
    iter_no += results[-1][0]
    if ifsave:
      filename = filename_prefix + '_iter{}'.format(iter_no)
      save_analysis.save(save_dir, filename, (results, errors))
    if results[-1][0] < N_maxiter:
      break
    v0 = results[-1][1]
    rho0 = results[-1][-3]
    phi0 = results[-1][-1]
  utils.timer.toc('method 2 with 2 var')
  

def main(argv):
  for key, value in FLAGS.__flags.items():
    print(value.name, ": ", value._value, flush=True)

  nt = FLAGS.nt
  nx = FLAGS.nx
  egno = FLAGS.egno
  ifsave = FLAGS.ifsave
  stepsz_param = FLAGS.stepsz_param
  if_precondition = FLAGS.if_precondition
  c_on_rho = FLAGS.c_on_rho
  epsl = FLAGS.epsl

  N_maxiter = 1000 # 1000000
  rept_num = 1 # 10  # repeat running iterations this many times
  eps = 1e-6
  T = 1
  x_period, y_period = 2, 2

  J, f_in_H_fn, c_in_H_fn = set_up_example_fns(egno, 1, x_period, y_period)

  dx = x_period / (nx)
  dt = T / (nt-1)
  spatial_arr = jnp.linspace(0.0, x_period - dx, num = nx)[None,:,None]  # [1, nx, 1]
  g = J(spatial_arr)  # [1, nx]
  f_in_H = f_in_H_fn(spatial_arr)  # [1, nx]
  c_in_H = c_in_H_fn(spatial_arr)  # [1, nx]

  phi0 = einshape("i...->(ki)...", g, k=nt)  # repeat each row of g to nt times, [nt, nx] or [nt, nx, ny]
  rho0 = jnp.zeros([nt-1, nx])
  m0 = jnp.zeros([nt-1, nx])
  mu0 = jnp.zeros([1, nx])
  vp0 = jnp.zeros([nt-1, nx])
  vm0 = jnp.zeros([nt-1, nx])
  v0 = [vp0, vm0]

  # put true solution for testing
  phi_dense = compute_ground_truth(egno, 1, T, x_period, y_period)
  phi0 = get_sol_on_coarse_grid_1d(phi_dense, nt, nx)
  # vp0 = jnp.zeros([nt-1, nx])
  # vm0 = jnp.zeros([nt-1, nx])
  v = (jnp.roll(phi0, -1, axis=1) - jnp.roll(phi0, 1, axis=1)) / (2 * dx)
  v = v[1:,:]
  vm0 = jnp.maximum(v, 0)
  vp0 = jnp.minimum(v, 0)
  v0 = [vp0, vm0]

  time_stamp = datetime.now(pytz.timezone('America/Los_Angeles')).strftime("%Y%m%d-%H%M%S")
  logging.info("current time: " + datetime.now(pytz.timezone('America/Los_Angeles')).strftime("%Y%m%d-%H%M%S"))

  save_dir, filename_prefix = get_save_dir(time_stamp, egno, 1, nt, nx, 0)
  
  print("nt = {}, nx = {}".format(nt, nx))
  print("shape g {}, f {}, c {}".format(jnp.shape(g), jnp.shape(f_in_H), jnp.shape(c_in_H)))

  # # use results from method 1 as initialization for method 2
  # # method 1 with 2 vars
  # print('===================== method 1 with 2 vars =====================')
  # N_maxiter = 100000
  # phi0, rho0, m0 = method1_2var(rept_num, eps, epsl, N_maxiter, ifsave, if_precondition, c_on_rho, stepsz_param,
  #               f_in_H, c_in_H, phi0, rho0, m0, g, dx, dt, save_dir, filename_prefix)
  
  # # method 2 with 2 vars
  # vp0 = jnp.minimum(m0, 0) / (rho0 + c_on_rho + eps)
  # vm0 = jnp.maximum(m0, 0) / (rho0 + c_on_rho + eps)
  # vm0 = jnp.roll(vm0, -1, axis=1)
  # v0 = [vp0, vm0]
  N_maxiter = 10000
  v_method, rho_method, updating_rho_first = FLAGS.v_method, FLAGS.rho_method, FLAGS.updating_rho_first
  print('===================== method 2 with 2 vars =====================')
  method2_2var(v_method, rho_method, updating_rho_first,
                rept_num, eps, epsl, N_maxiter, ifsave, if_precondition, c_on_rho, stepsz_param,
                f_in_H, c_in_H, phi0, rho0, v0, g, dx, dt, save_dir, filename_prefix)


  
  # # method 1 with 3 vars
  # print('===================== method 1 with 3 vars =====================')
  # method1_3var(rept_num, eps, epsl, N_maxiter, ifsave, if_precondition, c_on_rho, stepsz_param,
  #               f_in_H, c_in_H, phi0, rho0, m0, mu0, g, dx, dt, save_dir, filename_prefix)
  
  

  


if __name__ == '__main__':
  FLAGS = flags.FLAGS
  flags.DEFINE_integer('nt', 11, 'size of t grids')
  flags.DEFINE_integer('nx', 20, 'size of x grids')
  flags.DEFINE_integer('egno', 1, 'index of example')
  flags.DEFINE_boolean('ifsave', True, 'if save to pickle')
  flags.DEFINE_float('stepsz_param', 0.9, 'default step size constant')
  flags.DEFINE_boolean('if_precondition', True, 'if use preconditioning')
  flags.DEFINE_float('c_on_rho', 10.0, 'the constant added on rho')
  flags.DEFINE_float('epsl', 0.0, 'diffusion coefficient')

  flags.DEFINE_integer('v_method', 1, 'method for v')
  flags.DEFINE_integer('rho_method', 1, 'method for rho')
  flags.DEFINE_integer('updating_rho_first', 1, '1 if update rho first')

  app.run(main)
