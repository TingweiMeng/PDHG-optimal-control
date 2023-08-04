import jax
import jax.numpy as jnp
from einshape import jax_einshape as einshape
import numpy as np
from absl import app, flags, logging
import matplotlib.pyplot as plt
import os
from print_n_plot import read_solution


def plot_solution_2d(phi, nt, n_spatial, T, period_spatial, figname, epsl=0.0, T_divisor=4):
  '''
  @ parameters:
    phi: [nt, nx, ny]
    nt: integer
    n_spatial: [nx, ny]
    T: float
    period_spatial: [x_period, y_period]
    figname: string for saving figure
    epsl: float, diffusion coefficient
    T_divisor: integer, number of figures to plot
  '''
  nx, ny = n_spatial[0], n_spatial[1]
  x_period, y_period = period_spatial[0], period_spatial[1]
  name_prefix = 'nt{}_nx{}_ny{}'.format(nt, nx, ny)
  if epsl == 0:
    name_prefix += '_epsl0_'
  elif epsl == 0.1:
    name_prefix += '_epsl0p1_'
  else:
    raise ValueError("epsl must be 0 or 0.1")
  x_arr = np.linspace(0.0, x_period, num = nx + 1, endpoint = True)
  y_arr = np.linspace(0.0, y_period, num = ny + 1, endpoint = True)
  x_mesh, y_mesh = np.meshgrid(x_arr, y_arr, indexing='ij')

  dt = T / (nt-1)
  for i in range (T_divisor):
    ind = (nt-1)*(i+1)//T_divisor
    phi_terminal = jnp.concatenate([phi[ind,...], phi[ind, 0:1, :]], axis = 0)  # [nx+1, ny]
    phi_terminal_np = jnp.concatenate([phi_terminal, phi_terminal[:,0:1]], axis = 1)  # [nx+1, ny+1]
    t = dt * ind
    fig = plt.figure()
    plt.contourf(x_mesh, y_mesh, phi_terminal_np)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(figname + name_prefix + 'solution_t_{}.png'.format(t))
    plt.close(fig)
    

def main(argv):
  for key, value in FLAGS.__flags.items():
    print(value.name, ": ", value._value, flush=True)

  nt = FLAGS.nt
  nx = FLAGS.nx
  ny = FLAGS.ny
  figname = FLAGS.figname
  epsl = FLAGS.epsl
  T_divisor = FLAGS.T_divisor
  egno = FLAGS.egno
  num_filename = FLAGS.numerical_sol_filename

  ndim = 2
  n_spatial = [nx, ny]
  T = 1
  period_spatial = [2, 2]
  plot_foldername = "eg{}_{}d/".format(egno, ndim)
  if FLAGS.hero_folder != '':
    plot_foldername = FLAGS.hero_folder + '/' + plot_foldername
  if not os.path.exists(plot_foldername):
    os.makedirs(plot_foldername)

  num_sol = read_solution(num_filename)
  plot_solution_2d(num_sol, nt, n_spatial, T, period_spatial, figname, epsl, T_divisor)


if __name__ == '__main__':
  FLAGS = flags.FLAGS
  flags.DEFINE_integer('nt', 100, 'number of time steps')
  flags.DEFINE_integer('nx', 100, 'number of grid points in x')
  flags.DEFINE_integer('ny', 100, 'number of grid points in y')
  flags.DEFINE_integer('egno', 0, 'example number')
  flags.DEFINE_float('epsl', 0.0, 'diffusion coefficient')
  flags.DEFINE_string('figname', '2d_solution', 'figure name')
  flags.DEFINE_string('numerical_sol_filename', '', 'the name of the pickle file of numerical solution to read')
  flags.DEFINE_integer('T_divisor', 4, 'number of figures to plot')
  flags.DEFINE_string('hero_folder', '', 'the folder name of hero run')
  app.run(main)