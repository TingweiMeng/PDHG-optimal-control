import jax.numpy as jnp
import numpy as np
from absl import app, flags
import matplotlib.pyplot as plt
from einshape import jax_einshape as einshape
import os
from print_n_plot import read_solution

def plot_solution_1d(phi, nt, nspatial, T, period_spatial, foldername, filename, epsl = 0, T_divisor=1):
  ''' plot solution and error of 1d HJ PDE
  @ parameters:
    phi, error: [nt, nx]
    nt: integer
    nspatial: list of one number nx
    T: float
    period_spatial: list of one number x_period
    figname: string for saving figure
    epsl: float, diffusion coefficient
  '''
  nx = nspatial[0]
  x_period = period_spatial[0]
  name_prefix = 'nt{}_nx{}'.format(nt, nx)
  if epsl == 0:
    name_prefix += '_epsl0_'
  elif epsl == 0.1:
    name_prefix += '_epsl0p1_'
  else:
    raise ValueError("epsl must be 0 or 0.1")
  if phi.shape[0] != nt:
    raise ValueError("phi.shape[0] != nt")
  if phi.shape[1] == nx:
    phi = jnp.concatenate((phi, phi[:,0:1]), axis = 1)  
  elif phi.shape[1] != nx + 1:
    raise ValueError("phi.shape[1] != nx + 1 and phi.shape[1] != nx")
  x_arr = np.linspace(0.0, x_period, num = nx + 1, endpoint = True)
  t_arr = np.linspace(0.0, T, num = nt, endpoint = True)
  t_mesh, x_mesh = np.meshgrid(t_arr, x_arr)
  # plot solution
  phi_trans = einshape('ij->ji', phi)  # [nx, nt]
  fig = plt.figure()
  plt.contourf(x_mesh, t_mesh, phi_trans)
  plt.colorbar()
  plt.xlabel('x')
  plt.ylabel('t')
  plt.savefig(foldername + name_prefix + filename + '.png')



def plot_solution_2d(phi, nt, n_spatial, T, period_spatial, foldername, filename, epsl=0.0, T_divisor=4):
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

  # adjust boundary values
  if phi.shape[1] == nx:
    phi = jnp.concatenate((phi, phi[:,0:1,:]), axis = 1)
  elif phi.shape[1] != nx + 1:
    raise ValueError("phi.shape[1] != nx + 1 and phi.shape[1] != nx")
  if phi.shape[2] == ny:
    phi = jnp.concatenate((phi, phi[:,:,0:1]), axis = 2)
  elif phi.shape[2] != ny + 1:
    raise ValueError("phi.shape[2] != ny + 1 and phi.shape[2] != ny")

  dt = T / (nt-1)
  for i in range (T_divisor):
    ind = (nt-1)*(i+1)//T_divisor
    t = dt * ind
    fig = plt.figure()
    plt.contourf(x_mesh, y_mesh, phi[ind,...])
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(foldername + name_prefix + filename + '_t_{:.2f}.png'.format(t))
    plt.close(fig)
    

def main(argv):
  for key, value in FLAGS.__flags.items():
    print(value.name, ": ", value._value, flush=True)

  epsl = FLAGS.epsl
  T_divisor = FLAGS.T_divisor
  egno = FLAGS.egno
  num_filename = FLAGS.numerical_sol_filename
  ndim = FLAGS.ndim

  T = 1
  if ndim == 1:
    period_spatial = [2]
    plot_fn = plot_solution_1d
  else:
    period_spatial = [2, 2]
    plot_fn = plot_solution_2d

  plot_foldername = "eg{}_{}d/".format(egno, ndim)
  if FLAGS.parent_folder != '':
    plot_foldername = FLAGS.parent_folder + '/' + plot_foldername
  if not os.path.exists(plot_foldername):
    os.makedirs(plot_foldername)

  num_sol = read_solution(num_filename)
  if num_sol.ndim != ndim + 1:
    raise ValueError("num_sol.ndim != ndim + 1")
  nt = num_sol.shape[0]
  n_spatial = num_sol.shape[1:]  
  plot_fn(num_sol, nt, n_spatial, T, period_spatial, plot_foldername, 'solution', epsl, T_divisor = T_divisor)

  print('plotting done')


if __name__ == '__main__':
  FLAGS = flags.FLAGS
  flags.DEFINE_integer('egno', 0, 'example number')
  flags.DEFINE_integer('ndim', 1, 'number of dimensions')
  flags.DEFINE_float('epsl', 0.0, 'diffusion coefficient')
  flags.DEFINE_string('numerical_sol_filename', '', 'the name of the pickle file of numerical solution to read')
  flags.DEFINE_integer('T_divisor', 4, 'number of figures to plot')
  flags.DEFINE_string('parent_folder', '', 'the parent folder name')
  app.run(main)