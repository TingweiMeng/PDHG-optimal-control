import jax.numpy as jnp
import numpy as np
from absl import app, flags
import matplotlib.pyplot as plt
from einshape import jax_einshape as einshape
import os
from print_n_plot import read_solution
import utils.utils as utils
import tensorflow as tf

def plot_solution_1d(phi, x_arr, t_arr, title = '', tfboard = True):
  ''' plot solution and error of 1d HJ PDE
  @ parameters:
    phi, error: [nt, nx]
    x_arr: [1, nx, 1]
    t_arr: [nt, 1]
    title: string
    tfboard: bool, if True, return tf image
  '''
  x_arr = jnp.squeeze(x_arr)  # [nx]
  t_arr = jnp.squeeze(t_arr)  # [nt]
  assert phi.shape[0] == t_arr.shape[0], f"arr shape_0 ({phi.shape[0]}) != t_arr shape ({t_arr.shape[0]})"
  assert phi.shape[1] == x_arr.shape[0], f"arr shape_1 ({phi.shape[1]}) != x_arr shape ({x_arr.shape[0]})"
  t_mesh, x_mesh = np.meshgrid(t_arr, x_arr)  # [nx, nt]
  # plot solution
  phi_trans = einshape('ij->ji', phi)  # [nx, nt]
  fig = plt.figure()
  plt.pcolormesh(x_mesh, t_mesh, phi_trans)
  plt.contour(x_mesh, t_mesh, phi_trans, colors='k')
  plt.colorbar()
  plt.xlabel('x')
  plt.ylabel('t')
  if title != '' and title is not None:
    plt.title(title)
  if tfboard:
    return utils.plot_to_image(fig)
  else:
    return fig
    

def plot_solution_2d(phi, x_arr, t_arr, T_divisor=4, title = '', tfboard = True):
  '''
  @ parameters:
    phi: [nt, nx, ny]
    x_arr: [1, nx, ny, 2]
    t_arr: [nt, 1, 1]
    T_divisor: integer, number of figures to plot
    tfboard: bool, if True, return tf image
  @ returns:
    tf image or fig
  '''
  x_arr = jnp.squeeze(x_arr)  # [nx, ny, 2]
  t_arr = jnp.squeeze(t_arr)  # [nt]

  fig, axs = plt.subplots(T_divisor // 2, 2, figsize=(10, 10))
  nt = t_arr.shape[0]
  for i in range(T_divisor):
    ind = (nt-1) // (T_divisor - 1) * i
    ax = axs[i // 2, i % 2]
    ax.set_title('t = {:.2f}'.format(t_arr[ind]))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # contourf plot
    ct = ax.contourf(x_arr[...,0], x_arr[...,1], phi[ind,...])  # TODO: check domain 
    fig.colorbar(ct, ax=ax)
  if title != '' and title is not None:
    fig.suptitle(title)
  if tfboard:
    return utils.plot_to_image(fig)
  else:
    return fig
    
def plot_traj_1d(traj, t_arr, title = '', tfboard = True):
  ''' traj: [n_samples, nt], t_arr: [nt] '''
  fig = plt.figure()
  plt.plot(t_arr, traj.T)
  plt.xlabel('t')
  if title != '' and title is not None:
    plt.title(title)
  if tfboard:
    return utils.plot_to_image(fig)
  else:
    return fig
  
def plot_traj_2d(traj, title = '', tfboard = True):
  ''' traj: [n_samples, nt, 2], t_arr: [nt] '''
  fig = plt.figure()
  plt.plot(traj[...,0].T, traj[...,1].T)
  plt.xlabel('x')
  plt.ylabel('y')
  if title != '' and title is not None:
    plt.title(title)
  if tfboard:
    return utils.plot_to_image(fig)
  else:
    return fig

def save_fig(fig, filename, tfboard = True, foldername = None):
  # filename is the title for tfboard if tfboard is True
  if tfboard:
    tf.summary.image(filename, fig, step = 0)
  else:
    if foldername is not None:
      filename = foldername + filename
      if not os.path.exists(foldername):
        os.makedirs(foldername)
    fig.savefig(filename + '.png')
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