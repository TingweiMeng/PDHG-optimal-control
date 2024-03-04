import jax.numpy as jnp
import numpy as np
from absl import app, flags
import matplotlib.pyplot as plt
from einshape import jax_einshape as einshape
import os
import utils.utils as utils
import tensorflow as tf
import math

def plot_solution_1d(phi, x_arr, t_arr, title = '', tfboard = True, num_cols = 1):
  ''' plot solution and error of 1d HJ PDE
  @ parameters:
    phi, error: [nt, nx]
    x_arr: [1, nx, 1]
    t_arr: [nt, 1]
    title: string
    tfboard: bool, if True, return tf image
    num_cols: unused
  @ returns:
    tf image or fig
  '''
  x_arr = jnp.squeeze(x_arr)  # [nx]
  t_arr = jnp.squeeze(t_arr)  # [nt]
  assert phi.shape[0] == t_arr.shape[0], f"arr shape_0 ({phi.shape[0]}) != t_arr shape ({t_arr.shape[0]})"
  assert phi.shape[1] == x_arr.shape[0], f"arr shape_1 ({phi.shape[1]}) != x_arr shape ({x_arr.shape[0]})"
  t_mesh, x_mesh = np.meshgrid(t_arr, x_arr)  # [nx, nt]
  # plot solution
  phi_trans = einshape('ij->ji', phi)  # [nx, nt]
  fig = plt.figure()
  plt.contourf(x_mesh, t_mesh, phi_trans)
  plt.colorbar()
  plt.xlabel('x')
  plt.ylabel('t')
  if title != '' and title is not None:
    plt.title(title)
  if tfboard:
    return utils.plot_to_image(fig)
  else:
    return fig
    

def plot_solution_2d(phi, x_arr, t_arr, T_divisor=4, title = '', tfboard = True, num_cols = 2):
  '''
  @ parameters:
    phi: [nt, nx, ny]
    x_arr: [1, nx, ny, 2]
    t_arr: [nt, 1, 1]
    T_divisor: integer, number of figures to plot
    title: string
    tfboard: bool, if True, return tf image
    num_cols: integer, number of columns in the figure
  @ returns:
    tf image or fig
  '''
  x_arr = jnp.squeeze(x_arr)  # [nx, ny, 2]
  t_arr = jnp.squeeze(t_arr)  # [nt]

  fig, axs = plt.subplots(math.ceil(T_divisor / num_cols), num_cols, figsize=(10, 10))
  nt = t_arr.shape[0]
  for i in range(T_divisor):
    ind = (nt-1) // (T_divisor - 1) * i
    if num_cols > 1:
      ax = axs[i // num_cols, i % num_cols]
    else:
      ax = axs[i]
    ax.set_title('t = {:.2f}'.format(t_arr[ind]))
    # contourf plot
    ct = ax.contourf(x_arr[...,0], x_arr[...,1], phi[ind,...])  # TODO: check domain 
    fig.colorbar(ct, ax=ax)
  if title != '' and title is not None:
    fig.suptitle(title)
  plt.subplots_adjust(hspace=0.5)
  if tfboard:
    return utils.plot_to_image(fig)
  else:
    return fig
    
def plot_traj_1d(traj, t_arr, title = '', tfboard = True):
  ''' traj: [nt, n_samples], t_arr: [nt] '''
  fig = plt.figure()
  plt.plot(t_arr, traj)
  plt.xlabel('t')
  if title != '' and title is not None:
    plt.title(title)
  if tfboard:
    return utils.plot_to_image(fig)
  else:
    return fig
  
def plot_traj_2d(traj, title = '', tfboard = True):
  ''' traj: [nt, n_samples, 2], t_arr: [nt] '''
  fig = plt.figure()
  plt.plot(traj[:,:,0], traj[:,:,1])
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
      filename = os.path.join(foldername, filename)
      if not os.path.exists(foldername):
        os.makedirs(foldername)
    fig.savefig(filename + '.png')
    print('figure saved as', filename + '.png', flush=True)
