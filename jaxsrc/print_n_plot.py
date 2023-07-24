import jax
import jax.numpy as jnp
from einshape import jax_einshape as einshape
import utils
import numpy as np
from absl import app, flags, logging
import pickle
import matplotlib.pyplot as plt
import os
from solver import set_up_example_fns
from functools import partial

def H_L1_true_sol_1d(x, t, fn_J):
  '''
  @params:
    x: [1]
    t: scalar
    fn_J: function taking [...,1] and returning [...]
  @return:
    phi: scalar
  '''
  # phi(x,t) = min_{ |u-x| <= t } J(u)
  n_grids = 101
  x_tests = jnp.linspace(x[0]-t, x[0]+t, num = n_grids)[...,None]  # [n_grids, 1]
  J_tests = fn_J(x_tests)  # [n_grids]
  return jnp.min(J_tests)

H_L1_true_sol_1d_batch = partial(jax.jit, static_argnums=(2))(jax.vmap(H_L1_true_sol_1d, in_axes=(0, 0, None), out_axes=0))
H_L1_true_sol_1d_batch2 = jax.vmap(H_L1_true_sol_1d_batch, in_axes=(0, 0, None), out_axes=0)

# @jax.jit
def compute_EO_forward_solution_1d(nt, dx, dt, f_in_H, c_in_H, g):
    '''
    @parameters:
        nt: integers
        dx, dt: floats
        f_in_H, c_in_H, g: [nx]
        g: [nx]
    @return:
        phi: [nt, nx]
    '''
    phi = []
    phi.append(g)
    for i in range(nt-1):
        dphidx_left = (phi[i] - jnp.roll(phi[i], 1))/dx
        dphidx_right = (jnp.roll(phi[i], -1) - phi[i])/dx
        H_1norm = jnp.maximum(-dphidx_right, 0) + jnp.maximum(dphidx_left, 0)
        H_val = c_in_H * H_1norm + f_in_H
        phi.append(phi[i] - dt * H_val)
    phi_arr = jnp.stack(phi, axis = 0)
    print("phi dimension {}".format(jnp.shape(phi_arr)))
    print("phi {}".format(phi_arr))
    return phi_arr  # TODO: check dimension
    
def compute_EO_forward_solution_2d(nt, dx, dy, dt, f_in_H, c_in_H, g):
    '''
    @parameters:
        nt: integers
        dx, dy, dt: floats
        f_in_H, c_in_H, g: [nx, ny]
    @return:
        phi: [nt, nx, ny]
    '''
    phi = []
    phi.append(g)
    for i in range(nt-1):
        dphidx_left = (phi[i] - jnp.roll(phi[i], 1, axis = 0))/dx
        dphidx_right = (jnp.roll(phi[i], -1, axis = 0) - phi[i])/dx
        dphidy_left = (phi[i] - jnp.roll(phi[i], 1, axis = 1))/dy
        dphidy_right = (jnp.roll(phi[i], -1, axis = 1) - phi[i])/dy
        H_1norm = jnp.maximum(-dphidx_right, 0) + jnp.maximum(dphidx_left, 0) \
                    + jnp.maximum(-dphidy_right, 0) + jnp.maximum(dphidy_left, 0)  # [nx, ny]
        H_val = c_in_H * H_1norm + f_in_H  # [nx, ny]
        phi.append(phi[i] - dt * H_val)
    phi_arr = jnp.array(phi)
    print("phi dimension {}".format(jnp.shape(phi_arr)))
    return phi_arr  # TODO: check dimension

def read_solution(filename):
    with open(filename, 'rb') as f:
        results, errors = pickle.load(f)
    # compute errors
    phi = results[-1][-1]
    return phi

def get_sol_on_coarse_grid_1d(sol, coarse_nt, coarse_nx):
    nt, nx = jnp.shape(sol)
    if (nt - 1) % (coarse_nt - 1) != 0 or nx % coarse_nx != 0:
        raise ValueError("nx and nt-1 must be divisible by coarse_nx and coarse_nt-1")
    else:
        nt_factor = (nt-1) // (coarse_nt-1)
        nx_factor = nx // coarse_nx
        sol_coarse = sol[::nt_factor, ::nx_factor]
    return sol_coarse

def get_sol_on_coarse_grid_2d(sol, coarse_nt, coarse_nx, coarse_ny):
    nt, nx, ny = jnp.shape(sol)
    if (nt - 1) % (coarse_nt - 1) != 0 or nx % coarse_nx != 0 or ny % coarse_ny != 0:
        raise ValueError("nx, ny and nt-1 must be divisible by coarse_nx, coarse_ny and coarse_nt-1")
    else:
        nt_factor = (nt-1) // (coarse_nt-1)
        nx_factor = nx // coarse_nx
        ny_factor = ny // coarse_ny
        sol_coarse = sol[::nt_factor, ::nx_factor, ::ny_factor]
    return sol_coarse

def compute_err_1d(phi, true_soln):
    '''
    @parameters:
        phi: [nt, nx]
        true_soln: [nt_dense, nx_dense]
    @return:
        err_l1, err_l1_rel: scalar
        error: [nt, nx]
    '''
    nt, nx = jnp.shape(phi)
    phi_true = get_sol_on_coarse_grid_1d(true_soln, nt, nx)
    error = phi - phi_true
    err_l1 = jnp.mean(jnp.abs(error))
    err_l1_rel = err_l1/ jnp.mean(jnp.abs(phi_true))
    return err_l1, err_l1_rel, error

def compute_err_2d(phi, true_soln):
    '''
    @parameters:
        phi: [nt, nx, ny]
        true_soln: [nt_dense, nx_dense, ny_dense]
    @return:
        err_l1, err_l1_rel: scalar
        error: [nt, nx, ny]
    '''
    nt, nx, ny = jnp.shape(phi)
    phi_true = get_sol_on_coarse_grid_2d(true_soln, nt, nx, ny)
    error = phi - phi_true
    err_l1 = jnp.mean(jnp.abs(error))
    err_l1_rel = err_l1/ jnp.mean(jnp.abs(phi_true))
    return err_l1, err_l1_rel, error

def plot_solution_1d(phi, error, nt, nx, T, x_period, figname):
    '''
    @ parameters:
        phi, error: [nt, nx]
        nt, nx: integer
        T, x_period: float
        figname: string for saving figure
    '''
    x_arr = np.linspace(0.0, x_period, num = nx + 1, endpoint = True)
    t_arr = np.linspace(0.0, T, num = nt, endpoint = True)
    t_mesh, x_mesh = np.meshgrid(t_arr, x_arr)
    print("shape x_arr {}, t_arr {}".format(np.shape(x_arr), np.shape(t_arr)))
    print("shape x_mesh {}, t_mesh {}".format(np.shape(x_mesh), np.shape(t_mesh)))
    # plot solution
    phi_trans = einshape('ij->ji', phi)  # [nx, nt]
    phi_np = jax.device_get(jnp.concatenate([phi_trans, phi_trans[0:1,:]], axis = 0))  # [nx+1, nt]
    fig = plt.figure()
    plt.contourf(x_mesh, t_mesh, phi_np)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.savefig(figname + 'nt{}_nx{}_solution.png'.format(nt, nx))
    # plot error
    err_trans = einshape('ij->ji', error)  # [nx, nt]
    err_np = jax.device_get(jnp.concatenate([err_trans, err_trans[0:1,:]], axis = 0))  # [nx+1, nt]
    fig = plt.figure()
    plt.contourf(x_mesh, t_mesh, err_np)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.savefig(figname + 'nt{}_nx{}_error.png'.format(nt, nx))


def plot_solution_2d(phi, error, nt, nx, ny, T, x_period, y_period, figname):
    '''
    @ parameters:
        phi, error: [nt, nx, ny]
        nt, nx, ny: integer
        T, x_period, y_period: float
        figname: string for saving figure
    '''
    x_arr = np.linspace(0.0, x_period, num = nx + 1, endpoint = True)
    y_arr = np.linspace(0.0, y_period, num = ny + 1, endpoint = True)
    x_mesh, y_mesh = np.meshgrid(x_arr, y_arr)
    # plot solution
    phi_terminal = jnp.concatenate([phi[-1,...], phi[-1, 0:1, :]], axis = 0)  # [nx+1, ny]
    phi_terminal_np = jnp.concatenate([phi_terminal, phi_terminal[:,0:1]], axis = 1)  # [nx+1, ny+1]
    fig = plt.figure()
    plt.contourf(x_mesh, y_mesh, phi_terminal_np)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(figname + 'nt{}_nx{}_ny{}.png'.format(nt, nx, ny))
    # plot error
    err_terminal = jnp.concatenate([error[-1,...], error[-1, 0:1, :]], axis = 0)  # [nx+1, ny]
    err_terminal_np = jnp.concatenate([err_terminal, err_terminal[:,0:1]], axis = 1)  # [nx+1, ny+1]
    fig = plt.figure()
    plt.contourf(x_mesh, y_mesh, err_terminal_np)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(figname + 'nt{}_nx{}_ny{}_error.png'.format(nt, nx, ny))

def get_save_dir(time_stamp, egno, ndim, nt, nx, ny):
  save_dir = './check_points/{}'.format(time_stamp) + '/eg{}_{}d'.format(egno, ndim)
  if ndim == 1:
    filename_prefix = 'nt{}_nx{}'.format(nt, nx)
  elif ndim == 2:
    filename_prefix = 'nt{}_nx{}_ny{}'.format(nt, nx, ny)
  return save_dir, filename_prefix

def compute_ground_truth(egno, ndim, T, x_period, y_period):
    J, f_in_H_fn, c_in_H_fn = set_up_example_fns(egno, ndim, x_period, y_period)
    nt_dense = 16001
    nx_dense = 8000
    ny_dense = 8000
    dt_dense = T / (nt_dense - 1)
    dx_dense = x_period / nx_dense
    dy_dense = y_period / ny_dense
    if ndim == 1:
        x_arr_dense = np.linspace(0.0, x_period, num = nx_dense + 1, endpoint = True)
        x_arr_1d = jnp.array(x_arr_dense[:-1, None])  # [nx_dense, 1]
        g = J(x_arr_1d)  # [nx_dense]
        f_in_H = f_in_H_fn(x_arr_1d)  # [nx_dense]
        c_in_H = c_in_H_fn(x_arr_1d)  # [nx_dense]
        print("shape g {}, f_in_H {}, c_in_H {}".format(jnp.shape(g), jnp.shape(f_in_H), jnp.shape(c_in_H)))
        if egno == 0:
            t_arr_1d = jnp.linspace(0.0, T, num = nt_dense)  # [nt_dense]
            phi_dense_list = []
            for i in range(nt_dense):
                phi_dense_list.append(H_L1_true_sol_1d_batch(x_arr_1d, t_arr_1d[i] + jnp.zeros(nx_dense), J))
            phi_dense = jnp.stack(phi_dense_list, axis = 0)
        else:
            phi_dense = compute_EO_forward_solution_1d(nt_dense, dx_dense, dt_dense, f_in_H, c_in_H, g)
    elif ndim == 2:
        x_arr_dense = np.linspace(0.0, x_period, num = nx_dense + 1, endpoint = True)
        y_arr_dense = np.linspace(0.0, y_period, num = ny_dense + 1, endpoint = True)
        x_mesh_dense, y_mesh_dense = np.meshgrid(x_arr_dense, y_arr_dense)
        x_arr_2d = jnp.array(np.concatenate([x_mesh_dense[:-1,:-1, None], y_mesh_dense[:-1, :-1, None]], axis = -1))  # [nx_dense, ny_dense, 2]
        g = J(x_arr_2d)  # [nx_dense, ny_dense]
        f_in_H = f_in_H_fn(x_arr_2d)  # [nx_dense, ny_dense]
        c_in_H = c_in_H_fn(x_arr_2d)  # [nx_dense, ny_dense]
        print("shape g {}, f_in_H {}, c_in_H {}".format(jnp.shape(g), jnp.shape(f_in_H), jnp.shape(c_in_H)))
        phi_dense = compute_EO_forward_solution_2d(nt_dense, dx_dense, dy_dense, dt_dense, f_in_H, c_in_H, g)
    else:
        raise ValueError("ndim should be 1 or 2")
    return phi_dense

def main(argv):
    nt = FLAGS.nt
    nx = FLAGS.nx
    ny = FLAGS.ny
    ndim = FLAGS.ndim
    egno = FLAGS.egno

    T = 1.0
    x_period = 2.0
    y_period = 2.0

    figname = "./eg{}_{}d/".format(egno, ndim)
    if not os.path.exists(figname):
        os.makedirs(figname)

    # saved_file_dir, saved_filename_prefix = get_save_dir(FLAGS.time_stamp, egno, ndim, nt, nx, ny)

    # filename = saved_file_dir + '/' + saved_filename_prefix + '_iter{}.pickle'.format(iterno)
    filename = FLAGS.filename
    phi = read_solution(filename)
    phi_dense = compute_ground_truth(egno, ndim, T, x_period, y_period)

    # compute error
    if ndim == 1:        
        err_l1, err_l1_rel, error = compute_err_1d(phi, phi_dense)
        plot_solution_1d(phi, error, nt, nx, T, x_period, figname)
    elif ndim == 2:
        err_l1, err_l1_rel, error = compute_err_2d(phi, phi_dense)
        plot_solution_2d(phi, error, nt, nx, ny, T, x_period, y_period, figname)
    print("err_l1 {:.2E}, err_l1_rel {:.2E}".format(err_l1, err_l1_rel))
    

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('nt', 11, 'size of t grids')
    flags.DEFINE_integer('nx', 20, 'size of x grids')
    flags.DEFINE_integer('ny', 20, 'size of y grids')
    flags.DEFINE_integer('ndim', 1, 'dimensionality')
    flags.DEFINE_integer('egno', 1, 'index of example')
    # flags.DEFINE_integer('iterno', 100000, 'iteration number in filename')
    # flags.DEFINE_string('time_stamp', '', 'time stamp in the filename')
    flags.DEFINE_string('filename', '', 'the name of the pickle file to read')
    app.run(main)

    
    
