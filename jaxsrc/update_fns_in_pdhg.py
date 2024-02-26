import jax
import jax.numpy as jnp
from functools import partial
import os
from utils.utils_precond import H1_precond_1d, H1_precond_2d
from utils.utils_diff_op import Dx_right_decreasedim, Dx_left_decreasedim, Dxx_decreasedim, Dt_decreasedim, \
  Dx_left_increasedim, Dx_right_increasedim, Dxx_increasedim, Dt_increasedim, Dy_left_decreasedim, \
  Dy_right_decreasedim, Dy_left_increasedim, Dy_right_increasedim, Dyy_increasedim, Dyy_decreasedim

jax.config.update("jax_enable_x64", True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# def get_Dalp_2d(Df_fn, DL_fn, rho, alp, alp_sum, rho_prev, alp_prev, x_arr, t_arr, tau, Dphi):
#   ''' @ parameters:
#       Df_fn: function, [nt-1, nx, ny, n_ctrl], x_arr, t_arr -> [nt-1, nx, ny, n_ctrl]
#       DL_fn: function, [nt-1, nx, ny, n_ctrl], x_arr, t_arr -> [nt-1, nx, ny, n_ctrl]
#       rho: [nt-1, nx, ny, 1]
#       alp: [nt-1, nx, ny, n_ctrl]
#       alp_sum: [nt-1, nx, ny, n_ctrl] (alp1_x + alp2_x + alp1_y + alp2_y)
#       rho_prev: [nt-1, nx, ny, 1]
#       alp_prev: [nt-1, nx, ny, n_ctrl]
#       x_arr: vec that can be broadcasted to [nt-1, nx, ny]
#       t_arr: vec that can be broadcasted to [nt-1, nx, ny]
#       tau: float
#       Dphi: [nt-1, nx, ny, 1]
#     @ return:
#       Dalp: [nt-1, nx, ny, n_ctrl]
#   '''
#   Dalp = -Df_fn(alp, x_arr, t_arr) * Dphi * rho - rho / tau * (alp * rho - alp_prev * rho_prev)  # [nt-1, nx, ny, n_ctrl]
#   Dalp -= DL_fn(alp_sum, x_arr, t_arr) * rho  # [nt-1, nx, ny, n_ctrl]
#   return Dalp


# def get_Dalp_1d(Df_fn, DL_fn, rho, alp, alp_sum, rho_prev, alp_prev, x_arr, t_arr, tau, Dphi):
#   ''' @ parameters:
#       Df_fn: function, [nt-1, nx, 1], x_arr, t_arr -> [nt-1, nx, 1]
#       DL_fn: function, [nt-1, nx, 1], x_arr, t_arr -> [nt-1, nx, 1]
#       rho: [nt-1, nx, 1]
#       alp: [nt-1, nx, 1]
#       alp_sum: [nt-1, nx, 1] (alp1_x + alp2_x)
#       rho_prev: [nt-1, nx, 1]
#       alp_prev: [nt-1, nx, 1]
#       x_arr: vec that can be broadcasted to [nt-1, nx]
#       t_arr: vec that can be broadcasted to [nt-1, nx]
#       tau: float
#       Dphi: [nt-1, nx, 1]
#     @ return:
#       Dalp: [nt-1, nx, 1]
#   '''
#   Dalp = -Df_fn(alp, x_arr, t_arr) * Dphi * rho - rho / tau * (alp * rho - alp_prev * rho_prev)  # [nt-1, nx, ny, n_ctrl]
#   Dalp -= DL_fn(alp_sum, x_arr, t_arr) * rho  # [nt-1, nx, ny, n_ctrl]
#   return Dalp

# def get_D2alp_1d(Hf_fn, HL_fn, rho, alp, alp_sum, x_arr, t_arr, tau, Dphi):
#   ''' @ parameters:
#       Hf_fn: function, [nt-1, nx, 1], x_arr, t_arr -> [nt-1, nx, 1, 1]
#       HL_fn: function, [nt-1, nx, 1], x_arr, t_arr -> [nt-1, nx, 1, 1]
#       rho: [nt-1, nx, 1, 1]
#       alp: [nt-1, nx, 1]
#       alp_sum: [nt-1, nx, 1] (alp1_x + alp2_x)
#       x_arr: vec that can be broadcasted to [nt-1, nx]
#       t_arr: vec that can be broadcasted to [nt-1, nx]
#       tau: float
#       Dphi: [nt-1, nx, 1, 1]
#     @ return:
#       D2alp: [nt-1, nx, 1, 1]
#   '''
#   D2alp = -Hf_fn(alp, x_arr, t_arr) * Dphi * rho - rho ** 2 / tau  # [nt-1, nx, 1, 1]
#   D2alp -= HL_fn(alp_sum, x_arr, t_arr) * rho  # [nt-1, nx, 1, 1]
#   return D2alp

# def get_D2alp_2d(Hf_fn, HL_fn, rho, alp, alp_sum, x_arr, t_arr, tau, Dphi):
#   ''' @ parameters:
#       Hf_fn: function, [nt-1, nx, ny, n_ctrl], x_arr, t_arr -> [nt-1, nx, ny, n_ctrl, n_ctrl]
#       HL_fn: function, [nt-1, nx, ny, n_ctrl], x_arr, t_arr -> [nt-1, nx, ny, n_ctrl, n_ctrl]
#       rho: [nt-1, nx, ny, 1, 1]
#       alp: [nt-1, nx, ny, n_ctrl]
#       alp_sum: [nt-1, nx, ny, n_ctrl] (alp1_x + alp2_x + alp1_y + alp2_y)
#       x_arr: vec that can be broadcasted to [nt-1, nx, ny]
#       t_arr: vec that can be broadcasted to [nt-1, nx, ny]
#       tau: float
#       Dphi: [nt-1, nx, ny, 1, 1]
#     @ return:
#       D2alp: [nt-1, nx, ny, n_ctrl, n_ctrl]
#   '''
#   mat_eyes = jnp.eye(alp.shape[-1], dtype = jnp.float64)  # [n_ctrl, n_ctrl]
#   D2alp = -Hf_fn(alp, x_arr, t_arr) * Dphi * rho - rho ** 2 / tau * mat_eyes  # [nt-1, nx, ny, n_ctrl, n_ctrl]
#   D2alp -= HL_fn(alp_sum, x_arr, t_arr) * rho  # [nt-1, nx, ny, n_ctrl, n_ctrl]
#   return D2alp

# def get_DalpDrho_1d(Df_fn, DL_fn, rho, alp, alp_sum, rho_prev, alp_prev, x_arr, t_arr, tau, Dphi):
#   ''' @ parameters:
#       Df_fn: function, [nt-1, nx, 1], x_arr, t_arr -> [nt-1, nx, 1]
#       DL_fn: function, [nt-1, nx, 1], x_arr, t_arr -> [nt-1, nx, 1]
#       rho: [nt-1, nx, 1]
#       alp: [nt-1, nx, 1]
#       alp_sum: [nt-1, nx, 1] (alp1_x + alp2_x)
#       rho_prev: [nt-1, nx, 1]
#       alp_prev: [nt-1, nx, 1]
#       x_arr: vec that can be broadcasted to [nt-1, nx]
#       t_arr: vec that can be broadcasted to [nt-1, nx]
#       tau: float
#       Dphi: [nt-1, nx, 1]
#     @ return:
#       Dalp: [nt-1, nx, 1]
#   '''
#   DalpDrho = -Df_fn(alp, x_arr, t_arr) * Dphi - (2* alp * rho - alp_prev * rho_prev) / tau  # [nt-1, nx, 1]
#   DalpDrho -= DL_fn(alp_sum, x_arr, t_arr)  # [nt-1, nx, 1]
#   return DalpDrho

# def get_DalpDrho_2d(Df_fn, DL_fn, rho, alp, alp_sum, rho_prev, alp_prev, x_arr, t_arr, tau, Dphi):
#   ''' @ parameters:
#       Df_fn: function, [nt-1, nx, ny, n_ctrl], x_arr, t_arr -> [nt-1, nx, ny, n_ctrl]
#       DL_fn: function, [nt-1, nx, ny, n_ctrl], x_arr, t_arr -> [nt-1, nx, ny, n_ctrl]
#       rho: [nt-1, nx, ny, 1]
#       alp: [nt-1, nx, ny, n_ctrl]
#       alp_sum: [nt-1, nx, ny, n_ctrl] (alp1_x + alp2_x + alp1_y + alp2_y)
#       rho_prev: [nt-1, nx, ny, 1]
#       alp_prev: [nt-1, nx, ny, n_ctrl]
#       x_arr: vec that can be broadcasted to [nt-1, nx, ny]
#       t_arr: vec that can be broadcasted to [nt-1, nx, ny]
#       tau: float
#       Dphi: [nt-1, nx, ny, 1]
#     @ return:
#       Dalp: [nt-1, nx, ny, n_ctrl]
#   '''
#   DalpDrho = -Df_fn(alp, x_arr, t_arr) * Dphi - (2* alp * rho - alp_prev * rho_prev) / tau  # [nt-1, nx, ny, n_ctrl]
#   DalpDrho -= DL_fn(alp_sum, x_arr, t_arr)  # [nt-1, nx, ny, n_ctrl]
#   return DalpDrho

# # @partial(jax.jit, static_argnames=("fns_dict",))
# def get_diff_in_Newton_2d(x, arg_other, fns_dict):
#   rho = x[...,0:1]  # [nt-1, nx, ny, 1]
#   alp1_x = x[...,1:3]  # [nt-1, nx, ny, n_ctrl]
#   alp2_x = x[...,3:5]  # [nt-1, nx, ny, n_ctrl]
#   alp1_y = x[...,5:7]  # [nt-1, nx, ny, n_ctrl]
#   alp2_y = x[...,7:9]  # [nt-1, nx, ny, n_ctrl]
#   Dx_right, Dx_left, Dy_right, Dy_left, Dt_minus_diffusion, x_arr, t_arr, tau, \
#     rho_prev, alp1_x_prev, alp2_x_prev, alp1_y_prev, alp2_y_prev = arg_other
#   Df1_fn = fns_dict.Df1_fn  # [nt-1, nx, ny, n_ctrl], x_arr, t_arr -> [nt-1, nx, ny, n_ctrl]
#   Df2_fn = fns_dict.Df2_fn
#   DL_fn = fns_dict.DL_fn
#   Hf1_fn = fns_dict.Hf1_fn  # [nt-1, nx, ny, n_ctrl], x_arr, t_arr -> [nt-1, nx, ny, n_ctrl, n_ctrl]
#   Hf2_fn = fns_dict.Hf2_fn
#   HL_fn = fns_dict.HL_fn
#   alp_sum = alp1_x + alp2_x + alp1_y + alp2_y
#   rho_prev = rho_prev[..., None]  # [nt-1, nx, ny, 1]
#   Dx_right = Dx_right[..., None]  # [nt-1, nx, ny, 1]
#   Dx_left = Dx_left[..., None]  # [nt-1, nx, ny, 1]
#   Dy_right = Dy_right[..., None]  # [nt-1, nx, ny, 1]
#   Dy_left = Dy_left[..., None]  # [nt-1, nx, ny, 1]
#   Dalp1_x = get_Dalp_2d(Df1_fn, DL_fn, rho, alp1_x, alp_sum, rho_prev, alp1_x_prev, x_arr, t_arr, tau, Dx_right)
#   Dalp2_x = get_Dalp_2d(Df2_fn, DL_fn, rho, alp2_x, alp_sum, rho_prev, alp2_x_prev, x_arr, t_arr, tau, Dx_left)
#   Dalp1_y = get_Dalp_2d(Df1_fn, DL_fn, rho, alp1_y, alp_sum, rho_prev, alp1_y_prev, x_arr, t_arr, tau, Dy_right)
#   Dalp2_y = get_Dalp_2d(Df2_fn, DL_fn, rho, alp2_y, alp_sum, rho_prev, alp2_y_prev, x_arr, t_arr, tau, Dy_left)
#   f1_x, f2_x, f1_y, f2_y = get_f_vals_2d(fns_dict, (alp1_x, alp2_x, alp1_y, alp2_y), x_arr, t_arr)
#   vec = Dt_minus_diffusion[...,None]  # [nt-1, nx, ny, 1]
#   vec -= Dx_right * f1_x[...,None] + Dx_left * f2_x[...,None] + Dy_right * f1_y[...,None] + Dy_left * f2_y[...,None]
#   vec -= fns_dict.numerical_L_fn((alp1_x, alp2_x, alp1_y, alp2_y), x_arr, t_arr)[...,None]  # [nt-1, nx, ny, 1]
#   Drho = -vec - (rho - rho_prev) / tau  # [nt-1, nx, ny, 1]
#   Drho2 = (alp1_x * (alp1_x * rho - alp1_x_prev * rho_prev) + alp2_x * (alp2_x * rho - alp2_x_prev * rho_prev) \
#           + alp1_y * (alp1_y * rho - alp1_y_prev * rho_prev) + alp2_y * (alp2_y * rho - alp2_y_prev * rho_prev)) / tau  # [nt-1, nx, ny, n_ctrl]
#   Drho2 = jnp.sum(Drho2, axis = -1, keepdims = True)  # [nt-1, nx, ny, 1]
#   Drho -= Drho2  # [nt-1, nx, ny, 1]
#   diff_ret = jnp.concatenate([Drho, Dalp1_x, Dalp2_x, Dalp1_y, Dalp2_y], axis = -1)  # [nt-1, nx, ny, 4 * n_ctrl + 1]
#   # compute Hessian
#   Dalp1xDrho = get_DalpDrho_2d(Df1_fn, DL_fn, rho, alp1_x, alp_sum, rho_prev, alp1_x_prev, x_arr, t_arr, tau, Dx_right)  # [nt-1, nx, ny, n_ctrl]
#   Dalp2xDrho = get_DalpDrho_2d(Df2_fn, DL_fn, rho, alp2_x, alp_sum, rho_prev, alp2_x_prev, x_arr, t_arr, tau, Dx_left)  # [nt-1, nx, ny, n_ctrl]
#   Dalp1yDrho = get_DalpDrho_2d(Df1_fn, DL_fn, rho, alp1_y, alp_sum, rho_prev, alp1_y_prev, x_arr, t_arr, tau, Dy_right)  # [nt-1, nx, ny, n_ctrl]
#   Dalp2yDrho = get_DalpDrho_2d(Df2_fn, DL_fn, rho, alp2_y, alp_sum, rho_prev, alp2_y_prev, x_arr, t_arr, tau, Dy_left)  # [nt-1, nx, ny, n_ctrl]
#   rho = rho[..., None]  # [nt-1, nx, ny, 1, 1]
#   Dx_right = Dx_right[..., None]  # [nt-1, nx, ny, 1, 1]
#   Dx_left = Dx_left[..., None]  # [nt-1, nx, ny, 1, 1]
#   Dy_right = Dy_right[..., None]  # [nt-1, nx, ny, 1, 1]
#   Dy_left = Dy_left[..., None]  # [nt-1, nx, ny, 1, 1]
#   D2alp1_x = get_D2alp_2d(Hf1_fn, HL_fn, rho, alp1_x, alp_sum, x_arr, t_arr, tau, Dx_right)  # [nt-1, nx, ny, n_ctrl, n_ctrl]
#   D2alp2_x = get_D2alp_2d(Hf2_fn, HL_fn, rho, alp2_x, alp_sum, x_arr, t_arr, tau, Dx_left)  # [nt-1, nx, ny, n_ctrl, n_ctrl]
#   D2alp1_y = get_D2alp_2d(Hf1_fn, HL_fn, rho, alp1_y, alp_sum, x_arr, t_arr, tau, Dy_right)  # [nt-1, nx, ny, n_ctrl, n_ctrl]
#   D2alp2_y = get_D2alp_2d(Hf2_fn, HL_fn, rho, alp2_y, alp_sum, x_arr, t_arr, tau, Dy_left)  # [nt-1, nx, ny, n_ctrl, n_ctrl]
#   Dalp1Dalp2 = -HL_fn(alp_sum, x_arr, t_arr) * rho  # [nt-1, nx, ny, n_ctrl, n_ctrl]
#   D2rho = -(alp1_x **2 + alp2_x **2 + alp1_y **2 + alp2_y **2) / tau  # [nt-1, nx, ny, nctrl]
#   D2rho = jnp.sum(D2rho, axis = -1, keepdims = True) - 1/tau  # [nt-1, nx, ny, 1]
#   Hess_rho = jnp.concatenate([D2rho, Dalp1xDrho, Dalp2xDrho, Dalp1yDrho, Dalp2yDrho], axis = -1)  # [nt-1, nx, ny, 4 * n_ctrl + 1]
#   Hess_alp1_x = jnp.concatenate([Dalp1xDrho[...,None], D2alp1_x, Dalp1Dalp2, Dalp1Dalp2, Dalp1Dalp2], axis = -1)  # [nt-1, nx, ny, n_ctrl, 4 * n_ctrl + 1]
#   Hess_alp2_x = jnp.concatenate([Dalp2xDrho[...,None], Dalp1Dalp2, D2alp2_x, Dalp1Dalp2, Dalp1Dalp2], axis = -1)  # [nt-1, nx, ny, n_ctrl, 4 * n_ctrl + 1]
#   Hess_alp1_y = jnp.concatenate([Dalp1yDrho[...,None], Dalp1Dalp2, Dalp1Dalp2, D2alp1_y, Dalp1Dalp2], axis = -1)  # [nt-1, nx, ny, n_ctrl, 4 * n_ctrl + 1]
#   Hess_alp2_y = jnp.concatenate([Dalp2yDrho[...,None], Dalp1Dalp2, Dalp1Dalp2, Dalp1Dalp2, D2alp2_y], axis = -1)  # [nt-1, nx, ny, n_ctrl, 4 * n_ctrl + 1]
#   Hess_ret = jnp.concatenate([Hess_rho[...,None,:], Hess_alp1_x, Hess_alp2_x, Hess_alp1_y, Hess_alp2_y], axis = -2)  # [nt-1, nx, ny, 4 * n_ctrl + 1, 4 * n_ctrl + 1]
#   print('diff_ret: ', diff_ret)
#   print('Hess_ret: ', Hess_ret)
#   return diff_ret, Hess_ret


# @partial(jax.jit, static_argnames=("fns_dict",))
# def get_diff_in_Newton_1d(x, arg_other, fns_dict):
#   rho = x[...,0:1]  # [nt-1, nx, 1]
#   alp1_x = x[...,1:2]  # [nt-1, nx, 1]
#   alp2_x = x[...,2:3]  # [nt-1, nx, 1]
#   Dx_right, Dx_left, _, _, Dt_minus_diffusion, x_arr, t_arr, tau, \
#     rho_prev, alp1_x_prev, alp2_x_prev, _, _ = arg_other
#   Df_fn = fns_dict.Df_fn  # [nt-1, nx, 1], x_arr, t_arr -> [nt-1, nx, 1]
#   DL_fn = fns_dict.DL_fn  # [nt-1, nx, 1], x_arr, t_arr -> [nt-1, nx, 1]
#   Hf_fn = fns_dict.Hf_fn  # [nt-1, nx, 1], x_arr, t_arr -> [nt-1, nx, 1, 1]
#   HL_fn = fns_dict.HL_fn  # [nt-1, nx, 1], x_arr, t_arr -> [nt-1, nx, 1, 1]
#   alp_sum = alp1_x + alp2_x
#   rho_prev = rho_prev[..., None]  # [nt-1, nx, 1]
#   Dx_right = Dx_right[..., None]  # [nt-1, nx, 1]
#   Dx_left = Dx_left[..., None]  # [nt-1, nx, 1]
#   Dalp1_x = get_Dalp_1d(Df_fn, DL_fn, rho, alp1_x, alp_sum, rho_prev, alp1_x_prev, x_arr, t_arr, tau, Dx_right)
#   Dalp2_x = get_Dalp_1d(Df_fn, DL_fn, rho, alp2_x, alp_sum, rho_prev, alp2_x_prev, x_arr, t_arr, tau, Dx_left)
#   f1_x, f2_x = get_f_vals_1d(fns_dict, (alp1_x, alp2_x,), x_arr, t_arr)  # [nt-1, nx]
#   vec = Dt_minus_diffusion[...,None]  # [nt-1, nx, 1]
#   vec -= Dx_right * f1_x[...,None] + Dx_left * f2_x[...,None]
#   vec -= fns_dict.numerical_L_fn((alp1_x, alp2_x,), x_arr, t_arr)[...,None]  # [nt-1, nx, 1]
#   Drho = -vec - (rho - rho_prev) / tau  # [nt-1, nx, 1]
#   Drho -= (alp1_x * (alp1_x * rho - alp1_x_prev * rho_prev) + alp2_x * (alp2_x * rho - alp2_x_prev * rho_prev)) / tau  # [nt-1, nx, 1]
#   diff_ret = jnp.concatenate([Drho, Dalp1_x, Dalp2_x], axis = -1)  # [nt-1, nx, 3]
#   # compute Hessian
#   Dalp1xDrho = get_DalpDrho_1d(Df_fn, DL_fn, rho, alp1_x, alp_sum, rho_prev, alp1_x_prev, x_arr, t_arr, tau, Dx_right)  # [nt-1, nx, ny, n_ctrl]
#   Dalp2xDrho = get_DalpDrho_1d(Df_fn, DL_fn, rho, alp2_x, alp_sum, rho_prev, alp2_x_prev, x_arr, t_arr, tau, Dx_left)  # [nt-1, nx, ny, n_ctrl]
#   rho = rho[..., None]  # [nt-1, nx, 1, 1]
#   Dx_right = Dx_right[..., None]  # [nt-1, nx, 1, 1]
#   Dx_left = Dx_left[..., None]  # [nt-1, nx, 1, 1]
#   D2alp1_x = get_D2alp_1d(Hf_fn, HL_fn, rho, alp1_x, alp_sum, x_arr, t_arr, tau, Dx_right)  # [nt-1, nx, 1,1]
#   D2alp2_x = get_D2alp_1d(Hf_fn, HL_fn, rho, alp2_x, alp_sum, x_arr, t_arr, tau, Dx_left)  # [nt-1, nx, 1,1]
#   Dalp1Dalp2 = -HL_fn(alp_sum, x_arr, t_arr) * rho  # [nt-1, nx, 1, 1]
#   D2rho = -(alp1_x **2 + alp2_x **2) / tau  # [nt-1, nx, 1]
#   D2rho = jnp.sum(D2rho, axis = -1, keepdims = True) - 1/tau  # [nt-1, nx, 1]
#   eps = 1e-6
#   Hess_rho = jnp.concatenate([D2rho + eps, Dalp1xDrho, Dalp2xDrho], axis = -1)  # [nt-1, nx, 3]
#   Hess_alp1_x = jnp.concatenate([Dalp1xDrho[...,None], D2alp1_x + eps, Dalp1Dalp2], axis = -1)  # [nt-1, nx, 1, 3]
#   Hess_alp2_x = jnp.concatenate([Dalp2xDrho[...,None], Dalp1Dalp2, D2alp2_x + eps], axis = -1)  # [nt-1, nx, 1, 3]
#   Hess_ret = jnp.concatenate([Hess_rho[...,None,:], Hess_alp1_x, Hess_alp2_x], axis = -2)  # [nt-1, nx, 3, 3]
#   # print('diff_ret shape: ', diff_ret.shape)
#   # print('Hess_ret shape: ', Hess_ret.shape)
#   # print('Hessian: ', Hess_ret)
#   # print('Hessian determinant: ', jnp.min(jnp.abs(jnp.linalg.det(Hess_ret))))
#   return diff_ret, Hess_ret

# @partial(jax.jit, static_argnames=("fns_dict",))
# def proj_rhoalp_2d(fns_dict, x0, arg_other):
#   rho = x0[...,0:1]
#   alp1_x = x0[...,1:3]
#   alp2_x = x0[...,3:5]
#   alp1_y = x0[...,5:7]
#   alp2_y = x0[...,7:9]
#   Dx_right, Dx_left, Dy_right, Dy_left, Dt_minus_diffusion, x_arr, t_arr, tau, \
#     rho_prev, alp1_x_prev, alp2_x_prev, alp1_y_prev, alp2_y_prev = arg_other
#   rho = jnp.maximum(rho, 0.0)
#   f1_alp1_x = fns_dict.f_fn(alp1_x, x_arr, t_arr)[...,0:1]  # [nt-1, nx, ny, 1]
#   alp1_x *= (f1_alp1_x >= 0.0)
#   f1_alp2_x = fns_dict.f_fn(alp2_x, x_arr, t_arr)[...,0:1]  # [nt-1, nx, ny, 1]
#   alp2_x *= (f1_alp2_x < 0.0)
#   f2_alp1_y = fns_dict.f_fn(alp1_y, x_arr, t_arr)[...,1:2]  # [nt-1, nx, ny, 1]
#   alp1_y *= (f2_alp1_y >= 0.0)
#   f2_alp2_y = fns_dict.f_fn(alp2_y, x_arr, t_arr)[...,1:2]  # [nt-1, nx, ny, 1]
#   alp2_y *= (f2_alp2_y < 0.0)
#   x_ret = jnp.concatenate([rho, alp1_x, alp2_x, alp1_y, alp2_y], axis = -1)
#   return x_ret

# @partial(jax.jit, static_argnames=("fns_dict",))
# def proj_rhoalp_1d(fns_dict, x0, arg_other):
#   rho = x0[...,0:1]
#   alp1_x = x0[...,1:2]
#   alp2_x = x0[...,2:3]
#   Dx_right, Dx_left, Dy_right, Dy_left, Dt_minus_diffusion, x_arr, t_arr, tau, \
#     rho_prev, alp1_x_prev, alp2_x_prev, alp1_y_prev, alp2_y_prev = arg_other
#   rho = jnp.maximum(rho, 0.0)
#   f1_alp1_x = fns_dict.f_fn(alp1_x, x_arr, t_arr)  # [nt-1, nx, 1]
#   alp1_x *= (f1_alp1_x >= 0.0)
#   f1_alp2_x = fns_dict.f_fn(alp2_x, x_arr, t_arr)  # [nt-1, nx, 1]
#   alp2_x *= (f1_alp2_x < 0.0)
#   x_ret = jnp.concatenate([rho, alp1_x, alp2_x], axis = -1)  # [nt-1, nx, 3]
#   return x_ret

# def Newton_iter_2d(fns_dict, x0, args_other, N_max = 10):
#   for i in range(N_max):
#     Df, D2f = get_diff_in_Newton_2d(x0, args_other, fns_dict)
#     x1 = x0 - jnp.linalg.solve(D2f, Df)
#     x0 = proj_rhoalp_2d(fns_dict, x1, args_other)
#   return x0

# def Newton_iter_1d(fns_dict, x0, args_other, N_max = 10, eps = 1e-6):
#   for i in range(N_max):
#     Df, D2f = get_diff_in_Newton_1d(x0, args_other, fns_dict)
#     x1 = x0 - jnp.linalg.solve(D2f, Df)
#     x_prev = x0  # [nt-1, nx, 3]
#     x0 = proj_rhoalp_1d(fns_dict, x1, args_other)
#     err = jnp.sum((x0 - x_prev) ** 2, axis = (0,1)) / jnp.sum(x0 ** 2, axis = (0,1))
#     err = jnp.sum(err)
#     # err = jnp.max(jnp.abs(x0 - x_prev))
#     if err < eps:
#       print('Newton iter ', i, ' converged, err = ', err)
#       break
#   return x0

def get_f_vals_1d(f_fn, alp, x_arr, t_arr):
  ''' @ parameters:
      f_fn: a function taking in alp, x_arr, t_arr, returning [nt-1, nx, ndim]
      alp: tuple of alp1, alp2, each term is [nt-1, nx, n_ctrl]
      x_arr: vec that can be broadcasted to [nt-1, nx]
      t_arr: vec that can be broadcasted to [nt-1, nx]
    @ return:
      f_val: tuple of f1, f2, each term is [nt-1, nx]
  '''
  alp1, alp2 = alp  # [nt-1, nx, n_ctrl]
  f1 = f_fn(alp1, x_arr, t_arr)[...,0]  # [nt-1, nx]
  f1 = f1 * (f1 >= 0.0)  # [nt-1, nx]
  f2 = f_fn(alp2, x_arr, t_arr)[...,0]  # [nt-1, nx]
  f2 = f2 * (f2 < 0.0)  # [nt-1, nx]
  return f1, f2

def get_f_vals_2d(f_fn, alp, x_arr, t_arr):
  ''' @ parameters:
      f_fn: a function taking in alp, x_arr, t_arr, returning [nt-1, nx, ny, ndim]
      alp: tuple of alp1_x, alp2_x, alp1_y, alp2_y, each term is [nt-1, nx, ny, n_ctrl]
      x_arr: vec that can be broadcasted to [nt-1, nx, ny]
      t_arr: vec that can be broadcasted to [nt-1, nx, ny]
    @ return:
      f_val: tuple of f11, f12, f21, f22, each term is [nt-1, nx, ny]
  '''
  alp1_x, alp2_x, alp1_y, alp2_y = alp  # [nt-1, nx, ny, n_ctrl]
  f1_x = f_fn(alp1_x, x_arr, t_arr)[...,0]  # [nt-1, nx, ny]
  f1_x = f1_x * (f1_x >= 0.0)  # [nt-1, nx, ny]
  f2_x = f_fn(alp2_x, x_arr, t_arr)[...,0]  # [nt-1, nx, ny]
  f2_x = f2_x * (f2_x < 0.0)  # [nt-1, nx, ny]
  f1_y = f_fn(alp1_y, x_arr, t_arr)[...,1]  # [nt-1, nx, ny]
  f1_y = f1_y * (f1_y >= 0.0)  # [nt-1, nx, ny]
  f2_y = f_fn(alp2_y, x_arr, t_arr)[...,1]  # [nt-1, nx, ny]
  f2_y = f2_y * (f2_y < 0.0)  # [nt-1, nx, ny]
  return f1_x, f2_x, f1_y, f2_y

def compute_HJ_residual_1d(phi, alp, dt, dspatial, fns_dict, epsl, x_arr, t_arr, bc):
  dx = dspatial[0]
  L_val = fns_dict.numerical_L_fn(alp, x_arr, t_arr)
  f1, f2 = get_f_vals_1d(fns_dict.f_fn, alp, x_arr, t_arr)
  vec = Dt_decreasedim(phi, dt) - epsl * Dxx_decreasedim(phi, dx, bc)  # [nt-1, nx]
  vec -= Dx_right_decreasedim(phi, dx, bc) * f1 + Dx_left_decreasedim(phi, dx, bc) * f2
  vec -= L_val
  return vec

def compute_HJ_residual_2d(phi, alp, dt, dspatial, fns_dict, epsl, x_arr, t_arr, bc):
  dx, dy = dspatial
  bc_x, bc_y = bc
  L_val = fns_dict.numerical_L_fn(alp, x_arr, t_arr)  # [nt-1, nx, ny]
  Dx_right_phi = Dx_right_decreasedim(phi, dx, bc_x)  # [nt-1, nx, ny]
  Dx_left_phi = Dx_left_decreasedim(phi, dx, bc_x)  # [nt-1, nx, ny]
  Dy_right_phi = Dy_right_decreasedim(phi, dy, bc_y)  # [nt-1, nx, ny]
  Dy_left_phi = Dy_left_decreasedim(phi, dy, bc_y)  # [nt-1, nx, ny]
  f1_x, f2_x, f1_y, f2_y = get_f_vals_2d(fns_dict.f_fn, alp, x_arr, t_arr)
  vec = Dt_decreasedim(phi, dt) - epsl * Dxx_decreasedim(phi, dx, bc_x) - epsl * Dyy_decreasedim(phi, dy, bc_y)  # [nt-1, nx, ny]
  vec -= Dx_right_phi * f1_x + Dx_left_phi * f2_x + Dy_right_phi * f1_y + Dy_left_phi * f2_y
  vec -= L_val
  return vec

def compute_cont_residual_1d(rho, alp, dt, dspatial, fns_dict, c_on_rho, epsl, x_arr, t_arr, bc):
  dx = dspatial[0]
  eps = 1e-4
  f1, f2 = get_f_vals_1d(fns_dict.f_fn, alp, x_arr, t_arr)
  m1 = (rho + eps) * f1  # [nt-1, nx]
  m2 = (rho + eps) * f2  # [nt-1, nx]
  res = Dt_increasedim(rho,dt) + epsl * Dxx_increasedim(rho,dx, bc) # [nt, nx]
  res -= Dx_left_increasedim(m1, dx, bc) + Dx_right_increasedim(m2, dx, bc)
  res = jnp.concatenate([res[:-1,...], res[-1:,...] + c_on_rho/dt], axis = 0)
  return res

def compute_cont_residual_2d(rho, alp, dt, dspatial, fns_dict, c_on_rho, epsl, x_arr, t_arr, bc):
  dx, dy = dspatial
  bc_x, bc_y = bc
  eps = 1e-4
  f1_x, f2_x, f1_y, f2_y = get_f_vals_2d(fns_dict.f_fn, alp, x_arr, t_arr)
  m1_x = (rho + eps) * f1_x  # [nt-1, nx, ny]
  m2_x = (rho + eps) * f2_x  # [nt-1, nx, ny]
  m1_y = (rho + eps) * f1_y  # [nt-1, nx, ny]
  m2_y = (rho + eps) * f2_y  # [nt-1, nx, ny]
  res = Dt_increasedim(rho,dt) + epsl * Dxx_increasedim(rho,dx,bc_x) + epsl * Dyy_increasedim(rho,dy,bc_y)  # [nt, nx, ny]
  res -= Dx_left_increasedim(m1_x, dx, bc_x) + Dx_right_increasedim(m2_x, dx, bc_x) \
          + Dy_left_increasedim(m1_y, dy, bc_y) + Dy_right_increasedim(m2_y, dy, bc_y)
  res = jnp.concatenate([res[:-1,...], res[-1:,...] + c_on_rho/dt], axis = 0)
  return res


def update_rho_1d(rho_prev, phi, alp, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr, bc, fv):
  vec = compute_HJ_residual_1d(phi, alp, dt, dspatial, fns_dict, epsl, x_arr, t_arr, bc)
  rho_next = rho_prev + sigma * vec
  rho_next = jnp.maximum(rho_next, 0.0)  # [nt-1, nx]
  return rho_next

def update_alp_1d(alp_prev, phi, rho, sigma, dspatial, fns_dict, x_arr, t_arr, bc, eps=1e-4):
  dx = dspatial[0]
  Dx_right_phi = Dx_right_decreasedim(phi, dx, bc)  # [nt-1, nx]
  Dx_left_phi = Dx_left_decreasedim(phi, dx, bc)  # [nt-1, nx]
  if 'alp_update_fn' in fns_dict._fields:
    alp_next = fns_dict.alp_update_fn(alp_prev, Dx_right_phi, Dx_left_phi, rho, sigma, x_arr, t_arr)
  else:
    raise NotImplementedError
  return alp_next

def update_rho_2d(rho_prev, phi, alp, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr, bc, fv):
  vec = compute_HJ_residual_2d(phi, alp, dt, dspatial, fns_dict, epsl, x_arr, t_arr, bc)
  rho_next = rho_prev + sigma * vec
  rho_next = jnp.maximum(rho_next, 0.0)  # [nt-1, nx, ny]
  return rho_next

def update_alp_2d(alp_prev, phi, rho, sigma, dspatial, fns_dict, x_arr, t_arr, bc, eps=1e-4):
  dx, dy = dspatial
  bc_x, bc_y = bc
  Dx_right_phi = Dx_right_decreasedim(phi, dx, bc_x)  # [nt-1, nx, ny]
  Dx_left_phi = Dx_left_decreasedim(phi, dx, bc_x)  # [nt-1, nx, ny]
  Dy_right_phi = Dy_right_decreasedim(phi, dy, bc_y)  # [nt-1, nx, ny]
  Dy_left_phi = Dy_left_decreasedim(phi, dy, bc_y)  # [nt-1, nx, ny]
  Dphi = (Dx_right_phi, Dx_left_phi, Dy_right_phi, Dy_left_phi)
  if 'alp_update_fn' in fns_dict._fields:
    alp_next = fns_dict.alp_update_fn(alp_prev, Dphi, rho, sigma, x_arr, t_arr)
  else:
    raise NotImplementedError
  return alp_next

@partial(jax.jit, static_argnames=("fns_dict", "Ct", "bc"))
def update_primal_1d(phi_prev, rho_prev, c_on_rho, alp_prev, tau, dt, dspatial, fns_dict, fv, epsl, x_arr, t_arr, bc,
                     C = 1.0, pow = 1, Ct = 1):
  delta_phi = compute_cont_residual_1d(rho_prev, alp_prev, dt, dspatial, fns_dict, c_on_rho, epsl, x_arr, t_arr, bc)
  phi_next = phi_prev + tau * H1_precond_1d(delta_phi, fv, dt, bc, C = C, pow = pow, Ct = Ct)
  return phi_next

@partial(jax.jit, static_argnames=("fns_dict", "bc"))
def update_primal_2d(phi_prev, rho_prev, c_on_rho, alp_prev, tau, dt, dspatial, fns_dict, fv, epsl, x_arr, t_arr, bc, 
                     C = 1.0, pow = 1, Ct = 1):
  delta_phi = compute_cont_residual_2d(rho_prev, alp_prev, dt, dspatial, fns_dict, c_on_rho, epsl, x_arr, t_arr, bc)
  phi_next = phi_prev + tau * H1_precond_2d(delta_phi, fv, dt, bc, C = C)  # NOTE: pow and Ct are not implemented
  return phi_next


@partial(jax.jit, static_argnames=("fns_dict", "ndim", "bc"))
def update_dual_oneiter(phi_bar, rho_prev, c_on_rho, alp_prev, sigma, dt, dspatial, epsl, x_arr, t_arr, bc, fns_dict, ndim, fv):
  if ndim == 1:
    update_alp = update_alp_1d
    update_rho = update_rho_1d
  elif ndim == 2:
    update_alp = update_alp_2d
    update_rho = update_rho_2d
  else:
    raise NotImplementedError
  alp_next = update_alp(alp_prev, phi_bar, rho_prev, sigma, dspatial, fns_dict, x_arr, t_arr, bc)
  rho_next = update_rho(rho_prev, phi_bar, alp_next, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr, bc, fv)
  err = jnp.sum((rho_next - rho_prev) ** 2) / jnp.sum(rho_next ** 2)  # scalar
  for alp_p, alp_n in zip(alp_prev, alp_next):
    err += jnp.sum((alp_n - alp_p) ** 2) / jnp.sum(alp_n ** 2)  # scalar
  return rho_next, alp_next, err

def update_dual_alternative(phi_bar, rho_prev, c_on_rho, alp_prev, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr, ndim, bc, fv,
                   rho_alp_iters=10, eps=1e-7):
  '''
  @ parameters:
  fns_dict: dict of functions, see the function set_up_example_fns in set_fns.py
  '''
  for j in range(rho_alp_iters):
    rho_next, alp_next, err = update_dual_oneiter(phi_bar, rho_prev, c_on_rho, alp_prev, sigma, dt, dspatial, epsl,
                                                              x_arr, t_arr, bc, fns_dict, ndim, fv)
    if err < eps:
      break
    rho_prev = rho_next
    alp_prev = alp_next
  return rho_next, alp_next

# def update_dual_Newton_1d(phi_bar, rho_prev, c_on_rho, alp_prev, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr, ndim,
#                    rho_alp_iters=10, eps=1e-6):
#   '''
#   @ parameters:
#   fns_dict: dict of functions, see the function set_up_example_fns in set_fns.py
#   '''
#   for j in range(rho_alp_iters):
#     rho_next, alp_next, err = update_dual_oneiter(phi_bar, rho_prev, c_on_rho, alp_prev, sigma, dt, dspatial, epsl,
#                                                               x_arr, t_arr, fns_dict, ndim)
#     # if err < eps:
#     #   print('alternative iter ', j, ' converged, err = ', err)
#     #   break
#     rho_prev = rho_next
#     alp_prev = alp_next
#   # Newton  
#   # alp1_x_prev, alp2_x_prev = alp_prev
#   # x0 = jnp.concatenate([rho_prev[...,None], alp1_x_prev, alp2_x_prev], axis = -1)  # [nt-1, nx, 3]
#   # dx = dspatial[0]
#   # Dx_right = Dx_right_decreasedim(phi_bar, dx)  # [nt-1, nx]
#   # Dx_left = Dx_left_decreasedim(phi_bar, dx)  # [nt-1, nx]
#   # Dt_minus_diffusion = Dt_decreasedim(phi_bar, dt) - epsl * Dxx_decreasedim(phi_bar, dx)
#   # args_other = (Dx_right, Dx_left, None, None, Dt_minus_diffusion, x_arr, t_arr, sigma, rho_prev, alp1_x_prev, alp2_x_prev, None, None)
#   # New_ret = Newton_iter_1d(fns_dict, x0, args_other, eps = eps)
#   # rho_next = New_ret[...,0]
#   # alp1_x_next = New_ret[...,1:2]
#   # alp2_x_next = New_ret[...,2:3]
#   # alp_next = (alp1_x_next, alp2_x_next)
#   return rho_next, alp_next

# def update_dual_Newton_2d(phi_bar, rho_prev, c_on_rho, alp_prev, sigma, dt, dspatial, epsl, fns_dict, x_arr, t_arr, ndim,
#                    rho_alp_iters=10, eps=1e-7):
#   '''
#   @ parameters:
#   fns_dict: dict of functions, see the function set_up_example_fns in set_fns.py
#   '''
#   # for j in range(rho_alp_iters):
#   #   rho_next, alp_next, err = update_dual_oneiter(phi_bar, rho_prev, c_on_rho, alp_prev, sigma, dt, dspatial, epsl,
#   #                                                             x_arr, t_arr, fns_dict, ndim)
#   #   if err < eps:
#   #     break
#   #   rho_prev = rho_next
#   #   alp_prev = alp_next
#   alp1_x_prev, alp2_x_prev, alp1_y_prev, alp2_y_prev = alp_prev
#   x0 = jnp.concatenate([rho_prev[...,None], alp1_x_prev, alp2_x_prev, alp1_y_prev, alp2_y_prev], axis = -1)
#   dx, dy = dspatial
#   Dx_right = Dx_right_decreasedim(phi_bar, dx)  # [nt-1, nx, ny]
#   Dx_left = Dx_left_decreasedim(phi_bar, dx)  # [nt-1, nx, ny]
#   Dy_right = Dy_right_decreasedim(phi_bar, dy)  # [nt-1, nx, ny]
#   Dy_left = Dy_left_decreasedim(phi_bar, dy)  # [nt-1, nx, ny]
#   Dt_minus_diffusion = Dt_decreasedim(phi_bar, dt) - epsl * Dxx_decreasedim(phi_bar, dx) - epsl * Dyy_decreasedim(phi_bar, dy)
#   args_other = (Dx_right, Dx_left, Dy_right, Dy_left, Dt_minus_diffusion, x_arr, t_arr, sigma, rho_prev, alp1_x_prev, alp2_x_prev, alp1_y_prev, alp2_y_prev)
#   New_ret = Newton_iter_2d(fns_dict, x0, args_other)
#   rho_next = New_ret[...,0]
#   alp1_x_next = New_ret[...,1:3]
#   alp2_x_next = New_ret[...,3:5]
#   alp1_y_next = New_ret[...,5:7]
#   alp2_y_next = New_ret[...,7:9]
#   return rho_next, (alp1_x_next, alp2_x_next, alp1_y_next, alp2_y_next)
  