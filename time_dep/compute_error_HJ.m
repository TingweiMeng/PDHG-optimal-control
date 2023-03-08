function [error, err_l1, err_l1_rel] = compute_error_HJ(phi_pdhg, phi_dense, T)

% compare PDHG solution phi (with) with true dense solution (phi_dense with dx_dense, dt_dense)
[nt,nx] = size(phi_pdhg);
[nt_dense,nx_dense] = size(phi_dense);
phi_true = 0.* phi_pdhg;
if mod(nt_dense - 1, nt - 1) == 0
    % true values are on grid points
    nt_factor = (nt_dense - 1)/ (nt-1);
    nx_factor = nx_dense/ nx;
    phi_true = phi_dense(1:nt_factor:end, 1:nx_factor:end);
else
    x_ind = (1: (nx_dense/ nx): nx_dense);
    dt = T / (nt-1);
    dt_dense = T / (nt_dense - 1);
    t_ind = floor(dt * (0:nt-2)' / dt_dense) + 1;  % not include t = T
    t_grid = repmat((0: dt: T-dt)', [1,nx]);
    t_dense_left = dt_dense * (t_ind -1);
    t_dense_right = dt_dense * (t_ind);
    phi_dense_left = phi_dense(t_ind, x_ind);
    phi_dense_right = phi_dense(t_ind+1, x_ind);
    phi_true(1:end-1,:) = phi_dense_left + (phi_dense_right - phi_dense_left) ./(t_dense_right - t_dense_left) .*(t_grid - t_dense_left);
    phi_true(end,:) = phi_dense(end, x_ind);
end

error = phi_pdhg - phi_true;
err_l1 = mean(abs(error(:)));
err_l1_rel = err_l1/ mean(abs(phi_true(:)));
fprintf('error: %f\n', err_l1);
fprintf('relative error: %f\n', err_l1_rel);
end