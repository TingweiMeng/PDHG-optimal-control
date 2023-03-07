% check a solution is true or not
% dH is the gradient of H, L is Lagrangian, J is initial condition
function err = check_HJ_sol_usingLO(phi, dx, x_grid, t_grid, dH, L, J)
% using finite difference to approx derivative
dphidx_left = (phi - [phi(:, end), phi(:, 1:end-1)])/dx;
dphidx_right = ([phi(:, 2:end), phi(:, 1)] - phi)/dx;
% u = x - t v, v = nabla H (dphidx)
% assume H is half quad: Burgers
v_right = dH(dphidx_right);
v_left = dH(dphidx_left);
u_right = x_grid - t_grid .* v_right;
u_left = x_grid - t_grid .* v_left;
% phi_LO = min_u J(u) + tH^*((x-u)/t)
phi_LO = min(J(u_left) + t_grid.* L(v_left), J(u_right) + t_grid.* L(v_right));
err = phi - phi_LO;
fprintf('HJ solution err max: %f\n', max(abs(err(:))));
figure; contourf(err); colorbar; title('phi LO minus phi');
end



