% check a solution is true or not
% dH is the gradient of H, L is Lagrangian, J is initial condition
function check_HJ_sol_usingLF(phi, dt, dx, H, M)
% using finite difference to approx derivative
dphidx_left = (phi - [phi(:, end), phi(:, 1:end-1)])/dx;
dphidx_right = ([phi(:, 2:end), phi(:, 1)] - phi)/dx;
H_ave = H(dphidx_right) / 2 + H(dphidx_left) / 2;
% forward Euler LF residual
HJ_residual = (phi(2:end,:) - phi(1:end-1,:))/dt + H_ave(1:end-1,:) - M*(dphidx_right(1:end-1,:) - dphidx_left(1:end-1,:));
fprintf('HJ solution err max: %f\n', max(abs(HJ_residual(:))));
figure; contourf(HJ_residual); colorbar; title('HJ residual using LF');
end