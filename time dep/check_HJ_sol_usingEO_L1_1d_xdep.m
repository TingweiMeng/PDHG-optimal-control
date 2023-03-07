% check a solution is true or not
% H is L1, 1-dimensional
function HJ_residual = check_HJ_sol_usingEO_L1_1d_xdep(phi, dt, dx, if_fwd, f_in_H, c_in_H)
dphidx_left = (phi - [phi(:, end), phi(:, 1:end-1)])/dx;
dphidx_right = ([phi(:, 2:end), phi(:, 1)] - phi)/dx;
H_val = max(-dphidx_right, 0) + max(dphidx_left, 0);
H_val = c_in_H.* H_val + f_in_H;

if if_fwd  % forward Euler EO residual
    HJ_residual = (phi(2:end,:) - phi(1:end-1,:))/dt + H_val(1:end-1,:);
else
    HJ_residual = (phi(2:end,:) - phi(1:end-1,:))/dt + H_val(2:end,:);
end

% fprintf('HJ solution l1 err: %f\n', mean(abs(HJ_residual(:))));
% figure; contourf(HJ_residual); colorbar; title('HJ residual using EO');
end