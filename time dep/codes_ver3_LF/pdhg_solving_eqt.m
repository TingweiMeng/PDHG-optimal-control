% this function uses PDHG to solve
% inf_{rho} sup_{phi} <rho, f(phi)>

% phi0 size: N*1
% rho0 size: M*1 (M is the dimension of the output of f)
function [phi, error_all, phi_next, phi_bar, rho_next] = pdhg_solving_eqt(f_fn, df_fn, phi0, rho0, tau, sigma)
N_maxiter = 1e6; %1e7;
eps = 1e-6;

% phi size (N,1)
phi_prev = phi0;
% rho size (M,1)
rho_prev = rho0;

pdhg_param = 1;
error_all = zeros(N_maxiter, 2);

% tau = stepsz_param / (3 + 8 * M* dt / dx);
% sigma = tau;

for i = 1: N_maxiter
    % update phi: 
    % sup_phi <rho^l, f(phi)> - |phi - phi^l|^2/(2*tau)
    % approximate it by min_phi -<rho^l, f'(phi^l)phi> + |phi - phi^l|^2/(2*tau)
    % phi^{l+1} = phi^l  + tau* (f'(phi^l))^T rho^l
    phi_next = phi_prev + tau * df_fn(phi_prev)' * rho_prev;
    
    % extrapolation
    phi_bar = phi_next + pdhg_param * (phi_next - phi_prev);
    
    % update rho:
    % inf_{rho} <rho, f(phi_bar)> + |rho - rho^l|^2/(2*sigma)
    % rho^{l+1} = rho^l - sigma * f(phi_bar)
    rho_next = rho_prev - sigma * f_fn(phi_bar);

    % compute errors
    err1 = max([norm(phi_next - phi_prev), norm(rho_next - rho_prev)]);
    % err2: equation error
    err2 = f_fn(phi_next);
    err2_l1 = mean(abs(err2(:)));
    
    error_all(i, 1) = err1;
    error_all(i, 2) = err2_l1;
    
    if err1 < eps || sum(isnan(phi_next(:))) > 0 || sum(isnan(rho_next(:))) > 0
        break;
    end

    if mod(i, 10000) == 0
        fprintf('iteration %d, error with prev step %f, equation error %f\n', i, err1, err2_l1);
    end

    rho_prev = rho_next;
    phi_prev = phi_next;
end
phi = phi_next;

% return only computed error
error_all = error_all(1:i,:);

end
