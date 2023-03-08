% this function uses PDHG to solve
% inf_{rho,mu} sup_{phi} sum_{k=1,...,nt-1} -rho_{k}(phi_{k+1}-phi_{k})- mu(phi_{1}-g)
% phi0 size: (nt,1)
% rho0 size: (nt-1,1)
% mu0 size: (1,1)
function [phi, error_all, phi_all, rho_all, mu_all] = pdhg_test_rho_m_LF_constx(phi0, rho0, mu0, stepsz_param, g)
N_maxiter = 1e7; %1e7;
eps = 1e-6;

[nt,~] = size(phi0);

% phi size (nt,1)
phi_prev = phi0;
% rho size (nt-1,1)
rho_prev = rho0;
% mu size (1,1)
mu_prev = mu0;

pdhg_param = 1;
error_all = zeros(N_maxiter, 2);
phi_all = zeros(N_maxiter, nt);
rho_all = zeros(N_maxiter, nt-1);
mu_all = zeros(N_maxiter, 1);

dt_div_dx = 0.05;
M = 2;
tau = stepsz_param / (2* dt_div_dx + 3 + 4 * M* dt_div_dx);
sigma = tau;

for i = 1: N_maxiter
    % update phi: 
    % sup_{phi} sum_{k=1,...,nt-1} -rho_{k}(phi_{k+1}-phi_{k})- mu(phi_{1}-g) - |phi - phi^l|^2/(2*tau)
    % phi^{l+1} = phi^l  + tau* (rho - rho_{k+1}) (rho_{k+1} is mu if k=1)
    phi_next = phi_prev + tau * ([rho_prev;0] - [mu_prev; rho_prev]);
    
    % extrapolation
    phi_bar = phi_next + pdhg_param * (phi_next - phi_prev);
    
    % update mu
    % inf_{mu} - mu(phi_{1}-g) + |mu - mu^l|^2/(2*sigma)
    mu_next = mu_prev + sigma * (phi_bar(1) - g);
    % update rho
    % inf_{rho} sum_{k=1,...,nt-1} -rho_{k}(phi_{k+1}-phi_{k}) + |rho - rho^l|/2sigma
    rho_next = rho_prev + sigma * (phi_bar(2:end) - phi_bar(1:end-1));

    % compute errors
    err1 = max(norm(phi_next(:) - phi_prev(:)), norm([rho_next(:); mu_next(:)] - [rho_prev(:); mu_prev(:)]));
    error_all(i, 1) = err1;
    phi_all(i,:) = phi_next(:);
    rho_all(i,:) = rho_next(:);
    mu_all(i,:) = mu_next(:);
    
    if err1 < eps
        break;
    end

    if mod(i, 100) == 0
        fprintf('iteration %d, error with prev step %f\n', i, err1);
    end

    rho_prev = rho_next;
    phi_prev = phi_next;
    mu_prev = mu_next;
end
phi = phi_next;

% return only computed error
error_all = error_all(1:i,:);

end
