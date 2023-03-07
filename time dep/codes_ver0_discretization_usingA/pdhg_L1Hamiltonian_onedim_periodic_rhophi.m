% this function uses PDHG to solve
% sup_{rho, mu} inf_{phi} int (rho(dphi/dt + H(nabla_x phi))) dxdt + int (phi(x,0)-g(x))mu(x)dx
% where we use backward Euler to discretize dphi/dt, and Godunov scheme for H(nabla_x phi)

% for now, assume H = 1-norm, dim = 1, f = 0

% dx is the spatial grid size, dt is the time grid size
% f size: nx * 1
% g size: nx * 1
% phi0 size: nt * nx
% i is spatial index, and k is the time index

% does not work for g = 0, sin initialization
function [phi, error_all] = pdhg_L1Hamiltonian_onedim_periodic_rhophi(f, g, phi0, dx, dt)
N_maxiter = 10000000;
eps = 1e-6;

[nt, nx] = size(phi0);

% reshape g
g = reshape(g, [1,nx]);

% phi size (nt, nx)
phi_prev = phi0;
phi_next = phi_prev;
% rho size (nt-1, nx)
rho_prev = ones(nt-1, nx);
% mu size (1, nx)
mu_prev = ones(1, nx);

pdhg_param = 1;
error_all = zeros(N_maxiter, 2);


tau = 0.01 / (3 + 3* dt / dx);
sigma = tau;

for i = 1: N_maxiter
    % update phi: phi^{k+1} = phi^k - tau * (K1'(phi^k)^T * rho^k + K2'(phi^k)^T * mu^k)
    [a,b] = compute_leftd_rightd(phi_prev);
    % NOTE: be careful about the boundary case
    % ind1, ind2, ind3 have shape (nt-1)*nx
    ind1 = ((a > 0) | (b > 0)) & (a ~= b);
    ind2 = (a > b) & (a > 0);
    ind3 = (b > a) & (b > 0);
    % vec = rho_{i,k-1} - rho_{i,k} + dt/dx*(-rho_{i,k-1}*ind1_{i,k} + rho_{i-1,k-1}*ind2_{i-1,k} + rho_{i+1,k-1}*ind3_{i+1,k})
    %           for k = 2,...,nt
    rho_mul_ind1 = rho_prev .* ind1;
    rho_mul_ind2 = rho_prev .* ind2;
    rho_mul_ind3 = rho_prev .* ind3;
    vec = rho_prev - [rho_prev(2:end,:); zeros(1,nx)] + dt/dx * (-rho_mul_ind1 + [rho_mul_ind2(:,end), rho_mul_ind2(:,1:end-1)] + [rho_mul_ind3(:,2:end), rho_mul_ind3(:,1)]);
    phi_next(2:end,:) = phi_prev(2:end,:) - tau * vec;
    % for k=1, (K1'(phi^k)^T * rho^k + K2'(phi^k)^T * mu^k) = mu - rho_{k=1}
    phi_next(1,:) = phi_prev(1,:) - tau * (mu_prev - rho_prev(1,:));
    
    % extrapolation
    phi_bar = phi_next + pdhg_param * (phi_next - phi_prev);
    
    % update rho and mu:
    % rho^{k+1} = rho^k + sigma * K1(phi_bar)
    %   where K1(phi)_{i,k} = phi_{i,k+1} - phi_{i,k} + dt * max(a, b, 0) / dx,
    %           a = phi_{i+1,k+1} - phi_{i,k+1}; b = phi_{i-1,k+1} - phi_{i,k+1}
    % mu^{k+1} = mu^k + sigma * (phi_bar_{k=1} - g)
    [a,b] = compute_leftd_rightd(phi_bar);
    H_scaled = max(max(a, b), 0);
    rho_next = rho_prev + sigma * (phi_bar(2:end,:) - phi_bar(1:end-1,:) + dt/dx * H_scaled);
    mu_next = mu_prev + sigma * (phi_bar(1,:) - g);
    
    % compute errors
    err1 = max([norm(phi_next - phi_prev), norm(mu_next - mu_prev), norm(rho_next - rho_prev)]);
    % err2: HJ pde error
    err2_hj = (rho_next - rho_prev) / sigma /dt;
    err2_hj_bdry = (mu_next - mu_prev) / sigma;
    err2_hj_l1 = max(mean(abs(err2_hj(:))), mean(abs(err2_hj_bdry(:))));
    
    error_all(i, 1) = err1;
    error_all(i, 2) = err2_hj_l1;
    
    if err1 < eps
        break;
    end

    if mod(i, 100) == 0
        fprintf('iteration %d, error with prev step %f, hj pde error %f\n', i, err1, err2_hj_l1);
    end

    rho_prev = rho_next;
    phi_prev = phi_next;
    mu_prev = mu_next;
end
phi = phi_next;

figure; semilogy(error_all(1:i, 1)); title('error1');
figure; semilogy(error_all(1:i, 2)); title('error hj');

figure; contourf(err2_hj); colorbar; title('error hj');

end


% a = phi_{i+1, k+1} - phi_{i, k+1}
% b = phi_{i-1, k+1} - phi_{i, k+1}
% for k = 1,...,nt-1
function [a,b] = compute_leftd_rightd(phi)
a = [phi(2:end, 2:end), phi(2:end, 1)] - phi(2:end,:);
b = [phi(2:end, end), phi(2:end, 1:end-1)] - phi(2:end,:);
end