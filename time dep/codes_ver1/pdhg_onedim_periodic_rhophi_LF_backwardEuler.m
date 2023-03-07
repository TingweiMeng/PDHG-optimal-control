% this function uses PDHG to solve
% sup_{rho, mu} inf_{phi} int (rho(dphi/dt + H(nabla_x phi))) dxdt + int (phi(x,0)-g(x))mu(x)dx
% where we use forward Euler to discretize dphi/dt, and LF scheme for H(nabla_x phi)
%   H_flux = H(phi_{i+1} - phi_{i})/dx)/2 + H(phi_i - phi_{i-1})/dx)/2 -M(phi_{i+1}+phi_{i-1}-phi_{i})/dx

% for now, assume H = |.|^2/2 (burgers), dim = 1, f = 0

% dx is the spatial grid size, dt is the time grid size
% f size: nx * 1
% g size: nx * 1
% phi0 size: nt * nx
% i is spatial index, and k is the time index
% Hind = 1 for transport, 2 for burgers
function [phi, error_all] = pdhg_onedim_periodic_rhophi_LF_backwardEuler(f, g, phi0, dx, dt, M, Hind, stepsz_param, rho0, mu0)
N_maxiter = 1e5;

% NOTE: M should be negative for backward Euler
M = -abs(M);

eps = 1e-6;

[nt, nx] = size(phi0);


% reshape g
g = reshape(g, [1,nx]);

% phi size (nt, nx)
phi_prev = phi0;
phi_next = phi_prev;
phi_bar = phi_prev;
% rho size (nt-1, nx)
% rho_prev = ones(nt-1, nx);
rho_prev = rho0;
% mu size (1, nx)
% mu_prev = ones(1, nx);
mu_prev = mu0;

pdhg_param = 1;
error_all = zeros(N_maxiter, 2);

tau = stepsz_param / (3 + 8 * abs(M)* dt / dx);
sigma = tau;

for i = 1: N_maxiter
    % update phi: (in the l-th iteration)
    % for k=2,...,nt-1
    % phi^{l+1}_{k,i} = phi^l_{k,i} - tau * (rho_{k-1,i}-rho_{k,i} - M*dt/dx*(rho_{k-1,i+1}+ rho_{k-1,i-1}-2rho_{k-1,i}) + dt*A)
    % where A = (H'((phi_prev_{k,i} - phi_prev_{k,i-1})/dx)*(rho_{k-1,i-1}+rho_{k-1,i}) + H'((phi_prev_{k,i+1} - phi_prev_{k,i})/dx)*(rho_{k-1,i}+rho_{k-1,i+1}))/2/dx
    % for k=1: phi_{1,i} = phi_prev_{1,i} -tau * (mu_i - rho_{1,i})
    % for k=nt: replace rho_{nt,i} to 0
    A = compute_Kprime(phi_prev(2:end,:), rho_prev, dx, Hind);
    [drho_left, drho_right] = compute_leftd_rightd_centerd(rho_prev);
    phi_next(2:end,:) = phi_prev(2:end,:) - tau * (rho_prev - [rho_prev(2:end,:); zeros(1,nx)] - M*dt/dx*(drho_right - drho_left) + dt*A);
    phi_next(1,:) = phi_prev(1,:) - tau * (mu_prev - rho_prev(1,:));
    
    % extrapolation
    phi_bar = phi_next + pdhg_param * (phi_next - phi_prev);
    
    % update rho and mu: (here phi is phi_bar)
    % rho^{l+1} = rho^l + sigma * HJ_residual
    % where HJ_residual_{k,i} = phi_{k+1,i} - phi_{k,i} + dt * H - M*dt/dx*(phi_{k+1,i+1}+ phi_{k+1,i-1}-2phi_{k+1,i}),
    % and H = ((phi_{k+1,i+1} - phi_{k+1,i})^2 + (phi_{k+1,i} - phi_{k+1,i-1})^2)/4/dx^2
    % mu^{l+1} = mu^l + sigma * (phi_{1,i} - g_i)
    H = Hamiltonian_left_right_average(phi_bar(2:end,:), dx, Hind);
    [dphibar_left, dphibar_right] = compute_leftd_rightd_centerd(phi_bar(2:end,:));
    HJ_residual = phi_bar(2:end,:) - phi_bar(1:end-1,:) + dt * H - M*dt/dx*(dphibar_right - dphibar_left);
    rho_next = rho_prev + sigma * HJ_residual;
    mu_next = mu_prev + sigma * (phi_bar(1,:) - g);


    % compute errors
    err1 = max([norm(phi_next(:) - phi_prev(:)), norm([rho_next(:); mu_next(:)] - [rho_prev(:); mu_prev(:)])]);
    % err2: HJ pde error: compute error using phi_next
    H = Hamiltonian_left_right_average(phi_next(2:end,:), dx, Hind);
    [dphi_left, dphi_right] = compute_leftd_rightd_centerd(phi_next(2:end,:));
    err2_hj = phi_next(2:end,:) - phi_next(1:end-1,:) + dt * H - M*dt/dx*(dphi_right - dphi_left);
    err2_hj_bdry = phi_next(1,:) - g;
    err2_hj_l1 = mean([abs(err2_hj(:)); abs(err2_hj_bdry(:))]);
    
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

% compute H((phi_{k,i+1} - phi_{k,i})/dx) / 2 + H((phi_{k,i} - phi_{k,i-1})/dx) / 2
function H = Hamiltonian_left_right_average(phi, dx, Hind)
[dphi_left, dphi_right] = compute_leftd_rightd_centerd(phi);
if Hind == 1
    % linear: H(p) = p
    H = (dphi_left + dphi_right)/2/dx;
else
    if Hind == 2
        % quadratic Hamiltonian
        H = (dphi_left / dx).^2 /4 + (dphi_right / dx).^2 /4;
    end
end
end


% compute H'((phi0_{k,i} - phi0_{k,i-1})/dx)*(rho_{k-1,i-1}+rho_{k-1,i})/(2dx) - H'((phi0_{k,i+1} - phi0_{k,i})/dx)*(rho_{k,i}+rho_{k,i+1})/(2dx)
function result = compute_Kprime(phi0, rho, dx, Hind)
[dphi_left, dphi_right] = compute_leftd_rightd_centerd(phi0);
rho_sum_left = rho + [rho(:,end), rho(:,1:end-1)];
rho_sum_right = rho + [rho(:,2:end), rho(:,1)];
if Hind == 1
    % linear: H(p) = p
    result = rho_sum_left/2/dx - rho_sum_right/2/dx;
else 
    if Hind == 2
        % quadratic Hamiltonian
        result = dphi_left.* rho_sum_left / (2*dx*dx) - dphi_right.* rho_sum_right / (2*dx*dx);
    end
end
end


% um = phi_{i, k} - phi_{i-1, k}
% up = phi_{i+1, k} - phi_{i, k}
function [um,up] = compute_leftd_rightd_centerd(phi)
um = phi - [phi(:, end), phi(:, 1:end-1)];
up = [phi(:, 2:end), phi(:, 1)] - phi;
end