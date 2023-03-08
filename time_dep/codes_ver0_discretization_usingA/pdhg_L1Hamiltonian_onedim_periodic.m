
% dx is the spatial grid size, dt is the time grid size
% f size: nx * 1
% g size: nx * 1
% rho_tilde: terminal condition for rho, size nx * 1
% phi0 size: nt * nx
function [phi, error_all] = pdhg_L1Hamiltonian_onedim_periodic(f, g, rho_tilde, phi0, dx, dt)
N_maxiter = 1000000;
eps = 1e-6;
dim = 1;
eps_m = 1e-6;
c = -10;  % the constant added in theta
% shift terminal of rho such that rho + c > 0
rho_tilde = rho_tilde + max(- c, 0);

% viscosity coeff
gamma = 0.0;

[nt, nx] = size(phi0);

% the function theta of p and [p(:,2:end), p(:,1)]
% pi is the uniform distribution, whose element value is 1/(nx * dx)
% pi_inv = (nx * dx);
pi_inv = 1;
% mean: (rho1 + rho2)/2
theta = @(p) (p + [p(:, 2:end), p(:,1)])/2 * pi_inv + c;
d1theta = @(p) (0*p + 1.0)/2 * pi_inv;
d2theta = @(p) (0*p + 1.0)/2 * pi_inv;

% % log mean: (a-b)/(log a - log b) [problem: what happened to a==b?]
% theta = @(p) pi_inv * (p - [p(:, 2:end), p(:,1)])./ (log(p) - log([p(:, 2:end), p(:,1)]));
% d1theta = @(p) pi_inv*(1./ (log(p) - log([p(:, 2:end), p(:,1)])) - (p-[p(:, 2:end), p(:,1)])./p./ (log(p) - log([p(:, 2:end), p(:,1)])).^2);
% d2theta = @(p) pi_inv*(1./ (log([p(:, 2:end), p(:,1)]) - log(p)) - ([p(:, 2:end), p(:,1)] - p)./[p(:, 2:end), p(:,1)]./ (log(p) - log([p(:, 2:end), p(:,1)])).^2);

% % Geometric mean: sqrt(a*b)
% theta = @(p) sqrt(p.*[p(:, 2:end), p(:,1)]) * pi_inv;
% d1theta = @(p) sqrt([p(:, 2:end), p(:,1)]./ p)/2 * pi_inv;
% d2theta = @(p) sqrt(p./[p(:, 2:end), p(:,1)])/2 * pi_inv;


% modify phi0's initial condition
phi0(1,:) = reshape(g, [1,nx]);
% phi size (nt, nx)
phi_prev = phi0;
% phi_tilde size (1,nx)
phi_tilde_prev = phi0(end,:);
% rho size (nt, nx)
rho_prev = repmat(reshape(rho_tilde, [1,nx]), [nt,1]);
rho_next = rho_prev;
% i is spatial index, and k is the time index
% m size (nt, nx, dim): m_{ki} = 0 if k = 1
m_prev = zeros(nt,nx,dim);
m_prev(2:end,:,:) = reshape((sign([phi_prev(2:end,2:end), phi_prev(2:end,1)] - phi_prev(2:end,:))/dx) .* theta(rho_prev(1:nt-1,:)), [nt-1,nx,1]);
m_next = m_prev;

pdhg_param = 1;
error_all = zeros(N_maxiter, 4);

tau = 1.2 / (3*dx + 2 *dt + 4 * gamma * dt / dx);
sigma = tau;

stepsz = 0.1;  % stepsz for gd when updating rho

% define K
% K1 is the matrix between rho and phi: 
%   <K1 phi, rho> = -sum_{i,k=2:end} phi_{ki}(rho_{ki} - rho_{k-1,i}) dx
% first two indices are for rho, and the last two are for phi
K1 = zeros(nt, nx, nt, nx);
for k = 2: nt
    for i = 1: nx
        K1(k, i, k, i) = -dx;
        K1(k-1, i, k, i) = dx;
        % viscosity term
        K1(k-1, i, k, i) = K1(k-1, i, k, i) + 2* gamma * dt / dx;
        if i < nx
            K1(k-1, i+1, k, i) = K1(k-1, i+1, k, i) - gamma * dt / dx;
        end
        if i > 1
            K1(k-1, i-1, k, i) = K1(k-1, i-1, k, i) - gamma * dt / dx;
        end
    end
    K1(k-1, 1, k, nx) = K1(k-1, 1, k, nx) - gamma * dt / dx;
    K1(k-1, nx, k, 1) = K1(k-1, nx, k, 1) - gamma * dt / dx;
end
K1 = reshape(K1, [nt*nx, nt*nx]);

% K2 is the matrix between m and phi: 
%   <K2 phi, m> = -sum_{i,k=2:end} phi_{ki}(m_{ki} - m_{k,i-1}) dt
% first two indices are for m, and the last two are for phi
K2 = zeros(nt, nx, nt, nx);
for k = 2:nt
    K2(k, 1, k, 1) = -dt;
    K2(k, nx, k, 1) = dt;
    for i = 2:nx
        K2(k, i, k, i) = -dt;
        K2(k, i-1, k, i) = dt;
    end
end
K2 = reshape(K2, [nt*nx, nt*nx]);

% K3 is the matrix between rho and phi_tilde: 
%   <K3 phi_tilde, rho> = sum_{i} phi_tilde_{i} rho_{end,i} dx
% first two indices are for rho, and the last two are for phi_tilde
K3 = zeros(nt, nx, nx);
for i = 1:nx
    K3(nt, i, i) = dx;
end
K3 = reshape(K3, [nt*nx, nx]);

% shape for phi and rho: nt * nx
% shape for m: nt * nx * dim
% pi = uniform distribution

for i = 1: N_maxiter
    % update phi: phi^{i+1} = phi^i - tau * (K1^T * rho^i + K2^T * m^i)
    phi_next = phi_prev - tau * reshape(K1' * rho_prev(:) + K2' * m_prev(:), [nt, nx]);
    phi_tilde_next = phi_tilde_prev + reshape(- tau * K3' * rho_prev(:) + tau * dx *rho_tilde,[1,nx]);
    
    % extrapolation
    phi_bar = phi_next + pdhg_param * (phi_next - phi_prev);
    phi_tilde_bar = phi_tilde_next + pdhg_param * (phi_tilde_next - phi_tilde_prev);
    
    % update rho and m:
    % a = m_prev + sigma K2 * phi_bar
    % m^{i+1} = alp if a > alp; a if |a|\leq alp; -alp if a < -alp
    %           elementwisely
    % alp = theta(rho_{i,k} / pi_i, rho_{i+1,k} / pi_{i+1})
    % update rho using proj grad descent
    % rho^{i+1} = (b - stepsz * dL)_+
    %   where b = rho^i + sigma K1 * phi_bar + sigma K3 * phi_tilde_bar
    %   dL = -sigma *fi*dx*dt + (0 if k>1, sigma*gi * dx if k=1) 
    %        + sum_l neg(theta(b_{i,k}/pi_i, b_{i+1,k}/pi_{i+1}) -|a_{i,k+1,l}|) * d1theta /pi_i
    %        + sum_l neg(theta(b_{i-1,k}/pi_{i-1}, b_{i,k}/pi_{i}) -|a_{i-1,k+1,l}|) * d2theta /pi_i
    %       (neg means the negative part: neg(x) = -(-x)_+)
    %   dL_{nt} = 0.
    b = rho_prev + sigma * reshape(K1 * phi_bar(:) + K3 * phi_tilde_bar(:), [nt, nx]);
    a = m_prev + sigma * reshape(K2 * phi_bar(:), [nt, nx]);
    dL = -sigma * repmat(reshape(f, [1, nx]), [nt, 1]) * dx * dt;
    dL(1,:) = dL(1,:) + sigma * reshape(g, [1, nx]) * dx;
    dL(1:nt-1,:) = dL(1:nt-1,:) + sum(min(reshape(theta(b(1:nt-1,:)), [nt-1,nx,1])- abs(a(2:nt,:,:)), 0)...
        .* reshape(d1theta(b(1:nt-1,:)), [nt-1,nx,1]), 3);
    tmp = sum(min(reshape(theta(b(1:nt-1,:)), [nt-1,nx,1])- abs(a(2:nt,:,:)), 0)...
        .* reshape(d2theta(b(1:nt-1,:)), [nt-1,nx,1]), 3);
    dL(1:nt-1,2:end) = dL(1:nt-1,2:end) + tmp(:, 1:end-1);
    dL(1:nt-1,1) = dL(1:nt-1,1) + tmp(:, end);
    dL(nt,:) = 0;
    rho_gd = b - stepsz * dL;  % gradient descent
    % projecting to simplex
%     for j = 1: nt
%         rho_next(j,:) = projsplx(rho_gd(j,:)) / (nx * dx);  % projecting to simplex
%     end
    % no projection
    rho_next = rho_gd;
    % project to positive values
%     rho_next = max(rho_gd, 0);
%     rho_next = max(rho_gd, 1e-6);
    alp = repmat(reshape(theta(rho_next(1:nt-1,:)), [nt-1, nx, 1]), [1,1,dim]);
    m_next(2:nt,:,:) = min(max(a(2:nt,:,:), -alp), alp);
    m_next(1,:,:) = 0;
    
    err1 = max([norm(phi_next - phi_prev), norm(phi_tilde_next - phi_tilde_prev), norm(m_next - m_prev), norm(rho_next - rho_prev)]);
    % err2: HJ pde error
    gradx_phi = ([phi_next(2:end, 2:end),phi_next(2:end, 1)] - phi_next(2:end,:))/dx;
    err2_hj = (phi_next(2:end,:) - phi_next(1:end-1,:))/dt;
    err2_hj = err2_hj + abs(gradx_phi).* d1theta(rho_next(1:end-1,:));
    tmp = abs(gradx_phi).* d2theta(rho_next(1:end-1,:));
    err2_hj(:,1) = err2_hj(:,1) + tmp(:,end);
    err2_hj(:,2:end) = err2_hj(:,2:end) + tmp(:,1:end-1);
    err2_hj = err2_hj + repmat(reshape(f, [1,nx]), [nt-1,1]);
    % viscosity term
    err2_hj = err2_hj - gamma * ([phi_next(2:end, 2:end),phi_next(2:end, 1)] + [phi_next(2:end, end), phi_next(2:end, 1:end-1)] - 2*phi_next(2:end,:)) / (dx*dx);
    err2_hj_bdry = phi_next(1,:) - reshape(g, [1, nx]);

    % FP error
    err2_fp = (rho_next(2:end,:) - rho_next(1:end-1,:))/dt + reshape((m_next(2:end,:,:) - [m_next(2:end,end,:), m_next(2:end,1:end-1,:)])/dx, [nt-1,nx,1]);
    err2_fp = err2_fp + gamma * ([rho_next(1:end-1, 2:end),rho_next(1:end-1, 1)] + [rho_next(1:end-1, end), rho_next(1:end-1, 1:end-1)] - 2*rho_next(1:end-1,:)) / (dx*dx);
    err2_fp_bdry = rho_next(end,:) - reshape(rho_tilde, [1,nx]);

    % m error
    % err2_m is nabla H(nabla_x phi) - m/theta(rho)
    m_div_theta = reshape(m_next(2:end,:,:), [nt-1,nx,1]) ./ theta(rho_next(1:nt-1,:));
    gradH_upper = 0* gradx_phi + 1;
    gradH_upper(gradx_phi < -eps_m) = -1;
    gradH_lower = 0* gradx_phi - 1;
    gradH_lower(gradx_phi > eps_m) = 1;
    err2_m_upper = gradH_upper - m_div_theta;
    err2_m_lower = gradH_lower - m_div_theta;
    err2_hj_l1 = max(mean(abs(err2_hj(:))), mean(abs(err2_hj_bdry(:))));
    err2_fp_l1 = max(mean(abs(err2_fp(:))), mean(abs(err2_fp_bdry(:))));
    err2_m_sup = max(max(max(-err2_m_upper(:), 0)), max(max(err2_m_lower(:), 0)));

    error_all(i, 1) = err1;
    error_all(i, 2) = err2_hj_l1;
    error_all(i, 3) = err2_fp_l1;
    error_all(i, 4) = err2_m_sup;
    
%     if err2_hj_l1 < eps
%         break;
%     end

    if err1 < eps
        break;
    end

    if mod(i, 100) == 0
        fprintf('iteration %d, error with prev step %f, pde error %f, %f, %f\n', i, err1, err2_hj_l1, err2_fp_l1, err2_m_sup);
    end

    m_prev = m_next;
    rho_prev = rho_next;
    phi_prev = phi_next;
    phi_tilde_prev = phi_tilde_next;
end
phi = phi_next;

figure; semilogy(error_all(1:i, 1)); title('error1');
figure; semilogy(error_all(1:i, 2)); title('error hj');
figure; semilogy(error_all(1:i, 3)); title('error fp');
figure; semilogy(error_all(1:i, 4)); title('error m');

figure; contourf(err2_hj); colorbar; title('error hj');
figure; contourf(err2_fp); colorbar; title('error fp');


end