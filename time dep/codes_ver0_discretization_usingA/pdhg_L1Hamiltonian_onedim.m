
% dx is the spatial grid size, dt is the time grid size
% f size: nx * 1
% g size: nx * 1
% phi0 size: nt * nx
function phi = pdhg_L1Hamiltonian_onedim(f, g, phi0, dx, dt)
N_maxiter = 100;
eps = 1e-4;
dim = 1;

[nt, nx] = size(phi0);

% the function theta of p(:,1:end-1) and p(:,2:end)
theta = @(p) (p(:, 1:end-1) + p(:, 2:end))/2;
d1theta = @(p) (0*p(:, 1:end-1) + 1.0)/2;
d2theta = @(p) (0*p(:, 1:end-1) + 1.0)/2;

% phi size (nt, nx)
phi_prev = phi0;
% rho size (nt, nx)
rho_prev = ones(nt, nx)/ (nx * dx);
% m size (nt, nx-1, dim)
m_prev = ((phi_prev(:, 2:end) - phi_prev(:, 1:end-1))/dx) .* theta(rho_prev);

pdhg_param = 1;
error_all = zeros(N_maxiter, 2);

tau = 0.5;
sigma = tau;

stepsz = 0.1;

% TODO: define K
K1 = zeros(nt, nx, nt, nx);
for k = 1: nt-1
    for i = 1: nx
        K1(k, i, k, i) = dx;
        K1(k+1, i, k, i) = -dx;
    end
end
K1 = reshape(K1, [nt*nx, nt*nx]);
K2 = zeros(nt, nx-1, nt, nx);
for k = 1:nt
    for i = 1:nx-1
        K2(k, i, k, i) = -dt;
        K2(k, i, k, i+1) = dt;
    end
end
K2 = reshape(K2, [nt*(nx-1), nt*nx]);

% shape for phi and rho: nt * nx
% shape for m: nt * (nx-1) * dim
% pi = 1 ?????????

for i = 1: N_maxiter
    % update phi: phi^{i+1} = phi^i - tau * K^T * (rho^i, m^i)
    phi_next = phi_prev - tau * reshape(K1' * rho_prev(:) + K2' * m_prev(:), [nt, nx]);
    
    % extrapolation
    phi_bar = phi_next + pdhg_param * (phi_next - phi_prev);
    
    % update rho and m:
    % m^i = m_prev + sigma K2 * phi_bar
    % m^{i+1} = alp if m^i > alp; m^i if |m^i|\leq alp; -alp if m^i < -alp
    %           elementwisely
    % alp = theta(rho_{i,k} / pi_i, rho_{i+1,k} / pi_{i+1})
    % update rho using proj grad descent
    % rho^{i+1} = (p^i - stepsz * dL)_+
    %   where p^i = rho^i + sigma K1 * phi_bar
    %   dL = -sigma *fi*dx*dt + (0 if k>0, sigma*gi * dx if k=0) 
    %        + sum_l (theta(p_{i,k}/pi_i, p_{i+1,k}/pi_{i+1}) -|m^i_{ikl}|)_+ * d1theta /pi_i
    %        + sum_l (theta(p_{i-1,k}/pi_{i-1}, p_{i,k}/pi_{i}) -|m^i_{i-1,kl}|)_+ * d2theta /pi_i
    p = rho_prev + sigma * reshape(K1 * phi_bar(:), [nt, nx]);
    m_middle = m_prev + sigma * reshape(K2 * phi_bar(:), [nt, nx-1]);
    dL = -sigma * repmat(reshape(f, [1, nx]), [nt, 1]) * dx * dt;
    dL(1,:) = dL(1,:) + sigma * reshape(g, [1, nx]) * dx;
    dL(:,1:end-1) = dL(:,1:end-1) + sum(max(reshape(theta(p), [nt,nx-1,1])- abs(m_middle), 0)...
        .* reshape(d1theta(p), [nt,nx-1,1]), 3);
    dL(:,2:end) = dL(:,2:end) + sum(max(reshape(theta(p), [nt,nx-1,1])- abs(m_middle), 0)...
        .* reshape(d2theta(p), [nt,nx-1,1]), 3);
    rho_next = max(p - stepsz * dL, 0);  % gradient descent and projection
    alp = repmat(reshape(theta(rho_next), [nt, nx-1, 1]), [1,1,dim]);
    m_next = min(max(m_middle, -alp), alp);
    
    err1 = max([norm(phi_next - phi_prev), norm(m_next - m_prev), norm(rho_next - rho_prev)]);
    % err2: HJ pde error
    err2_pde = (phi_next(2:end, 1:end-1) - phi_next(1:end-1,1:end-1))/dt + abs((phi_next(2:end, 2:end) - phi_next(2:end, 1:end-1))/dx);
    err2_bdry = phi_next(1,:) - reshape(g, [1, nx]);
    err2 = max(mean(abs(err2_pde(:))), mean(abs(err2_bdry(:))));
    
    error_all(i, 1) = err1;
    error_all(i, 2) = err2;
    
    if err1 < eps
        break;
    end

    if mod(i, 100) == 0
        fprintf('iteration %d, error with prev step %f, pde error %f\n', i, err1, err2);
    end

    m_prev = m_next;
    rho_prev = rho_next;
    phi_prev = phi_next;
end
phi = phi_next;

figure; semilogy(error_all(1:i, 1)); title('error1');
figure; semilogy(error_all(1:i, 2)); title('error2');

figure; contourf(err2_pde); colorbar; title('error pde');

end