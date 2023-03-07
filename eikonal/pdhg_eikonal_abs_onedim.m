
% dx is the grid size in interior domain
% f size: n_bdry * 1
% phi0 size: n_grid * 1
function phi = pdhg_eikonal_abs_onedim(f, phi0, dx)
N_maxiter = 100000;
eps = 1e-8;

n_grid = size(phi0,1);
n_brdy = size(f, 1);
bdry_index = [1;n_grid];

% m size (n_grid - 1, 1)
% m_prev = zeros(n_grid-1,1);
m_prev = (phi0(2:end) - phi0(1:end-1))/dx;
% phi size (n_grid, 1)
phi_prev = phi0;
% lambda size (n_bdry, 1)
lam_prev = [-m_prev(1); m_prev(end)];
pdhg_param = 1;
error_all = zeros(N_maxiter, 3);

tau = 0.5;
sigma = tau;

A = eye(n_grid) + diag(-ones(1,n_grid-1), 1);
A = A(1:end-1,:);  % size: (n_grid - 1, n_grid)

% pde for m: div(m) = m_pde_rhs
m_pde_rhs = -1.0;

for i = 1: N_maxiter
    phi_prev_bdry = reshape(phi_prev(bdry_index), [n_brdy,1]);

    % update m: y = m_prev - tau*A*phi, 
    %           where A is the negative discrete grad op (without divided by dx)
    % min_m,lam sum_i dx|m_i|_2 - <grad_phi, m> + <lam, phi-f>
    % m_next = argmin_m dx*tau\sum_i |m_i|_2 + |m-y|^2/2 = shrink_{dx*tau}(y)
    % lam_next = lam + tau * (f - phi_prev)
    y = m_prev - tau * A * phi_prev;
    m_next_pos = max(y- dx * tau, 0);
    m_next_neg = min(y + dx*tau, 0);
    m_next = m_next_neg + m_next_pos;
    lam_next = lam_prev + tau * (f - phi_prev_bdry);

%         % accerleration step
%         pdhg_param = 1/sqrt(1 + 2*mu * sigma);
%         tau = tau * pdhg_param;
%         sigma = sigma / pdhg_param;


    % extrapolation
    m_bar = m_next + pdhg_param * (m_next - m_prev);
    lam_bar = lam_next + pdhg_param * (lam_next - lam_prev);
    
    % update phi:
    % phi_{k+1} = phi_k + sigma(A'*m_bar + lam_bar (note this index issue))
    %                   - sigma * c * dx
    phi_next = phi_prev + sigma * (A' * m_bar);
    phi_next(bdry_index) = phi_next(bdry_index) + sigma * lam_bar;
    % add a linear term wrt phi to avoid the degeneracy of m
    % F^*(phi) = sum phi_i * dx
    phi_next = phi_next - m_pde_rhs * sigma * ones(n_grid, 1) * dx;
    
    err1 = max([norm(phi_next - phi_prev), norm(m_next - m_prev), norm(lam_next - lam_prev)]);
    % err2: pde error
    err2 = norm(abs(A * phi_next / dx) - 1);
    err2 = max(err2, norm(phi_next(bdry_index) - f));
    err2 = max(err2, norm((m_next(2:end) - m_next(1:end-1))/dx - m_pde_rhs));
    err2 = max(err2, norm([-m_next(1); m_next(end)] - lam_next));
    % err3: check primal-dual gap (ignore indicator fns)
    primal = dx* sum(abs(m_next(:))) - lam_next' *f;  % function of m and lam
    dual = - m_pde_rhs * sum(phi_next) * dx;
    err3 = primal - dual;

    error_all(i, 1) = err1;
    error_all(i, 2) = err2;
    error_all(i, 3) = err3;

    if err1 < eps
        break;
    end

    if mod(i, 100) == 0
        fprintf('iteration %d, error with prev step %f, primal-dual err %f\n', i, err1, err3);
    end

    m_prev = m_next;
    lam_prev = lam_next;
    phi_prev = phi_next;
end
phi = phi_next;

figure; semilogy(error_all(1:i, 1));
figure; semilogy(error_all(1:i, 2));
figure; semilogy(abs(error_all(1:i, 3)));

grad_phi = (phi(2:end) - phi(1:end-1))/dx;
sign_m = sign(m_next);
figure; plot(grad_phi); hold on;
plot(sign_m);

figure; plot(m_next); title('m');
end