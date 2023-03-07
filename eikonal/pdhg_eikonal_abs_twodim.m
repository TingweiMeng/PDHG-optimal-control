
% dx is the grid size in x direction
% dy is the grid size in y direction
% fx size: nx * 2 (bdry condition on top and bottom sides of the rectangular domain)
% fy size: ny * 2 (bdry condition on left and right sides of the rectangular domain)
% phi0 size: nx * ny
% phi0(1,1) is the upper left corner, phi0(nx, ny) is the bottom right corner
% phi0(nx,1) is upper right, phi0(1,ny) is left bottom
function phi = pdhg_eikonal_abs_twodim(fx, fy, phi0, dx, dy)
N_maxiter = 2000;
eps = 1e-8;

[nx,ny] = size(phi0);

% m1 size (nx - 1, ny), is the first coordinate of m
% m2 size (nx, ny-1), is the second coordinate of m
m1_prev = (phi0(2:end, :) - phi0(1:end-1, :))/dx;
m2_prev = (phi0(:, 2:end) - phi0(:, 1:end-1))/dy;
% phi size (nx, ny)
phi_prev = phi0;
% lamx size (nx,2), lamy size (ny,2)
lamx_prev = [m2_prev(:,1), -m2_prev(:,end)];
lamy_prev = [-m1_prev(1,:)', m1_prev(end,:)'];

pdhg_param = 1;
error_all = zeros(N_maxiter, 3);

tau = 0.5;
sigma = tau;

% initialize A1, A2
A = eye(nx) + diag(-ones(1,nx-1), 1);
A = A(1:end-1,:);  % size: (nx - 1, nx)
A1 = zeros(nx-1, ny, nx, ny);
for i = 1:ny
    A1(:,i,:,i) = reshape(A, [nx-1, 1, nx, 1]) * dy;
end
A1 = reshape(A1, [(nx-1)*ny, nx * ny]);

A = eye(ny) + diag(-ones(1,ny-1), 1);
A = A(1:end-1,:);  % size: (ny-1, ny)
A2 = zeros(nx, ny-1, nx, ny);
for i = 1:nx
    A2(i,:,i,:) = reshape(A, [1, ny-1, 1, ny]) * dx;
end
% note: y-axis direction is reversed wrt the index, so we have neg sign
A2 = reshape(-A2, [(ny-1)*nx, nx * ny]);

% pde for m: div(m) = m_pde_rhs
m_pde_rhs = -1.0;

for i = 1: N_maxiter
    % update m: y = m_prev - tau*A*phi, 
    %           where A is the negative discrete grad op scaled by dxdy
    y1 = m1_prev - tau * reshape(A1 * phi_prev(:), [nx-1, ny]);
    y2 = m2_prev - tau * reshape(A2 * phi_prev(:), [nx, ny-1]);
%     % two norm
%     % m_next_ij = argmin_m dxdy*tau |m|_2 + |m-y_{ij}|^2/2
%     y_norm = sqrt([y1; zeros(1,ny)].^2 + [y2, zeros(nx,1)].^2);  % add zeros
%     m_scaling = max(y_norm - dx*dy*tau, 0) ./ y_norm;
%     m1_next = m_scaling(1:end-1,:) .* y1;
%     m2_next = m_scaling(:,1:end-1) .* y2;
    % one norm
    m1_next_pos = max(y1 - dx * dy * tau, 0);
    m1_next_neg = min(y1 + dx * dy * tau, 0);
    m1_next = m1_next_neg + m1_next_pos;
    m2_next_pos = max(y2 - dx * dy * tau, 0);
    m2_next_neg = min(y2 + dx * dy * tau, 0);
    m2_next = m2_next_neg + m2_next_pos;
    % lam_next = lam + tau * (f - phi_prev)*d
    lamx_next = lamx_prev + tau * dx * (fx - phi_prev(:, [1,end]));
    lamy_next = lamy_prev + tau * dy * (fy - phi_prev([1,end], :)');

%         % accerleration step
%         pdhg_param = 1/sqrt(1 + 2*mu * sigma);
%         tau = tau * pdhg_param;
%         sigma = sigma / pdhg_param;

    % extrapolation
    m1_bar = m1_next + pdhg_param * (m1_next - m1_prev);
    m2_bar = m2_next + pdhg_param * (m2_next - m2_prev);
    lamx_bar = lamx_next + pdhg_param * (lamx_next - lamx_prev);
    lamy_bar = lamy_next + pdhg_param * (lamy_next - lamy_prev);
    
    % update phi:
    % phi_{k+1} = phi_k + sigma(A'*m_bar + d *lam_bar (note this index issue))
    %                   - sigma * c * dxdy
    phi_next = phi_prev + sigma * reshape(A1' * m1_bar(:) + A2' * m2_bar(:),[nx,ny]);
    phi_next(:,[1,end]) = phi_next(:,[1,end]) + sigma * dx * lamx_bar;
    phi_next([1,end],:) = phi_next([1,end],:) + sigma * dy * lamy_bar';
    phi_next = phi_next - m_pde_rhs * sigma * dx * dy;
    
    err1 = max([norm(phi_next - phi_prev), norm(m1_next - m1_prev), norm(m2_next - m2_prev), ...
        norm(lamx_next - lamx_prev), norm(lamy_next - lamy_prev)]);
    % err2: pde error
    dphidx = reshape(A1 * phi_next(:) /dx/dy, [nx-1, ny]);
    dphidy = reshape(A2 * phi_next(:) /dx/dy, [nx, ny-1]);
    % 2-norm
%     eikonal_pde_err = sqrt(dphidx(:,1:end-1).^2 + dphidy(1:end-1,:).^2) - 1;
    % 1-norm
    eikonal_pde_err = max(abs(dphidx(:, 1:end-1)), abs(dphidy(1:end-1,:))) - 1;
    eikonal_bdry_err_x = phi_next(:,[1,end]) - fx;
    eikonal_bdry_err_y = phi_next([1,end],:)' - fy;
    eikonal_err = max(norm(eikonal_pde_err(:)), norm([eikonal_bdry_err_x(:); eikonal_bdry_err_y(:)]));
    m_pde_err = (m1_next(2:end, 2:end-1) - m1_next(1:end-1, 2:end-1))/dx + ...
        (m2_next(2:end-1, 2:end) - m2_next(2:end-1, 1:end-1))/dy - m_pde_rhs;
    m_bdry_err_x = [m2_next(:,1), -m2_next(:,end)] - lamx_next;
    m_bdry_err_y = [-m1_next(1,:)', m1_next(end,:)'] - lamy_next;
    m_err = max(norm(m_pde_err), norm([m_bdry_err_x(:); m_bdry_err_y(:)]));
    err2 = max(eikonal_err, m_err);
    % err3: check primal-dual gap (ignore indicator fns)
    primal = dx*dy* (sum(abs(m1_next(:))) + sum(abs(m2_next(:)))) ...
        - dx * sum(lamx_next(:) .*fx(:)) - dy * sum(lamy_next(:) .*fy(:));  % function of m and lam
    dual = - m_pde_rhs * sum(phi_next(:)) * dx * dy;
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

    m1_prev = m1_next;
    m2_prev = m2_next;
    lamx_prev = lamx_next;
    lamy_prev = lamy_next;
    phi_prev = phi_next;
end
phi = phi_next;

figure; semilogy(error_all(1:i, 1));
figure; semilogy(error_all(1:i, 2));
figure; semilogy(abs(error_all(1:i, 3)));

% grad_phi = (phi(2:end) - phi(1:end-1))/dx;
% sign_m = sign(m_next);
% figure; plot(grad_phi); hold on;
% plot(sign_m);

figure; surf(m1_next); title('m1');
figure; surf(m2_next); title('m2');
end