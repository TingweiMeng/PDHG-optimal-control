% this function uses PDHG to solve
% inf_{rho,m,mu} sup_{phi} sum_{k=1,...,nt-1} m_{k,i}(-phi_{a,i+1}+phi_{a,i})*dt/dx -rho_{k,i}(phi_{k+1,i}-phi_{k,i}) + dt*F_rho^*(m) - sum_i mu_i(phi_{1,i}-g_i)
%       a is k if fwd, k+1 if bwd; % F_rho(p) = -(rho_i+c) p if p<=0; (rho_{i+1} +c) p if p>0
% phi0 size: (nt,nx)
% rho0 and m0 size: (nt-1,nx)
% mu0 size: (1,nx)
% c_on_rho: the constant c
function [phi, error_all, phi_next, rho_next, mu_next] = pdhg_onedim_periodic_rho_m_EO_L1(phi0, rho0, m0, mu0, stepsz_param, if_fwd, g, dx, dt, c_on_rho)
N_maxiter = 1e7; %1e7;
eps = 1e-6;

[nt,nx] = size(phi0);
g = reshape(g(:), [1,nx]);

% phi size (nt,nx)
phi_prev = phi0;
% rho and m size (nt-1,nx)
rho_prev = rho0;
m_prev = m0;
% mu size (1,nx)
mu_prev = mu0;

pdhg_param = 1;
error_all = zeros(N_maxiter, 3);

tau = stepsz_param / (2*dt/dx + 3);
sigma = tau;

% define the matrix K
row_k_1_to_ntm1 = repmat((1:nt-1)', [1,nx]);  % each element is the first index for (k,i), k=1,...,nt-1
col_i = repmat((1:nx), [nt-1,1]);  % each element is the second index for (k,i), k=1,...,nt-1
col_i_plus_one = [col_i(:,2:end), col_i(:,1)];
col_i_minus_one = [col_i(:,end), col_i(:,1:end-1)];

ind_rho_k_i = sub2ind([nt-1, nx], row_k_1_to_ntm1(:), col_i(:));  % k from 1 to nt-1, size [nt-1,nx]
ind_phi_k_i = sub2ind([nt, nx], row_k_1_to_ntm1(:), col_i(:));  % k from 1 to nt-1, but size is [nt, nx]

if if_fwd  % fwd, a = k
    ind_a = row_k_1_to_ntm1;
else  % bwd, a = k+1
    ind_a = row_k_1_to_ntm1 + 1;
end

ind_phi_a_iplus1 = sub2ind([nt, nx], ind_a(:), col_i_plus_one(:));  % (a,i+1)
ind_phi_a_i = sub2ind([nt, nx], ind_a(:), col_i(:));  % (a,i)
ind_phi_a_iminus1 = sub2ind([nt, nx], ind_a(:), col_i_minus_one(:));  % (a,i-1)

% <m, A1 phi> = sum_{i, k=1,...,nt-1} m_{k,i}(-phi_{a,i+1}+phi_{a,i})*dt/dx
val_1_1 = -dt/dx + 0*ind_phi_a_iplus1;
ind_output_1 = [ind_rho_k_i(:); ind_rho_k_i(:)];
ind_input_1 = [ind_phi_a_iplus1(:); ind_phi_a_i(:)];
val_1 = [val_1_1(:); - val_1_1(:)];  % note: the second value is negative of val_1_1
A1 = sparse(ind_output_1, ind_input_1, val_1, (nt-1)*nx, nt*nx);

% <rho, A2 phi> = sum_{i, k=1,...,nt-1} (-rho_{k,i}(phi_{k+1,i}-phi_{k,i}))
ind_phi_2_1 = sub2ind([nt, nx], row_k_1_to_ntm1(:) + 1, col_i(:));  % (k+1,i)
val_2_1 = -1 + 0*ind_phi_2_1;
ind_phi_2_2 = sub2ind([nt, nx], row_k_1_to_ntm1(:), col_i(:));  % (k,i)
ind_output_2 = [ind_rho_k_i(:); ind_rho_k_i(:)];
ind_input_2 = [ind_phi_2_1(:); ind_phi_2_2(:)];
val_2 = [val_2_1(:); -val_2_1(:)];  % note: the second value is negative of val_2_1
A2 = sparse(ind_output_2, ind_input_2, val_2, (nt-1)*nx, nt*nx);

rho_candidates = zeros(nt-1, nx, 5);
for i = 1: N_maxiter
    % update phi: 
    % sup_{phi} <m, A1 phi> + <rho, A2 phi> - sum_i mu_i * phi_{1,i} - |phi - phi^l|^2/(2*tau)
    % phi^{l+1} = phi^l  + tau* (A1' m + A2' rho) (- tau * mu if k = 1)
    phi_next = phi_prev + tau * reshape(A1' * m_prev(:) + A2' * rho_prev(:), [nt, nx]);
    phi_next(1,:) = phi_next(1,:) - tau * mu_prev;
    
    % extrapolation
    phi_bar = phi_next + pdhg_param * (phi_next - phi_prev);
    
    % update mu
    % inf_{mu} sum_i mu_i *(g_i- phi_{1,i}) + |mu - mu^l|^2/(2*sigma)
    mu_next = mu_prev + sigma * (phi_bar(1,:) - g);
    % update rho and m:
    % F(rho, p) = -(rho_i+c) p if p<=0; (rho_{i+1} +c) p if p>0
    % min_{rho,m} <m, A1 phi> + <rho, A2 phi> + dt*F^*(m) + |rho - rho^l|^2/2sigma + |m-m^l|^2/2sigma
    % m = truncation of m_prev - sigma * A1* phi_bar between -(rho_i+c) and rho_{i+1}+c
    % plug this minimizer m into the original problem, it becomes:
    % min_rho <rho, A2 phi> + |rho - rho^l|^2/2sigma + G(rho)^2/2sigma,
    %   let a = m_prev - sigma * A1* phi_bar
    %   if a_i < 0, G(rho)_i = rho_i + c + a_i if rho_i+c in [0,-a_i]; 0 o.w.
    %   if a_i >= 0, G(rho)_i = rho_{i+1} + c - a_i if rho_{i+1}+c in [0,a_i]; 0 o.w.
    % we have several candidates for rho_i + c:  (we do projection s.t. rho+c >=0)
    A2_mul_phi = A2 * phi_bar(:);
    vec1 = m_prev - sigma * reshape(A1 * phi_bar(:), [nt-1, nx]);  % a
    vec1_left = [vec1(:,end), vec1(:,1:end-1)];  % a_{i-1}
    vec2 = rho_prev - sigma * reshape(A2_mul_phi, [nt-1, nx]);
    rho_candidates(:,:,1) = - c_on_rho;  % left bound
    % two possible quadratic terms on G, 4 combinations
    rho_candidates(:,:,2) = max(vec2, - c_on_rho);  % for rho large, G = 0
    rho_candidates(:,:,3) = max((vec2 - c_on_rho - vec1)/2, - c_on_rho);  % if G_i = rho_i + c + a_i
    rho_candidates(:,:,4) = max((vec2 - c_on_rho + vec1_left)/2, - c_on_rho);  % if G_{i-1} = rho_i + c - a_{i-1}
    rho_candidates(:,:,5) = max((vec2 - 2*c_on_rho - vec1 + vec1_left)/3, - c_on_rho);  % we have both terms above
    rho_next = get_minimizer_ind(rho_candidates, A2_mul_phi, rho_prev, sigma, c_on_rho, vec1);
    % m is truncation of vec1 into [-(rho_i + c), rho_{i+1}+c]
    m_next = min(max(vec1, -(rho_next + c_on_rho)), [rho_next(:,2:end), rho_next(:,1)] + c_on_rho);

    % primal error
    err1 = norm(phi_next(:) - phi_prev(:));
    % err2: dual error
    err2 = norm([rho_next(:); m_next(:); mu_next(:)] - [rho_prev(:); m_prev(:); mu_prev(:)]);
    % err3: equation error
    HJ_residual = check_HJ_sol_usingEO_L1_1d(phi_next, dt, dx, if_fwd);
    err3 = mean(abs(HJ_residual(:)));

    error_all(i, 1) = err1;
    error_all(i, 2) = err2;
    error_all(i, 3) = err3;

    if err1 < eps && err2 < eps || sum(isnan(phi_next(:))) > 0 || sum(isnan(rho_next(:))) > 0 || sum(isnan(m_next(:))) > 0
        break;
    end

    if mod(i, 10000) == 0
        fprintf('iteration %d, primal error with prev step %f, dual error with prev step %f, eqt error %f\n', i, err1, err2, err3);
    end

    rho_prev = rho_next;
    phi_prev = phi_next;
    m_prev = m_next;
    mu_prev = mu_next;
end
phi = phi_next;

% return only computed error
error_all = error_all(1:i,:);

end


% A2_mul_phi is of size ((nt-1)*nx, 1)
% for each (k,i) index, find min_r r*(A2 phi)_{k,i} + (r - rho_prev_{k,i})^2/2sigma + G(rho)_{k,i}^2/2sigma in candidates
function rho_min = get_minimizer_ind(rho_candidates, A2_mul_phi, rho_prev, sigma, c, a)
[nt_m1, nx, n_candidates] = size(rho_candidates);
% compute <rho, A2 phi> + |rho - rho^l|^2/2sigma + G(rho)^2/2sigma,
rho_reshape = reshape(rho_candidates, [nt_m1 * nx, n_candidates]); % (allpts, n_can)
fn_val = rho_reshape.* A2_mul_phi;  % (allpts, n_can)
fn_val = fn_val + (rho_reshape - rho_prev(:)).^2/2/sigma;
fn_val = fn_val + get_G_from_rho(rho_candidates, c, a) / sigma;
[~,I] = min(fn_val, [], 2, 'linear');
rho_min = reshape(rho_reshape(I), [nt_m1, nx]);
end


% rho size (nt-1, nx, n_can)
% a size (nt-1, nx)
% starting from G = 0  (note: the index for G follows the index for rho)
%   if a_i < 0, G(rho)_i += rho_i + c + a_i if rho_i+c in [0,-a_i]; 0 o.w.
%   if a_{i-1} >= 0, G(rho)_i += rho_{i} + c - a_{i-1} if rho_{i}+c in [0,a_{i-1}]; 0 o.w.
% return G_i.^2/2, size ((nt-1)*nx, n_can)
function fn_val = get_G_from_rho(rho, c, a)
[nt_m1, nx, n_can] = size(rho);
a_left = [a(:, end), a(:, 1:end-1)];
a_rep = repmat(reshape(a, [nt_m1, nx, 1]), [1,1,n_can]);
a_left_rep = repmat(reshape(a_left, [nt_m1, nx, 1]), [1,1,n_can]);
G1 = min(rho + c + a_rep, 0);  % when a < 0
G2 = min(rho + c - a_left_rep, 0);  % when a >=0
G = 0* rho;
G(a_rep < 0) = G1(a_rep < 0);
G(a_left_rep >= 0) = G(a_left_rep >= 0) + G2(a_left_rep >= 0);
fn_val = reshape(G.^2 /2, [nt_m1 * nx, n_can]);  % ((nt-1)*nx, n_can)
end