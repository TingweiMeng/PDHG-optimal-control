% Hamiltonian is c(x)|x| + f(x)
% this function uses PDHG to solve
% inf_{rho,m,mu} sup_{phi} sum_{k=1,...,nt-1} m_{k,i}(-phi_{a,i+1}+phi_{a,i})*dt/dx -rho_{k,i}(phi_{k+1,i}-phi_{k,i}) - (rho_{k,i}+c)f(xi)dt + dt*F_rho^*(m) - sum_i mu_i(phi_{1,i}-g_i)
%       a is k if fwd, k+1 if bwd; % F_rho(p) = -(rho_i+c)c(xi) p if p<=0; (rho_{i+1} +c)c(x_{i+1}) p if p>0
% phi0 size: (nt,nx)
% rho0 and m0 size: (nt-1,nx)
% mu0, f_in_H, c_in_H size: (1,nx)
% c_on_rho: the constant c
function [phi, error_all, phi_next, rho_next, mu_next] = pdhg_onedim_periodic_rho_m_EO_L1_xdep_fft(f_in_H, c_in_H, phi0, rho0, m0, mu0, stepsz_param, if_fwd, g, dx, dt, c_on_rho, if_precondition)
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

if if_precondition
    tau = stepsz_param;
else
    tau = stepsz_param / (2*dt/dx + 3);
end
sigma = tau;

% scale two parameters
sigma_scale = 1.5;
sigma = sigma * sigma_scale;
tau = tau / sigma_scale;

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


if if_precondition
    % fft for preconditioning
    Lap_vec = zeros(nx, 1);
    Lap_vec(1,1) = -2 / (dx^2);
    Lap_vec(2,1) = 1 / (dx^2);
    Lap_vec(end,1) = 1 / (dx^2);
    fv = fft(Lap_vec);
end

rho_candidates = zeros(nt-1, nx, 5);
for i = 1: N_maxiter
    % update phi: 
    % argmax_{phi} <m, A1 phi> + <rho, A2 phi> - sum_i mu_i * phi_{1,i} - |phi - phi^l|^2/(2*tau)
    % = argmin_phi <(-A1'm - A2'rho), phi> + <mu, phi_1> + |phi - phi^l|^2/(2*tau)
    % phi^{l+1} = phi^l  + tau* (A1' m + A2' rho) (- tau * mu if k = 1)
    delta_phi = -tau * reshape(A1' * m_prev(:) + A2' * rho_prev(:), [nt, nx]);
    delta_phi(1,:) = delta_phi(1,:) + tau* mu_prev;
    if if_precondition
        % (Dxx + Dtt)(phi - phi_prev) = delta_phi
        phi_next = update_phi_preconditioning(delta_phi, phi_prev, fv, dt);
    else
        % no preconditioning
        phi_next = phi_prev - delta_phi;
    end
    
    
    % extrapolation
    phi_bar = phi_next + pdhg_param * (phi_next - phi_prev);
    
    % update mu
    % inf_{mu} sum_i mu_i *(g_i- phi_{1,i}) + |mu - mu^l|^2/(2*sigma)
    mu_next = mu_prev + sigma * (phi_bar(1,:) - g);
    % update rho and m:
    % F(rho, p) = -(rho_i+c) p if p<=0; (rho_{i+1} +c) p if p>0
    % min_{rho,m} <m, A1 phi> + <rho, A2 phi> - dt<rho,f> + dt*F^*(m) + |rho - rho^l|^2/2sigma + |m-m^l|^2/2sigma
    % m = truncation of m_prev - sigma * A1* phi_bar between -(rho_i+c)c(xi) and (rho_{i+1}+c)c(x_{i+1})
    % plug this minimizer m into the original problem, it becomes:
    % min_rho <rho, A2 phi> + |rho - rho^l- sigma * dt*f|^2/2sigma + G(rho)^2/2sigma,
    %   let a = m_prev - sigma * A1* phi_bar
    %   if a_i < 0, G(rho)_i = (rho_i + c)c(xi) + a_i if rho_i+c in [0,-a_i/c(xi)]; 0 o.w.
    %   if a_{i-1} >= 0, G(rho)_i = (rho_i + c)c(xi) - a_{i-1} if rho_i+c in [0,a_{i-1}/c(xi)]; 0 o.w.
    % we have several candidates for rho_i + c:  (we do projection s.t. rho+c >=0)
    A2_mul_phi = A2 * phi_bar(:);
    vec1 = m_prev - sigma * reshape(A1 * phi_bar(:), [nt-1, nx]);  % a
    vec1_left = [vec1(:,end), vec1(:,1:end-1)];  % a_{i-1}
    vec2 = rho_prev - sigma * reshape(A2_mul_phi, [nt-1, nx]) + sigma * f_in_H * dt;
    rho_candidates(:,:,1) = - c_on_rho;  % left bound
    % two possible quadratic terms on G, 4 combinations
    vec3 = -c_in_H.^2 * c_on_rho - vec1 .* c_in_H;
    vec4 = -c_in_H.^2 * c_on_rho + vec1_left .* c_in_H;
    rho_candidates(:,:,2) = max(vec2, - c_on_rho);  % for rho large, G = 0
    rho_candidates(:,:,3) = max((vec2 + vec3)./(1+ c_in_H.^2), - c_on_rho);  % if G_i = (rho_i + c)c(xi) + a_i
    rho_candidates(:,:,4) = max((vec2 + vec4)./(1+ c_in_H.^2), - c_on_rho);  % if G_i = (rho_i + c)c(xi) - a_{i-1}
    rho_candidates(:,:,5) = max((vec2 + vec3 + vec4)./(1+ 2*c_in_H.^2), - c_on_rho);  % we have both terms above
    rho_next = get_minimizer_ind(rho_candidates, vec2, c_on_rho, vec1, c_in_H);
    % m is truncation of vec1 into [-(rho_i+c)c(xi), (rho_{i+1}+c)c(x_{i+1})]
    m_next = min(max(vec1, -(rho_next + c_on_rho) .* c_in_H), ([rho_next(:,2:end), rho_next(:,1)] + c_on_rho).* [c_in_H(:,2:end), c_in_H(:,1)]);

    % primal error
    err1 = norm(phi_next(:) - phi_prev(:));
    % err2: dual error
    err2 = norm([rho_next(:); m_next(:); mu_next(:)] - [rho_prev(:); m_prev(:); mu_prev(:)]);
    % err3: equation error
    HJ_residual = check_HJ_sol_usingEO_L1_1d_xdep(phi_next, dt, dx, if_fwd, f_in_H, c_in_H);
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
% for each (k,i) index, find min_r (r - shift_term)^2 + G(rho)_{k,i}^2 in candidates
function rho_min = get_minimizer_ind(rho_candidates, shift_term, c, a, c_in_H)
[nt_m1, nx, n_candidates] = size(rho_candidates);
% compute |rho - shift_term|^2 + G(rho)^2,
rho_reshape = reshape(rho_candidates, [nt_m1 * nx, n_candidates]); % (allpts, n_can)
fn_val = (rho_reshape - shift_term(:)).^2;  % (allpts, n_can)
fn_val = fn_val + get_Gsq_from_rho((rho_candidates + c).* c_in_H, a);
[~,I] = min(fn_val, [], 2, 'linear');
rho_min = reshape(rho_reshape(I), [nt_m1, nx]);
end


% rho_plus_c_mul_cinH size (nt-1, nx, n_can), whose i-th element denotes (rho_i + c)*c(xi)
% a size (nt-1, nx)
% starting from G = 0  (note: the index for G follows the index for rho)
%   if a_i < 0, G(rho)_i += (rho_i + c)c(xi) + a_i if rho_i+c in [0,-a_i/c(xi)]; 0 o.w.
%   if a_{i-1} >= 0, G(rho)_i += (rho_i + c)c(xi) - a_{i-1} if rho_{i}+c in [0,a_{i-1}/c(xi)]; 0 o.w.
% return G_i.^2, size ((nt-1)*nx, n_can)
function fn_val = get_Gsq_from_rho(rho_plus_c_mul_cinH, a)
[nt_m1, nx, n_can] = size(rho_plus_c_mul_cinH);
a_left = [a(:, end), a(:, 1:end-1)];
a_rep = repmat(reshape(a, [nt_m1, nx, 1]), [1,1,n_can]);
a_left_rep = repmat(reshape(a_left, [nt_m1, nx, 1]), [1,1,n_can]);
G1 = min(rho_plus_c_mul_cinH + a_rep, 0);  % when a < 0
G2 = min(rho_plus_c_mul_cinH - a_left_rep, 0);  % when a >=0
G = 0* rho_plus_c_mul_cinH;
G(a_rep < 0) = G1(a_rep < 0);
G(a_left_rep >= 0) = G(a_left_rep >= 0) + G2(a_left_rep >= 0);
fn_val = reshape(G.^2, [nt_m1 * nx, n_can]);  % ((nt-1)*nx, n_can)
end


% update phi by
% min_phi <v, phi> + |(Dx + Dt)(phi - phi_prev)|^2/2
% this gives 0 = v - Dxx (phi-phi_prev) - Dtt (phi-phi_prev), if 0<t<T
%   0 = v_1 - Dxx(phi-phi_prev)_1 - Dt(phi-phi_prev)_2 /dt  (where Dtf_2 = (f_2-f_1)/dt)
%   0 = v_nt - Dxx(phi-phi_prev)_nt + Dt(phi-phi_prev)_nt /dt  (where Dtf_nt = (f_nt-f_{nt-1})/dt)

% (Dxx + Dtt)(phi - phi_prev) = vec
function phi_next = update_phi_preconditioning(vec, phi_prev, fv, dt)
% Let f = phi-phi_prev, g_k = k-th Fourier of f wrt x
% 0 = \hat v + k^2 g_k - Dtt g_k if 0<t<T

% 0 = tau* v + (Dx + Dt)^T(Dx + Dt)(phi - phi_prev)
% (Dxx + Dtt)(phi - phi_prev) = -tau * v

[nt, nx] = size(phi_prev);
v_Fourier = 0*phi_prev;

F_phi_updates = zeros(nt, nx);
% do fourier transform
for l = 1:nt
    v_Fourier(l,:) =  fft(vec(l,:));
end

phi_fouir_part = zeros(nt, nx);
negative_onesa =(1/(dt^2))*ones(nt-1,1);
negative_onesc =negative_onesa;
for i = 1: nx %j is the corresponding fourier mode
    f =  squeeze(v_Fourier(:,i));
    cc =  (fv(i))- 2/(dt^2);
    thomas_b = cc*ones(nt,1);
    thomas_n = nt;
    s = ThomasAlgorithm(negative_onesa, thomas_b, negative_onesc, f, thomas_n);
    phi_fouir_part(:,i) = s;
end

for l =1:nt
    F_phi_updates(l,:) = ifft(phi_fouir_part(l,:));
end

phi_next = phi_prev + F_phi_updates;
end