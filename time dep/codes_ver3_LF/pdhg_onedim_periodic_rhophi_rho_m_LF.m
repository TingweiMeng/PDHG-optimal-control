% this function uses PDHG to solve
% inf_{rho,m,mu} sup_{phi} sum_{k=1,...,nt-1} m_{k,i}(-phi_{a,i+1}+phi_{a,i})*dt/dx -rho_{k,i}(phi_{k+1,i}-phi_{k,i}) +M*dt/dx*rho_{k,i}(phi_{a,i+1}+phi_{a,i-1}-2phi_{a,i}) + dt*(rho_{k,i}+rho_{k,i+1} + 2c)/2*H^*(2m_{k,i}/(rho_{k,i}+rho_{k,i+1} + 2c)) - sum_i mu_i(phi_{1,i}-g_i)
%       a is k if fwd, k+1 if bwd
% phi0 size: (nt,nx)
% rho0 and m0 size: (nt-1,nx)
% mu0 size: (1,nx)
% c_on_rho: the constant c
function [phi, error_all, phi_next, rho_next, mu_next] = pdhg_onedim_periodic_rhophi_rho_m_LF(H_fn, dL_fn, phi0, rho0, m0, mu0, stepsz_param, M, if_fwd, g, dx, dt, c_on_rho)
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

if if_fwd ~= 1
    % M is negative for bwd
    M = -abs(M);
end

pdhg_param = 1;
error_all = zeros(N_maxiter, 2);

tau = stepsz_param / (2*dt/dx + 3 + 4 * abs(M)* dt / dx);
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

% <rho, A3 phi> = sum_{i, k=1,...,nt-1} M*dt/dx*rho_{k,i}(phi_{a,i+1}+phi_{a,i-1}-2phi_{a,i})
val_3_1 = M*dt/dx + 0*ind_phi_a_iplus1;
ind_output_3 = [ind_rho_k_i(:); ind_rho_k_i(:); ind_rho_k_i(:)];
ind_input_3 = [ind_phi_a_iplus1(:); ind_phi_a_i(:); ind_phi_a_iminus1(:)];
val_3 = [val_3_1(:); -2* val_3_1(:); val_3_1(:)];  % note: the second value is -2*val_3_1, and the third val is val_3_1
A3 = sparse(ind_output_3, ind_input_3, val_3, (nt-1)*nx, nt*nx);

A23 = A2 + A3;

for i = 1: N_maxiter
    % update phi: 
    % sup_{phi} <m, A1 phi> + <rho, A23 phi> - sum_i mu_i * phi_{1,i} - |phi - phi^l|^2/(2*tau)
    % phi^{l+1} = phi^l  + tau* (A1' m + A23' rho) (- tau * mu if k = 1)
    phi_next = phi_prev + tau * reshape(A1' * m_prev(:) + A23' * rho_prev(:), [nt, nx]);
    phi_next(1,:) = phi_next(1,:) - tau * mu_prev;
    
    % extrapolation
    phi_bar = phi_next + pdhg_param * (phi_next - phi_prev);
    
    % update mu
    % inf_{mu} sum_i mu_i *(g_i- phi_{1,i}) + |mu - mu^l|^2/(2*sigma)
    mu_next = mu_prev + sigma * (phi_bar(1,:) - g);
    % update rho and m using gd:
    % compute (rho_{k,i} + rho_{k,i+1})/2 using previous rho
    rho_ave = (rho_prev + [rho_prev(:,2:end), rho_prev(:,1)])/2 + c_on_rho;
    % compute (nabla H^*)(2m^l_{k,i}/(rho^l_{k,i}+rho^l_{k,i+1} + 2c))
    % NOTE: if dL is not differentiable, using subgrad descent: 
    %       dL returns proj_{a + dL}(0) - a, where a is the second argument
    A1_mul_phibar = reshape(A1 * phi_bar(:), [nt-1, nx]);
    A23_mul_phibar = reshape(A23 * phi_bar(:), [nt-1, nx]);
    dL_prev = dL_fn(m_prev./rho_ave, A1_mul_phibar/dt);
%     % inf_{rho, m} <m, A1 phi> + <rho, A23 phi> + dt*(rho_{k,i}+rho_{k,i+1}+2c)/2 * H^*(2m_{k,i}/(rho_{k,i}+rho_{k,i+1}+2c)) + |rho - rho^l|/2sigma + |m-m^l|/2sigma
%     % approximate it by
%     % inf_{m} <m, A1 phi> + dt* (nabla H^*)(2m^l_{k,i}/(rho^l_{k,i}+rho^l_{k,i+1}+2c)) * m_{k,i} + |m-m^l|/2sigma
%     m_next = m_prev - tau * (A1_mul_phibar + dt * dL_prev);
    % when H is |.|^2/2
    m_next = (m_prev - tau * A1_mul_phibar)./(tau./rho_ave + 1);
    % inf_{rho} <rho, A23 phi> + dt*(-H(nabla H^*(m^l_{k,i}/rho_ave_{k,i}))/2 - H(nabla H^*(m^l_{k,i-1}/rho_ave_{k,i-1}))/2) * rho_{k,i} + |rho - rho^l|/2sigma
    rho_next = rho_prev - sigma * (A23_mul_phibar - H_fn(dL_prev)/2 - H_fn([dL_prev(:,end), dL_prev(:,1:end-1)])/2);

    % compute errors
    err1 = max(norm(phi_next(:) - phi_prev(:)), norm([rho_next(:); m_next(:); mu_next(:)] - [rho_prev(:); m_prev(:); mu_prev(:)]));
    % err2: equation error
    err2 = (phi_next(2:end,:) - phi_next(1:end-1,:)) / dt;
    if if_fwd
        err2 = err2 + H_fn(phi_next(1:end-1,:));
    else
        err2 = err2 + H_fn(phi_next(2:end,:));
    end
    err2_l1 = mean(abs(err2(:)));
    
    error_all(i, 1) = err1;
    error_all(i, 2) = err2_l1;

    if err1 < eps || sum(isnan(phi_next(:))) > 0 || sum(isnan(rho_next(:))) > 0 || sum(isnan(m_next(:))) > 0
        break;
    end

    if mod(i, 10000) == 0
        fprintf('iteration %d, error with prev step %f, equation error %f\n', i, err1, err2_l1);
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
