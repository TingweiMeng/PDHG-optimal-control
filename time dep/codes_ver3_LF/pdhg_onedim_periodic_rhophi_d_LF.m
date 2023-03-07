% this function uses PDHG to solve
% dphi/dt + H(d)=0, d = nabla_x phi, and phi(x,0)-g(x) = 0
% where we use forward Euler to discretize dphi/dt, and LF scheme for H(nabla_x phi)
%   H_flux = H(d_i)/2 + H(d_{i-1})/2 -M(phi_{i+1}+phi_{i-1}-phi_{i})/dx,
%   d_i = (phi_{i+1} - phi_{i})/dx

% for now, assume dim = 1

% dx is the spatial grid size, dt is the time grid size
% g size: nx * 1
% phi0 size: (nt, nx)
% d0 size: (nt-1, nx)
% rho0 size: (nt * nx + (nt-1)*nx, 1) [NOTE: whether it is 1d or 2d doesn't matter]
% i is spatial index, and k is the time index
% if_forward = 1 if using forward Euler; otherwise backward
% if_square = 1 if using |f|_2^2/2 in saddle pt formula; 2 if using f.^2/2; 0 if using f
function [phi, error_all, phi_next, phi_bar, rho_next] = pdhg_onedim_periodic_rhophi_d_LF(g, phi0, dx, dt, M, H_fn, dH_fn, stepsz_param, rho0, if_forward, if_square)
[nt,nx] = size(phi0);

if if_forward == 1
    % initialize d0_{k,i} = (phi0_{k,i+1} - phi0_{k,i})/dx
    d0 = ([phi0(1:end-1, 2:end), phi0(1:end-1, 1)] - phi0(1:end-1,:))/dx;
else
    % NOTE: M should be negative for backward Euler
    M = -abs(M);
    % initialize d0_{k,i} = (phi0_{k+1,i+1} - phi0_{k+1,i})/dx
    d0 = ([phi0(2:end, 2:end), phi0(2:end, 1)] - phi0(2:end,:))/dx;
end

tau = stepsz_param / (dt*(2 + 4 * abs(M) / dx + 2/dx + 2/dt + abs(M)));
sigma = tau;

% reshape g to row vec
g = reshape(g, [1,nx]);

if if_square == 0
    f_fn = @(phi) f_fn_HJ(phi, H_fn, nx, nt, dx, dt, M, g, if_forward);
    df_fn = @(phi) df_fn_HJ(phi, dH_fn, nx, nt, dx, dt, M, if_forward);
else
    if if_square == 1  % |f|_2^2/2
        f_fn = @(phi) sum(f_fn_HJ(phi, H_fn, nx, nt, dx, dt, M, g, if_forward).^2/2);
        df_fn = @(phi) f_fn_HJ(phi, H_fn, nx, nt, dx, dt, M, g, if_forward)' * df_fn_HJ(phi, dH_fn, nx, nt, dx, dt, M, if_forward);
    else  % f.^2/2
        f_fn = @(phi) f_fn_HJ(phi, H_fn, nx, nt, dx, dt, M, g, if_forward).^2/2;
        df_fn = @(phi) f_fn_HJ(phi, H_fn, nx, nt, dx, dt, M, g, if_forward) .* df_fn_HJ(phi, dH_fn, nx, nt, dx, dt, M, if_forward);
    end
end

% rho0 = zeros(nt,nx);  % combining rho and mu
[phi, error_all, phi_next, phi_bar, rho_next] = pdhg_solving_eqt(f_fn, df_fn, [phi0(:); d0(:)], rho0(:), tau, sigma);

phi = reshape(phi(1:nt*nx), [nt, nx]);
end


% define f for HJ
% the remaining input parameters are H_fn, nx, nt, dx, dt, M, g, if_forward
% output has size (nt * nx, 1)
function output = f_fn_HJ(input, varargin)
H_fn = varargin{1};
nx = varargin{2};
nt = varargin{3};
dx = varargin{4};
dt = varargin{5};
M = varargin{6};
g = varargin{7};
if_forward = varargin{8};
phi_bar = reshape(input(1:nt*nx), [nt, nx]);
d_var = reshape(input(nt*nx+1:end), [nt-1, nx]);
if if_forward == 1  % forward Euler: using phi^k in H and viscosity term
    % phi_{k,i} - phi_{k,i-1} and phi_{k,i+1} - phi_{k,i}
    [dphibar_left, dphibar_right] = compute_leftd_rightd_centerd(phi_bar(1:end-1,:));
else  % backward Euler: using phi^{k+1} in H and viscosity term
    % phi_{k+1,i} - phi_{k+1,i-1} and phi_{k+1,i+1} - phi_{k+1,i}
    [dphibar_left, dphibar_right] = compute_leftd_rightd_centerd(phi_bar(2:end,:));
end

% FOR THE FOLLOWING: a = k if forward Euler; k+1 if backward Euler

% f1 = (phi_{a,i+1} - phi_{a,i})/dx - d_{k,i}
f1 = dphibar_right / dx - d_var;
% f2 = (phi_{k+1,i} - phi_{k,i})/dt + H(d_{k,i})/2 + H(d_{k,i-1})/2 - M/dx*(phi_{a,i+1} + phi_{a,i-1} - 2*phi_{a,i}) 
f2 = (phi_bar(2:end,:) - phi_bar(1:end-1,:))/dt + H_fn(d_var) / 2 + H_fn([d_var(:,end), d_var(:,1:end-1)]) / 2- M/dx*(dphibar_right - dphibar_left);
f3 = phi_bar(1,:) - g;
output = [f1(:)*dt; f2(:)*dt; f3(:)];  % scaling to approximate integration
end


% define df for HJ
% f = [f1; f2; f3]
% (f1)_{k,i} = (phi_{a,i+1} - phi_{a,i})/dx - d_{k,i}
% (f2)_{k,i} = (phi_{k+1,i} - phi_{k,i})/dt + H(d_{k,i})/2 + H(d_{k,i-1})/2 - M(phi_{a,i+1} + phi_{a,i-1} - 2phi_{a,i})/dx
% (f3)_{i} = phi_{1,i} - g_i
% input of f has size nt*nx + (nt-1)*nx, output of f1 has size (nt-1)*nx, output of f2 has size (nt-1)*nx, output of f3 has size nx
% the remaining input parameters are dH_fn, nx, nt, dx, dt, M, if_forward
% output has size (nt*nx + (nt-1)*nx, nt*nx + (nt-1)*nx)
function output = df_fn_HJ(input, varargin)
dH_fn = varargin{1};
nx = varargin{2};
nt = varargin{3};
dx = varargin{4};
dt = varargin{5};
M = varargin{6};
if_forward = varargin{7};

d_var = reshape(input(nt*nx+1:end), [nt-1, nx]);

% all the index for output of f1
row_k_1_to_ntm1 = repmat((1:nt-1)', [1,nx]);  % each element is the first index for (k,i), k=1,...,nt-1
col_i = repmat((1:nx), [nt-1,1]);  % each element is the second index for (k,i), k=1,...,nt-1
col_i_plus_one = [col_i(:,2:end), col_i(:,1)];
col_i_minus_one = [col_i(:,end), col_i(:,1:end-1)];

% the difference between f1 index and phi index is the size: f1 is (nt-1,nx), phi is (nt,nx)
% NOTE: the index for d is the same as f1
ind_f1_k_i = sub2ind([nt-1, nx], row_k_1_to_ntm1(:), col_i(:));
ind_phi_k_i = sub2ind([nt, nx], row_k_1_to_ntm1(:), col_i(:));

if if_forward == 1  % forward Euler: using phi^k in H and viscosity term    
    ind_phi_k_iplusone = sub2ind([nt, nx], row_k_1_to_ntm1(:), col_i_plus_one(:));

    % (f1)_{k,i} = (phi_{k,i+1} - phi_{k,i})/dx - d_{k,i}
    % d(f1)_{k,i} / dphi_{k,i} = -1/dx
    ind_input_1_1 = ind_phi_k_i;
    val_1_1 = 0* ind_input_1_1 - 1/dx;
    % d(f1)_{k,i} / dphi_{k,i+1} = 1/dx
    ind_input_1_2 = ind_phi_k_iplusone;
    val_1_2 = 0* ind_input_1_1 + 1/dx;
    % d(f1)_{k,i} / dd_{k,i} = -1
    ind_input_1_3 = ind_f1_k_i + nt * nx;
    val_1_3 = 0* ind_input_1_1 - 1.0;

    % (f2)_{k,i} = (phi_{k+1,i} - phi_{k,i})/dt + H(d_{k,i})/2 + H(d_{k,i-1})/2 - M(phi_{k,i+1} + phi_{k,i-1} - 2phi_{k,i})/dx
    % d(f2)_{k,i} / dphi_{k,i} = -1/dt + 2M/dx
    ind_input_2_1 = ind_phi_k_i;
    val_2_1 = 0* ind_input_1_1 - 1/dt + 2*M/dx;
    % d(f2)_{k,i} / dphi_{k+1,i} = 1/dt
    ind_input_2_2 = sub2ind([nt, nx], row_k_1_to_ntm1(:) + 1, col_i(:));
    val_2_2 = 0* ind_input_1_1 + 1/dt;
    % d(f2)_{k,i} / dphi_{k,i+1} = d(f2)_{k,i} / dphi_{k,i-1} = -M/dx
    ind_input_2_3 = ind_phi_k_iplusone;
    val_2_3 = 0* ind_input_1_1 - M/dx;
    ind_input_2_4 = sub2ind([nt, nx], row_k_1_to_ntm1(:), col_i_minus_one(:));
    val_2_4 = 0* ind_input_1_1 - M/dx;
    % d(f2)_{k,i} / dd_{k,i} = H'(d_{k,i})/2
    ind_input_2_5 = ind_f1_k_i + nt * nx;
    val_2_5 = dH_fn(d_var) / 2;
    % d(f2)_{k,i} / dd_{k,i-1} = H'(d_{k,i-1})/2
    ind_input_2_6 = sub2ind([nt-1, nx], row_k_1_to_ntm1(:), col_i_minus_one(:)) + nt * nx;
    val_2_6 = dH_fn([d_var(:, end), d_var(:, 1:end-1)]) / 2;
    
else  % backward Euler: using phi^{k+1} in H and viscosity term
    ind_phi_kp1_iplusone = sub2ind([nt, nx], row_k_1_to_ntm1(:) + 1, col_i_plus_one(:));
    ind_phi_kp1_i = sub2ind([nt, nx], row_k_1_to_ntm1(:) + 1, col_i(:));

    % (f1)_{k,i} = (phi_{k+1,i+1} - phi_{k+1,i})/dx - d_{k,i}
    % d(f1)_{k,i} / dphi_{k+1,i} = -1/dx
    ind_input_1_1 = ind_phi_kp1_i;
    val_1_1 = 0* ind_input_1_1 - 1/dx;
    % d(f1)_{k,i} / dphi_{k+1,i+1} = 1/dx
    ind_input_1_2 = ind_phi_kp1_iplusone;
    val_1_2 = 0* ind_input_1_1 + 1/dx;
    % d(f1)_{k,i} / dd_{k,i} = -1
    ind_input_1_3 = ind_f1_k_i + nt * nx;
    val_1_3 = 0* ind_input_1_1 - 1.0;

    % (f2)_{k,i} = (phi_{k+1,i} - phi_{k,i})/dt + H(d_{k,i})/2 + H(d_{k,i-1})/2 - M(phi_{k+1,i+1} + phi_{k+1,i-1} - 2phi_{k+1,i})/dx
    % d(f2)_{k,i} / dphi_{k,i} = -1/dt
    ind_input_2_1 = ind_phi_k_i;
    val_2_1 = 0* ind_input_1_1 - 1/dt;
    % d(f2)_{k,i} / dphi_{k+1,i} = 1/dt + 2M/dx
    ind_input_2_2 = sub2ind([nt, nx], row_k_1_to_ntm1(:) + 1, col_i(:));
    val_2_2 = 0* ind_input_1_1 + 1/dt + 2*M/dx;
    % d(f2)_{k,i} / dphi_{k+1,i+1} = d(f2)_{k,i} / dphi_{k+1,i-1} = -M/dx
    ind_input_2_3 = ind_phi_kp1_iplusone;
    val_2_3 = 0* ind_input_1_1 - M/dx;
    ind_input_2_4 = sub2ind([nt, nx], row_k_1_to_ntm1(:) + 1, col_i_minus_one(:));
    val_2_4 = 0* ind_input_1_1 - M/dx;
    % d(f2)_{k,i} / dd_{k,i} = H'(d_{k,i})/2
    ind_input_2_5 = ind_f1_k_i + nt * nx;
    val_2_5 = dH_fn(d_var) / 2;
    % d(f2)_{k,i} / dd_{k,i-1} = H'(d_{k,i-1})/2
    ind_input_2_6 = sub2ind([nt-1, nx], row_k_1_to_ntm1(:), col_i_minus_one(:)) + nt * nx;
    val_2_6 = dH_fn([d_var(:, end), d_var(:, 1:end-1)]) / 2;
   
end

% (f3)_{i} = phi_{1,i} - g_i
% d(f3)_i / dphi_{1,i} = 1
ind_input_3 = sub2ind([nt, nx], ones(nx,1), (1:nx)');
ind_output_3 = 2*(nt-1)*nx + (1:nx)';
val_3 = ones(nx,1);


ind_input_1 = [ind_input_1_1(:); ind_input_1_2(:); ind_input_1_3(:)];
ind_output_1 = [ind_f1_k_i(:); ind_f1_k_i(:); ind_f1_k_i(:)];
val_1 = [val_1_1(:); val_1_2(:); val_1_3(:)];

ind_input_2 = [ind_input_2_1(:); ind_input_2_2(:); ind_input_2_3(:); ind_input_2_4(:); ind_input_2_5(:); ind_input_2_6(:)];
ind_output_2 = [ind_f1_k_i(:); ind_f1_k_i(:); ind_f1_k_i(:); ind_f1_k_i(:); ind_f1_k_i(:); ind_f1_k_i(:)] + (nt-1)*nx;
val_2 = [val_2_1(:); val_2_2(:); val_2_3(:); val_2_4(:); val_2_5(:); val_2_6(:)];

mat1 = sparse(ind_output_1, ind_input_1, val_1, nt*nx + (nt-1)*nx, nt*nx + (nt-1)*nx);
mat2 = sparse(ind_output_2, ind_input_2, val_2, nt*nx + (nt-1)*nx, nt*nx + (nt-1)*nx);
mat3 = sparse(ind_output_3(:), ind_input_3(:), val_3(:), nt*nx + (nt-1)*nx, nt*nx + (nt-1)*nx);

output = dt * mat1 + dt * mat2 + mat3;
end


% um = phi_{i, k} - phi_{i-1, k}
% up = phi_{i+1, k} - phi_{i, k}
function [um,up] = compute_leftd_rightd_centerd(phi)
um = phi - [phi(:, end), phi(:, 1:end-1)];
up = [phi(:, 2:end), phi(:, 1)] - phi;
end
