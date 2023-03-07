% this function uses PDHG to solve
% dphi/dt + H(nabla_x phi)=0, and phi(x,0)-g(x) = 0
% where we use forward Euler to discretize dphi/dt, and LF scheme for H(nabla_x phi)
%   H_flux = H(phi_{i+1} - phi_{i})/dx)/2 + H(phi_i - phi_{i-1})/dx)/2 -M(phi_{i+1}+phi_{i-1}-phi_{i})/dx

% for now, assume dim = 1

% dx is the spatial grid size, dt is the time grid size
% f size: nx * 1
% g size: nx * 1
% phi0 size: nt * nx
% i is spatial index, and k is the time index
function [phi, error_all, phi_next, phi_bar, rho_next] = pdhg_onedim_periodic_rhophi_LF_backwardEuler_ver2(g, phi0, dx, dt, M, H_fn, dH_fn, stepsz_param, rho0)

% NOTE: M should be negative for backward Euler
M = -abs(M);

[nt,nx] = size(phi0);
tau = stepsz_param / (3 + 8 * abs(M)* dt / dx);
sigma = tau;

% reshape g to row vec
g = reshape(g, [1,nx]);

f_fn = @(phi) f_fn_HJ(phi, H_fn, nx, nt, dx, dt, M, g);
df_fn = @(phi) df_fn_HJ(phi, dH_fn, nx, nt, dx, dt, M);

% rho0 = zeros(nt,nx);  % combining rho and mu
[phi, error_all, phi_next, phi_bar, rho_next] = pdhg_solving_eqt(f_fn, df_fn, phi0(:), rho0(:), tau, sigma);

phi = reshape(phi, [nt, nx]);
end


% define f for HJ
% the remaining input parameters are H_fn, nx, nt, dx, dt, M, g
function output = f_fn_HJ(phi, varargin)
H_fn = varargin{1};
nx = varargin{2};
nt = varargin{3};
dx = varargin{4};
dt = varargin{5};
M = varargin{6};
g = varargin{7};
phi_bar = reshape(phi, [nt, nx]);
% phi_{k+1,i} - phi_{k+1,i-1} and phi_{k+1,i+1} - phi_{k+1,i}
[dphibar_left, dphibar_right] = compute_leftd_rightd_centerd(phi_bar(2:end,:));
% compute H((phi_{k,i+1} - phi_{k,i})/dx) / 2 + H((phi_{k+1,i} - phi_{k+1,i-1})/dx) / 2
H = H_fn(dphibar_left / dx) / 2 + H_fn(dphibar_right / dx) / 2;
HJ_residual = (phi_bar(2:end,:) - phi_bar(1:end-1,:))/dt + H - M/dx*(dphibar_right - dphibar_left);
bc_residual = phi_bar(1,:) - g;
output = [HJ_residual(:); bc_residual(:)];

% scaling for consistency with version 1
output(1:(nt-1)*nx) = output(1:(nt-1)*nx) * dt;
output = - output;
end


% define df for HJ
% f = [f1; f2]
% (f1)_{i,k} = (phi_{i,k+1} - phi_{i,k})/dt + H((phi_{i+1,k+1} - phi_{i,k+1})/dx)/2 + H((phi_{i,k+1} - phi_{i-1,k+1})/dx)/2 - M(phi_{i+1,k+1} + phi_{i-1,k+1} - 2phi_{i,k+1})/dx
% (f2)_{i} = phi_{i,1} - g_i
% input of f has size (nt, nx), output of f1 has size (nt-1, nx), output of f2 has size (nx,1)
% the remaining input parameters are dH_fn, nx, nt, dx, dt, M
function output = df_fn_HJ(phi, varargin)
dH_fn = varargin{1};
nx = varargin{2};
nt = varargin{3};
dx = varargin{4};
dt = varargin{5};
M = varargin{6};
phi_bar = reshape(phi, [nt, nx]);

% phi_{k+1,i} - phi_{k+1,i-1} and phi_{k+1,i+1} - phi_{k+1,i}
[dphibar_left, dphibar_right] = compute_leftd_rightd_centerd(phi_bar(2:end,:));

% all the index for output of f1
row_output_f1 = repmat((1:nt-1)', [1,nx]);  % each element is the first index for (k,i), k=1,...,nt-1
col_output_f1 = repmat((1:nx), [nt-1,1]);  % each element is the second index for (k,i), k=1,...,nt-1
ind_output_f1 = sub2ind([nt-1, nx], row_output_f1(:), col_output_f1(:));

% d(f1)_{k,i} / dphi_{k+1,i} = 1/dt - dH((phi_{k+1,i+1} - phi_{k+1,i})/dx)/(2dx) + dH((phi_{k+1,i} - phi_{k+1,i-1})/dx)/(2dx) + 2M/dx
ind_input_1 = sub2ind([nt, nx], row_output_f1(:)+1, col_output_f1(:));
val_1 = 1/dt - dH_fn(dphibar_right / dx) /(2*dx) + dH_fn(dphibar_left / dx)/(2*dx) + 2*M/dx;

% d(f1)_{k,i} / dphi_{k+1,i+1} = dH((phi_{k+1,i+1} - phi_{k+1,i})/dx)/(2dx)- M/dx
col_output_f1_right = [col_output_f1(:,2:end), col_output_f1(:,1)];
ind_input_2 = sub2ind([nt, nx], row_output_f1(:)+1, col_output_f1_right(:));
val_2 = dH_fn(dphibar_right / dx) /(2*dx) - M/dx;

% d(f1)_{k,i} / dphi_{k+1,i-1} = -dH((phi_{k+1,i} - phi_{k+1,i-1})/dx)/(2dx)- M/dx
col_output_f1_left = [col_output_f1(:,end), col_output_f1(:,1:end-1)];
ind_input_3 = sub2ind([nt, nx], row_output_f1(:)+1, col_output_f1_left(:));
val_3 = -dH_fn(dphibar_left / dx) /(2*dx) - M/dx;

% d(f1)_{k,i} / dphi_{k,i} = -1/dt
ind_input_4 = sub2ind([nt, nx], row_output_f1(:), col_output_f1(:));
val_4 = zeros((nt-1)*nx, 1) - 1/dt;

% d(f2)_i / dphi_{1,i} = 1
ind_input_5 = sub2ind([nt, nx], ones(nx,1), (1:nx)');
ind_output_5 = (nt-1)*nx + (1:nx)';
val_5 = ones(nx,1);

ind_input = [ind_input_1(:); ind_input_2(:); ind_input_3(:); ind_input_4(:)];
ind_output = [ind_output_f1(:); ind_output_f1(:); ind_output_f1(:); ind_output_f1(:)];
val = [val_1(:); val_2(:); val_3(:); val_4(:)];

mat1 = sparse(ind_output, ind_input, val, nt*nx, nt*nx);
mat2 = sparse(ind_output_5(:), ind_input_5(:), val_5(:), nt*nx, nt*nx);

% scaling for consistency with version 1
mat1 = -dt * mat1;
mat2 = - mat2;

output = mat1 + mat2;
end


% um = phi_{i, k} - phi_{i-1, k}
% up = phi_{i+1, k} - phi_{i, k}
function [um,up] = compute_leftd_rightd_centerd(phi)
um = phi - [phi(:, end), phi(:, 1:end-1)];
up = [phi(:, 2:end), phi(:, 1)] - phi;
end