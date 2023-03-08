
%% problem setup: Hind = 1: transport, Hind = 2: Burgers; Hind = 3: L1 Hamiltonian
% one dimentional
dim = 1;

% spatial domain [0,x_period] (periodic), time domain [0,T]
T = 0.1;
x_period = 2;

% Hind = 1 for transport, 2 for Burgers, 3 for L1
Hind = 3; 
% Jind = 1 for zero, 2 for sin
Jind = 2;
[H, dH, J, alpha] = generating_H_J(Hind, Jind, x_period);

%% setup grids: LF and CFL condition
M = 0.5;  % the constant in LF, depending on J and H: M >= max|H'|/2

% CFL: dt/dx <= 1/(2M), nt-1 >= M*nx
nx = 40;
dx = x_period / (nx);
dt = dx / 2/M;
nt = max(ceil(T / dt + 1), nx+1);
dt = T / (nt-1);

x_grid = repmat((0: dx: x_period - dx/2), [nt, 1]);
t_grid = repmat((0: dt: T)', [1,nx]);

% source term f and initial condition g
f = zeros(nx, 1);
g = J(x_grid(1,:));
g = g(:);

phi_true = generating_true_sol_LO(Hind, Jind, J, x_grid, t_grid, alpha, x_period);


% %% LF traditional solution: forward Euler
% phi_LF = LF_quadH_1d(f, g, nt, nx, dx, dt, M);
% phi_err_LF = abs(phi_true - phi_LF);
% phi_err_LF_max = max(phi_err_LF(:));
% phi_err_LF_l1 = mean(phi_err_LF(:));
% fprintf('LF phi error: max %f, l1 %f\n', phi_err_LF_max, phi_err_LF_l1);


%% PDHG
% initialization for phi
% usual way to do initialization
phi0 = repmat(reshape(g, [1,nx]), [nt, 1]);
% sanity check
% phi0 = phi_true;

% phi0 = (1.0: dx: 3.0)';
% phi0 = min((1.0: dx: 3.0)', (3.0: -dx: 1.0)');
% phi0 = randn(nt, nx);
% phi0 = zeros(nt,nx)+ 1;
% phi0 = sin(pi * (x_grid + t_grid));

rho0_ver2 = zeros(nt, nx);

% figure; contourf(phi_true); colorbar; title('phi LO');

stepsz_param = 0.9;

[phi_PDHG_bwd_2, error_all_bwd_2, phi_next_bwd_2, phi_bar_bwd_2, rho_next_bwd_2] = pdhg_onedim_periodic_rhophi_LF_backwardEuler_ver2(g, phi0, dx, dt, M, H, dH, stepsz_param, rho0_ver2);
phi_err_bwd_2 = abs(phi_true - phi_PDHG_bwd_2);
phi_err_bwd_2_max = max(phi_err_bwd_2(:));
phi_err_bwd_2_l1 = mean(phi_err_bwd_2(:));
fprintf('bwd ver 2 phi error with LO: max %f, l1 %f\n', phi_err_bwd_2_max, phi_err_bwd_2_l1);

[phi_PDHG_fwd_2, error_all_fwd_2, phi_next_fwd_2, phi_bar_fwd_2, rho_next_fwd_2] = pdhg_onedim_periodic_rhophi_LF_forwardEuler_ver2(g, phi0, dx, dt, M, H, dH, stepsz_param, rho0_ver2);
phi_err_fwd2 = abs(phi_true - phi_PDHG_fwd_2);
phi_err_fwd2_max = max(phi_err_fwd2(:));
phi_err_fwd2_l1 = mean(phi_err_fwd2(:));
fprintf('fwd phi error with LF: max %f, l1 %f\n', phi_err_fwd2_max, phi_err_fwd2_l1);
% figure; contourf(phi_PDHG_fwd_2); colorbar; title('phi PDHG fwd');


% [phi_PDHG_bwd, error_all_bwd] = pdhg_onedim_periodic_rhophi_LF_backwardEuler(f, g, phi0, dx, dt, M, Hind, stepsz_param, rho0, mu0);
% phi_err_bwd = abs(phi_true - phi_PDHG_bwd);
% phi_err_bwd_max = max(phi_err_bwd(:));
% phi_err_bwd_l1 = mean(phi_err_bwd(:));
% fprintf('bwd phi error: max %f, l1 %f\n', phi_err_bwd_max, phi_err_bwd_l1);
% figure; contourf(phi_PDHG_bwd); colorbar; title('phi PDHG bwd');


