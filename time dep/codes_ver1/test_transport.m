
%% one dimentional
dim = 1;
Hind = 1; % transport equation

M = 0.5;  % TODO: the constant in LF, depending on J and H

% CFL: dt/dx <= 1/(2M), nt-1 >= M*nx
% spatial domain [0,2], time domain [0,1]
x_period = 2;
nx = 20;
nt = ceil(abs(M)*(nx) + 1);
dx = x_period / (nx);
dt = 1.0 / (nt-1);

f = zeros(nx, 1);
x_grid = repmat((0: dx: x_period - dx/2), [nt, 1]);
t_grid = repmat((0: dt: 1)', [1,nx]);


%% testing case
test_case = 2;

if test_case == 1
    % example 1: zero initial condition
    g = zeros(nx, 1);
    % true solution
    phi_true = zeros(nt,nx);
else
    if test_case == 2
        % example 2: sin initial condition
        alpha = 2*pi / x_period;
        g = sin(alpha * x_grid(1,:));
        g = g(:);
        % true solution: H(p) = p
        u = x_grid - t_grid;
        phi_true = sin(alpha * u);
    end
end


% phi_LF = LF_quadH_1d(f, g, nt, nx, dx, dt);

% initialization for phi
% usual way to do initialization
phi0 = repmat(reshape(g, [1,nx]), [nt, 1]);
% sanity check
% phi0 = phi_true;

% phi0 = (1.0: dx: 3.0)';
% phi0 = min((1.0: dx: 3.0)', (3.0: -dx: 1.0)');
% phi0 = rand(nt, nx);
% phi0 = zeros(nt,nx)+ 1;
% phi0 = sin(pi * (x_grid + t_grid));

[phi_PDHG_fwd, error_all_fwd] = pdhg_onedim_periodic_rhophi_LF_forwardEuler(f, g, phi0, dx, dt, M, Hind);
[phi_PDHG_bwd, error_all_bwd] = pdhg_onedim_periodic_rhophi_LF_backwardEuler(f, g, phi0, dx, dt, M, Hind);
phi_err_fwd = abs(phi_true - phi_PDHG_fwd);
phi_err_fwd_max = max(phi_err_fwd(:));
phi_err_fwd_l1 = sum(phi_err_fwd(:));
phi_err_bwd = abs(phi_true - phi_PDHG_bwd);
phi_err_bwd_max = max(phi_err_bwd(:));
phi_err_bwd_l1 = sum(phi_err_bwd(:));
fprintf('fwd phi error: max %f, l1 %f\n', phi_err_fwd_max, phi_err_fwd_l1);
fprintf('bwd phi error: max %f, l1 %f\n', phi_err_bwd_max, phi_err_bwd_l1);

figure; contourf(phi_PDHG_fwd); colorbar; title('phi PDHG fwd');
figure; contourf(phi_PDHG_bwd); colorbar; title('phi PDHG bwd');
figure; contourf(phi_true); colorbar; title('phi LO');
% figure; contourf(phi_LF); colorbar; title('phi LF');

