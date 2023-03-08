%% one dimentional
dim = 1;

% spatial domain [0,2], time domain [0,1]
x_period = 2;
nx = 11;
nt = 11;
dx = x_period / (nx);
dt = 1.0 / (nt-1);

f = zeros(nx, 1);
x_grid = repmat((0: dx: x_period - dx/2), [nt, 1]);
t_grid = repmat((0: dt: 1)', [1,nx]);


%% testing case
test_case = 1;

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
        % true solution
        phi_true = zeros(nt,nx) -1;
        non_negone_index = (abs(3*pi / 2/alpha - x_grid) > t_grid) & (abs(-pi / 2/alpha - x_grid) > t_grid);
        phi_true(non_negone_index) = min(sin(alpha*(x_grid(non_negone_index)-t_grid(non_negone_index))), sin(alpha*(x_grid(non_negone_index)+t_grid(non_negone_index))));
    end
end


% terminal condition for rho: uniform
rho_tilde = (zeros(nx,1) + 1.0) / nx / dx;

% initialization for phi
% usual way to do initialization
% phi0 = repmat(reshape(g, [1,nx]), [nt, 1]);
% sanity check
% phi0 = phi_true;

% phi0 = (1.0: dx: 3.0)';
% phi0 = min((1.0: dx: 3.0)', (3.0: -dx: 1.0)');
% phi0 = rand(nt, nx);
% phi0 = zeros(nt,nx)+ 1;
phi0 = sin(pi * (x_grid + t_grid));


% [phi, error_all] = pdhg_L1Hamiltonian_onedim_periodic(f, g, rho_tilde, phi0, dx, dt);
[phi, error_all] = pdhg_L1Hamiltonian_onedim_periodic_rhophi_LF_forwardEuler(f, g, phi0, dx, dt);
phi_err = abs(phi_true - phi);
phi_err_max = max(phi_err(:));
phi_err_l1 = sum(phi_err(:));
fprintf('phi error: max %f, l1 %f\n', phi_err_max, phi_err_l1);

figure; contourf(phi); colorbar; title('phi');