
%% problem setup
% one dimentional
dim = 1;
Hind = 2; % Burgers
if Hind == 1
    H = @(p) p;
    dH = @(p) 0.*p+1;
else
    if Hind == 2
        H = @(p) p.^2/2;
        dH = @(p) p;
    end
end

% spatial domain [0,x_period] (periodic), time domain [0,T]
T = 0.1;
x_period = 2;


%% initial condition
test_case = 2;

if test_case == 1
    % example 1: zero initial condition
    J = @(x) 0.*x;
else
    if test_case == 2
        % example 2: sin initial condition
        alpha = 2*pi / x_period;
        J = @(x) sin(alpha * x);
    end
end

% setup grids: LF and CFL condition
M = 2;  % TODO: the constant in LF, depending on J and H

% CFL: dt/dx <= 1/(2M), nt-1 >= M*nx
nx = 10;
dx = x_period / (nx);
dt = dx / 2/M;
nt = max(ceil(T / dt + 1), nx);
dt = T / (nt-1);

x_grid = repmat((0: dx: x_period - dx/2), [nt, 1]);
t_grid = repmat((0: dt: T)', [1,nx]);

% source term f and initial condition g
f = zeros(nx, 1);
g = J(x_grid(1,:));
g = g(:);

%% true solution
if test_case == 1
    % example 1: zero initial condition
    % true solution: zero
    phi_true = zeros(nt,nx);
else
    if test_case == 2
        % example 2: sin initial condition
%         % true solution: L1 Hamiltonian
%         phi_true = zeros(nt,nx) -1;
%         non_negone_index = (abs(3*pi / 2/alpha - x_grid) > t_grid) & (abs(-pi / 2/alpha - x_grid) > t_grid);
%         phi_true(non_negone_index) = min(sin(alpha*(x_grid(non_negone_index)-t_grid(non_negone_index))), sin(alpha*(x_grid(non_negone_index)+t_grid(non_negone_index))));
        % true solution: Burgers
        % search first
        phi = J(x_grid);
        u_min = x_grid;
        for kk = 1:nx
            u_curr = x_grid(1, kk);
            phi_curr = sin(alpha * u_curr) + (x_grid - u_curr).^2/2./t_grid;
            u_min(phi_curr < phi) = u_curr;
            phi = min(phi, phi_curr);
            % right
            u_curr = x_grid(1, kk) + x_period;
            phi_curr = sin(alpha * u_curr) + (x_grid - u_curr).^2/2./t_grid;
            u_min(phi_curr < phi) = u_curr;
            phi = min(phi, phi_curr);
            % left
            u_curr = x_grid(1, kk) - x_period;
            phi_curr = sin(alpha * u_curr) + (x_grid - u_curr).^2/2./t_grid;
            u_min(phi_curr < phi) = u_curr;
            phi = min(phi, phi_curr);
        end
        u_min(1,:) = x_grid(1,:);
        u_next = u_min;

        % Newton solves alp * t * cos(alp * u) + u = x;
        f = @(u) alpha * t_grid .* cos(alpha * u) + u - x_grid;
        u_lower = u_min - dx/2;
        u_upper = u_min + dx/2;
        f_lower = f(u_lower);
        f_upper = f(u_upper);
        u_prev = u_min;
        N_newton = 10;
        for kk = 1:N_newton
            f_prev = f(u_prev);
            f_prime = 1 - alpha * alpha * t_grid .* sin(alpha * u_prev);
            u_newton = u_prev - f_prev./ f_prime;
            u_next = (u_upper + u_lower)/2;
            u_next((u_newton < u_upper) & (u_newton > u_lower)) = u_newton((u_newton < u_upper) & (u_newton > u_lower));
            f_next = f(u_next);
            u_upper((f_lower < -1e-6) & (f_next > 1e-6)) = u_next((f_lower < -1e-6) & (f_next > 1e-6));
            u_upper((f_lower > 1e-6) & (f_next < -1e-6)) = u_next((f_lower > 1e-6) & (f_next < -1e-6));
            u_lower((f_upper < -1e-6) & (f_next > 1e-6)) = u_next((f_upper < -1e-6) & (f_next > 1e-6));
            u_lower((f_upper > 1e-6) & (f_next < -1e-6)) = u_next((f_upper > 1e-6) & (f_next < -1e-6));
            fprintf('newton iter %d, error %f\n', kk, max(abs(f_next(:))));
            u_prev = u_next;
        end

        % periodic bc: add periods s.t. u in [0,x_period]
%         u_next = u_next - floor(u_next / x_period);
        phi_true = sin(alpha * u_next) + (x_grid - u_next).^2 /2 ./ t_grid;

        phi_true(1,:) = reshape(g, [1, nx]);
    end
end


%% LF traditional solution: forward Euler
phi_LF = LF_quadH_1d(f, g, nt, nx, dx, dt, M);
phi_err_LF = abs(phi_true - phi_LF);
phi_err_LF_max = max(phi_err_LF(:));
phi_err_LF_l1 = mean(phi_err_LF(:));
fprintf('LF phi error: max %f, l1 %f\n', phi_err_LF_max, phi_err_LF_l1);


%% PDHG
% initialization for phi
% usual way to do initialization
% phi0 = repmat(reshape(g, [1,nx]), [nt, 1]);
% sanity check
% phi0 = phi_true;
% rho0 = rand(nt-1,nx) *0;
% mu0 = rand(1, nx) *0;
% for debugging
% phi0 = randn(5,5);
% rho0 = randn(4,5);
% mu0 = randn(1,5);
% rho0_ver2 = [rho0(:); mu0(:)];

% phi0 = (1.0: dx: 3.0)';
% phi0 = min((1.0: dx: 3.0)', (3.0: -dx: 1.0)');
% phi0 = rand(nt, nx);
% phi0 = zeros(nt,nx)+ 1;
% phi0 = sin(pi * (x_grid + t_grid));

rho0_ver2 = zeros(nt, nx);

figure; contourf(phi_true); colorbar; title('phi LO');

stepsz_param = 0.1;

[phi_PDHG_bwd_2, error_all_bwd_2, phi_next_bwd_2, phi_bar_bwd_2, rho_next_bwd_2] = pdhg_onedim_periodic_rhophi_LF_backwardEuler_ver2(g, phi0, dx, dt, M, H, dH, stepsz_param, rho0_ver2);
phi_err_bwd_2 = abs(phi_true - phi_PDHG_bwd_2);
phi_err_bwd_2_max = max(phi_err_bwd_2(:));
phi_err_bwd_2_l1 = mean(phi_err_bwd_2(:));
fprintf('bwd ver 2 phi error with LO: max %f, l1 %f\n', phi_err_bwd_2_max, phi_err_bwd_2_l1);

% [phi_PDHG_fwd_2, error_all_fwd_2, phi_next_fwd_2, phi_bar_fwd_2, rho_next_fwd_2] = pdhg_onedim_periodic_rhophi_LF_forwardEuler_ver2(g, phi0, dx, dt, M, H, dH, stepsz_param, rho0_ver2);
% phi_err_fwd2 = abs(phi_true - phi_PDHG_fwd_2);
% phi_err_fwd2_max = max(phi_err_fwd2(:));
% phi_err_fwd2_l1 = mean(phi_err_fwd2(:));
% fprintf('fwd phi error with LF: max %f, l1 %f\n', phi_err_fwd2_max, phi_err_fwd2_l1);
% figure; contourf(phi_PDHG_fwd_2); colorbar; title('phi PDHG fwd');


% [phi_PDHG_bwd, error_all_bwd] = pdhg_onedim_periodic_rhophi_LF_backwardEuler(f, g, phi0, dx, dt, M, Hind, stepsz_param, rho0, mu0);
% phi_err_bwd = abs(phi_true - phi_PDHG_bwd);
% phi_err_bwd_max = max(phi_err_bwd(:));
% phi_err_bwd_l1 = mean(phi_err_bwd(:));
% fprintf('bwd phi error: max %f, l1 %f\n', phi_err_bwd_max, phi_err_bwd_l1);
% figure; contourf(phi_PDHG_bwd); colorbar; title('phi PDHG bwd');


