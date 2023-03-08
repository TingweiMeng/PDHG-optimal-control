close all;


%% problem setup: Hind = 1: transport, Hind = 2: Burgers; Hind = 3: L1 Hamiltonian
% one dimentional
dim = 1;

% method =  0 if LO; 1 if PDHG variables phi, rho, mu; 2 for PDHG var phi, d, rho, mu, m
%           3 for PDHG var phi, rho, mu, m. All 1-3 uses PDHG
%           4 for PDHG var phi, rho, mu, m using EO
method = 4;

% Jind = 1 for zero, 2 for sin
Jind = 2;

if_save = 0;

nx_arr = [10; 20; 40];

if method == 1 || method == 2
    if_sq_arr = [0]; %0:2;
else
    if_sq_arr = [0];
end

for if_forward = 0:-1:0
    for if_sq_ind = 1:length(if_sq_arr)
        if_square = if_sq_arr(if_sq_ind);
        for nx_ind = 1:length(nx_arr)
            nx = nx_arr(nx_ind);
            % Hind = 1 for transport, 2 for Burgers, 3 for L1
            Hind = 3; 
            if method == 4  % EO only implemented for L1 Hamiltonian
                Hind = 3;
            end
            M = 2;  % the constant in LF, depending on J and H: M >= max|H'|/2, for Burgers
            phi_output = test_fn(method, M, if_forward, if_square, Hind, Jind, if_save, nx);
    
%             % Hind = 1 for transport, 2 for Burgers, 3 for L1
%             Hind = 3; 
%             M = 0.5;  % the constant in LF, depending on J and H: M >= max|H'|/2, for L1
%             phi_output = test_fn(method, M, if_forward, if_square, Hind, Jind, if_save, nx);
        end
    end
end





% method = 0 if LO; 1 if PDHG variables phi, rho, mu; 2 for PDHG var phi, d, rho, mu, m
%          3 for PDHG var phi, rho, mu, m 
function phi_output = test_fn(method, M, if_fwd, if_square, Hind, Jind, if_save, nx)
% spatial domain [0,x_period] (periodic), time domain [0,T]
T = 0.1;
x_period = 2;

[H, dH, J, alpha, dL] = generating_H_J(Hind, Jind, x_period);

%% setup grids: LF and CFL condition

% CFL: dt/dx <= 1/(2M), nt-1 >= M*nx
dx = x_period / (nx);
dt = dx / 2/M;
nt = max(ceil(T / dt + 1), ceil(nx + 1));
if method == 4
    nt = 2;
end
dt = T / (nt-1);

x_grid = repmat((0: dx: x_period - dx/2), [nt, 1]);
t_grid = repmat((0: dt: T)', [1,nx]);

% source term f and initial condition g
g = J(x_grid(1,:));
g = g(:);

% PDHG filename
filename = "./method" + method;
if if_fwd == 1 % forward
    filename = filename + "_fwd_";
else % backward
    filename = filename + "_bwd_";
end
filename = filename + "sqr" + if_square + "_";
filename = filename + "Hind" + Hind + "_nx" + nx + "_nt" + nt;

% step size param for PDHG
stepsz_param = 1.6;
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

switch method
    case 0
        phi_output = generating_true_sol_LO(Hind, Jind, J, x_grid, t_grid, alpha, x_period);
        filename = "./true_sol_LO.mat";  % true solution: rewrite filename


% %% LF traditional solution: forward Euler
% phi_LF = LF_quadH_1d(f, g, nt, nx, dx, dt, M);
% phi_err_LF = abs(phi_true - phi_LF);
% phi_err_LF_max = max(phi_err_LF(:));
% phi_err_LF_l1 = mean(phi_err_LF(:));
% fprintf('LF phi error: max %f, l1 %f\n', phi_err_LF_max, phi_err_LF_l1);

    case 1 
        % PDHG for one primal, one dual
        if if_square == 1
            rho0 = 0;
        else
            rho0 = zeros(nt, nx);
        end
    
        [phi_output, error_all, ~, ~, ~] = pdhg_onedim_periodic_rhophi_LF_ver3(g, phi0, dx, dt, M, H, dH, stepsz_param, rho0, if_fwd, if_square);
    case 2
        % PDHG for two primal, two dual
        if if_square == 1
            rho0 = 0;
        else
            rho0 = zeros(nt* nx + (nt-1) * nx, 1);
        end
        
        [phi_output, error_all, ~, ~, ~] = pdhg_onedim_periodic_rhophi_d_LF(g, phi0, dx, dt, M, H, dH, stepsz_param, rho0, if_fwd, if_square);
    case 3
        % PDHG for one primal, two dual
        c = 10.0;  % constant to avoid rho = 0
        rho0 = zeros(nt-1, nx);
        m0 = zeros(nt-1, nx);
        mu0 = zeros(1, nx);
        [phi_output, error_all, ~, rho_next, ~] = pdhg_onedim_periodic_rhophi_rho_m_LF(H, dL, phi0, rho0, m0, mu0, stepsz_param, M, if_fwd, g, dx, dt, c);
    case 4
        % PDHG for one primal, two dual using EO
        c = 10.0;  % constant to avoid rho = 0
        rho0 = zeros(nt-1, nx);
        m0 = zeros(nt-1, nx);
        mu0 = zeros(1, nx);
        [phi_output, error_all, ~, rho_next, ~] = pdhg_onedim_periodic_rho_m_EO_L1(phi0, rho0, m0, mu0, stepsz_param, if_fwd, g, dx, dt, c);
%         if if_fwd
%             [phi_output, error_all, ~, rho_next, ~] = pdhg_onedim_periodic_rho_m_EO_L1(phi0, rho0, m0, mu0, stepsz_param, if_fwd, g, dx, dt, c);
%         else
%             [phi_output1, error_all1, ~, rho_next1, ~] = pdhg_onedim_periodic_rho_m_EO_L1(phi0, rho0, m0, mu0, stepsz_param, if_fwd, g, dx, dt, c);
%             [phi_output, error_all, ~, rho_next, ~] = pdhg_onedim_periodic_rho_m_EO_L1_implicit(phi0, rho0, m0, mu0, stepsz_param, if_fwd, g, dx, dt, c);
%         end
        HJ_residual = check_HJ_sol_usingEO_L1_1d(phi_output, dt, dx, if_fwd);
end

% fprintf('max sol val %f, min sol val %f\n', max(phi_output(:)), min(phi_output(:)));

% plot
plot_title = "method" + method + " Hind" + Hind + " fwd"+if_fwd + " sqr"+if_square + " nx"+nx;
figure; semilogy(error_all(:, 1)); title("err1 " + plot_title);
figure; semilogy(error_all(:, 2)); title("err eqt " + plot_title);

% save results
if if_save == 1
    save(filename);
end
end

