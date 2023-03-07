close all;


%% problem setup: 1-dimensional, H(x,p) = c(x)|p| + f(x)
% solve this using PDHG + EO (the rho,m,phi formulation)
% one dimentional
dim = 1;

if_save = 1;
if_precondition = 0;
% nx_arr = [10; 20; 40];
nx_arr = [160];

nt = 81;

% spatial domain [0,x_period] (periodic), time domain [0,T]
T = 1;
x_period = 2;


% Jind = 1 for zero, 2 for sin
alpha = 2*pi / x_period;
J = @(x) sin(alpha * x);
% J = @(x) (x.^2-1)/2;
% [H, dH, J, alpha, dL] = generating_H_J(Hind, Jind, x_period);

%% setup grids: LF and CFL condition

% step size param for PDHG
stepsz_param = 0.9;  %0.9;


% PDHG for one primal, two dual using EO
c = 10.0;  % constant to avoid rho = 0


f_in_H_fn = @(x) 0*x;
c_in_H_fn = @(x) 1 + 3* exp(-4* (x-1).^2);  % H in Yat Tin's paper

for if_fwd = 0:-1:0
    for nx_ind = 1:length(nx_arr)
        nx = nx_arr(nx_ind);

        fprintf('nx %d, nt %d, T %f, if fwd %d\n', nx, nt, T, if_fwd);

        dx = x_period / (nx);
        dt = T / (nt-1);

        x_arr = (0: dx: x_period - dx/2);
        x_grid = repmat(x_arr, [nt, 1]);
        t_grid = repmat((0: dt: T)', [1,nx]);
        
        % source term f and initial condition g
        g = J(x_grid(1,:));
        g = g(:);

        % general f and c
        f_in_H = f_in_H_fn(x_arr);
        c_in_H = c_in_H_fn(x_arr);
        
        % PDHG filename
        filename = "EO_xdep";
        if if_fwd == 1 % forward
            filename = filename + "_fwd_";
        else % backward
            filename = filename + "_bwd_";
        end
        filename = filename + "nx" + nx + "_nt" + nt;
        
        if if_precondition == 1
            filename = filename + "_preconditioning";
        else
            filename = filename + "_no_preconditioning";
        end
        
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

        rho0 = zeros(nt-1, nx);
        m0 = zeros(nt-1, nx);
        mu0 = zeros(1, nx);
        [phi_output, error_all, ~, rho_next, ~] = pdhg_onedim_periodic_rho_m_EO_L1_xdep(f_in_H, c_in_H, phi0, rho0, m0, mu0, stepsz_param, if_fwd, g, dx, dt, c, if_precondition);

        HJ_residual = check_HJ_sol_usingEO_L1_1d_xdep(phi_output, dt, dx, if_fwd, f_in_H, c_in_H);
        
        % plot
        plot_title = "EO xdep" + " fwd"+if_fwd + " nx"+nx + " nt" + nt;
        figure; semilogy(error_all(:, 1)); title("primal error " + plot_title);
        figure; semilogy(error_all(:, 2)); title("dual error " + plot_title);
        figure; semilogy(error_all(:, 3)); title("equation error " + plot_title);
        
        % save results
        if if_save == 1
            save(filename);
        end

    end
end




