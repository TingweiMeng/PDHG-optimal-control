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
Hind = 3;  % L1

if_save = 0;
% nx_arr = [10; 20; 40];
nx_arr = [40];


% spatial domain [0,x_period] (periodic), time domain [0,T]
T = 1;
x_period = 2;

[H, dH, J, alpha, dL] = generating_H_J(Hind, Jind, x_period);

%% setup grids: LF and CFL condition

nt = 2;

% step size param for PDHG
stepsz_param = 0.9;


% PDHG for one primal, two dual using EO
c = 10.0;  % constant to avoid rho = 0


for if_fwd = 0:-1:0
    for nx_ind = 1:length(nx_arr)
        nx = nx_arr(nx_ind);

        fprintf('nx %d, nt %d, T %f, if fwd %d\n', nx, nt, T, if_fwd);

        dx = x_period / (nx);
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
        filename = filename + "Hind" + Hind + "_nx" + nx + "_nt" + nt;
        
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
        [phi_output, error_all, ~, rho_next, ~] = pdhg_onedim_periodic_rho_m_EO_L1(phi0, rho0, m0, mu0, stepsz_param, if_fwd, g, dx, dt, c);
        HJ_residual = check_HJ_sol_usingEO_L1_1d(phi_output, dt, dx, if_fwd);
        
        % plot
        plot_title = "method" + method + " Hind" + Hind + " fwd"+if_fwd + " nx"+nx + " nt" + nt;
        figure; semilogy(error_all(:, 1)); title("primal error " + plot_title);
        figure; semilogy(error_all(:, 2)); title("dual error " + plot_title);
        figure; semilogy(error_all(:, 3)); title("equation error " + plot_title);
        
        % save results
        if if_save == 1
            save(filename);
        end

    end
end




