
%% problem setup: Hind = 1: transport, Hind = 2: Burgers; Hind = 3: L1 Hamiltonian
% one dimentional
dim = 1;

if_forward = 1;
if_square = 0;
method = 3;

% Jind = 1 for zero, 2 for sin
Jind = 1;

if_save = 0;

% nx_arr = [10; 20; 40];
nx_arr = [5];

for nx_ind = 1:length(nx_arr)
    nx = nx_arr(nx_ind);
    % Hind = 1 for transport, 2 for Burgers, 3 for L1
    Hind = 2; 
    M = 2;  % the constant in LF, depending on J and H: M >= max|H'|/2, for Burgers
    if_forward = 1;
    [phi_fwd, err_fwd, phi2, err2] = test_fn(M, if_forward, Hind, Jind, nx);
    if_forward = 0;
    [phi_bwd, err_bwd, ~, ~] = test_fn(M, if_forward, Hind, Jind, nx);
end





% err1 using our main code, err2 using const testing code
function [phi1, err1, phi2, err2] = test_fn(M, if_fwd, Hind, Jind, nx)
% spatial domain [0,x_period] (periodic), time domain [0,T]
T = 0.1;
x_period = 2;

[H, dH, J, alpha, dL] = generating_H_J(Hind, Jind, x_period);

%% setup grids: LF and CFL condition

% CFL: dt/dx <= 1/(2M), nt-1 >= M*nx
dx = x_period / (nx);
dt = dx / 2/M;
nt = max(ceil(T / dt + 1), nx+1);
dt = T / (nt-1);

x_grid = repmat((0: dx: x_period - dx/2), [nt, 1]);
t_grid = repmat((0: dt: T)', [1,nx]);

% source term f and initial condition g
g = J(x_grid(1,:));
g = g(:);

% step size param for PDHG
stepsz_param = 0.9;
% initialization for phi
% usual way to do initialization
% phi0 = repmat(reshape(g, [1,nx]), [nt, 1]);

% sanity check
% phi0 = phi_true;

% phi0 = (1.0: dx: 3.0)';
% phi0 = min((1.0: dx: 3.0)', (3.0: -dx: 1.0)');
% phi0 = randn(nt, nx);
% phi0 = zeros(nt,nx)+ 1;
phi0 = sin(pi * (x_grid + t_grid));


% PDHG for one primal, two dual
c = 50.0;  % constant to avoid rho = 0
rho0 = zeros(nt-1, nx);
m0 = zeros(nt-1, nx);
mu0 = zeros(1, nx);
[phi1, err1, phi_next, rho_next, mu_next] = pdhg_onedim_periodic_rhophi_rho_m_LF(H, dL, phi0, rho0, m0, mu0, stepsz_param, M, if_fwd, g, dx, dt, c);
[phi2, err2, phi_next2, rho_next2, mu_next2] = pdhg_test_rho_m_LF_constx(phi0(:,1), rho0(:,1), mu0(:,1), stepsz_param, g(1));
phi_diff = phi_next  - phi_next2;
rho_diff = rho_next  - rho_next2;
mu_diff = mu_next  - mu_next2;
fprintf('phi %f, mu %f, rho %f\n', max(abs(phi_diff(:))), max(abs(mu_diff(:))), max(abs(rho_diff(:))));
end

