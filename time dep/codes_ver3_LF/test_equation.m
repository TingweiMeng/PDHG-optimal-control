
f_fn = @(x) x.^2/2;
df_fn = @(x) diag(x);

true_sol = zeros(dim,1);

dim = 3;
phi0 = randn(dim,1);
rho0 = randn(dim,1);
tau = 0.9;
sigma = 0.9;
[phi, error_all] = pdhg_solving_eqt(f_fn, df_fn, phi0, rho0, tau, sigma);

fprintf('err: %f\n', norm(phi - true_sol));