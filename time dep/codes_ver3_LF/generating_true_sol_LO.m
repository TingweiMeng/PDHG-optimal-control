% solution HJ PDE using Lax-Oleinik formula
function phi_true = generating_true_sol_LO(Hind, Jind, J, x_grid, t_grid, alpha, x_period)
if Hind == 1
    phi_true = sol_H_identity(J, x_grid, t_grid);
else
    if Hind == 2
        phi_true = sol_H_Burgers(Jind, J, alpha, x_grid, t_grid, x_period);
    else
        if Hind == 3
            phi_true = sol_H_L1(Jind, J, alpha, x_grid, t_grid);
        end
    end
end
end


function phi_true = sol_H_identity(J, x_grid, t_grid)
% true solution: S(x,t) = J(x - t)
phi_true = J(x_grid - t_grid);
end

function phi_true = sol_H_Burgers(Jind, J, alpha, x_grid, t_grid, x_period)
[nt, nx] = size(x_grid);
if Jind == 1
    % example 1: zero initial condition
    % true solution: zero
    phi_true = zeros(nt,nx);
else
    if Jind == 2
        % example 2: sin initial condition
        % true solution: Burgers
        dx = x_period / (nx);
        % search first
        phi = J(x_grid);
        u_min = x_grid;
        for kk = 1:nx
            u_curr = x_grid(1, kk);
            phi_curr = J(u_curr) + (x_grid - u_curr).^2/2./t_grid;
            u_min(phi_curr < phi) = u_curr;
            phi = min(phi, phi_curr);
            % right
            u_curr = x_grid(1, kk) + x_period;
            phi_curr = J(u_curr) + (x_grid - u_curr).^2/2./t_grid;
            u_min(phi_curr < phi) = u_curr;
            phi = min(phi, phi_curr);
            % left
            u_curr = x_grid(1, kk) - x_period;
            phi_curr = J(u_curr) + (x_grid - u_curr).^2/2./t_grid;
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
        phi_true = sin(alpha * u_next) + (x_grid - u_next).^2 /2 ./ t_grid;
        phi_true(1,:) = reshape(J(x_grid(1,:)), [1, nx]);
    end
end
end

function phi_true = sol_H_L1(Jind, J, alpha, x_grid, t_grid)
[nt, nx] = size(x_grid);
%% true solution: S(x,t) = min_{|x-u|_inf <= t} J(u)
% otherwise, S = min{J(x-t), J(x+t), possible local minimizer in [x-t, x+t]}
if Jind == 1
    % example 1: zero initial condition
    % true solution: zero
    phi_true = zeros(nt,nx);
else
    if Jind == 2
        % example 2: sin initial condition
        % true solution: L1 Hamiltonian
        phi_true = zeros(nt,nx) -1;
        non_negone_index = (abs(3*pi / 2/alpha - x_grid) > t_grid) & (abs(-pi / 2/alpha - x_grid) > t_grid);
        phi_true(non_negone_index) = min(J(x_grid(non_negone_index)-t_grid(non_negone_index)), J(x_grid(non_negone_index)+t_grid(non_negone_index)));
    end
end
end