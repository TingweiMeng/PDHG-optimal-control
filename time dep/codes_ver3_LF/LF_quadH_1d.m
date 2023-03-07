% this function solves HJ PDE with IC g and quadratic H using LF scheme with forward Euler
function phi = LF_quadH_1d(f, g, nt, nx, dx, dt, M)
phi = zeros(nt, nx);
phi(1,:) = reshape(g, [1, nx]);

for k = 2:nt
    dphibar_left = phi(k-1,:) - [phi(k-1, end), phi(k-1, 1:end-1)];
    dphibar_right = [phi(k-1, 2:end), phi(k-1, 1)] - phi(k-1,:);
    A = (dphibar_right.^2 + dphibar_left.^2) / 4 / (dx*dx);
    phi(k,:) = phi(k-1,:) - dt * A + M*dt/dx*(dphibar_right - dphibar_left);
end
end