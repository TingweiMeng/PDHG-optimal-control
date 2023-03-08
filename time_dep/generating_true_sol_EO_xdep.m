
% g is initial conditon, size [1, nx]
% H = c(x)|p| + f(x)
% c_in_H and f_in_H has size [1,nx]
% phi_{k+1} = phi_k - dt * H_flux(x, dphidx_left, dphidx_right)
function phi = generating_true_sol_EO_xdep(g, f_in_H, c_in_H, nt, dt, dx)
g = g(:)';
phi = repmat(g, [nt,1]);

for i = 2:nt
    dphidx_left = (phi(i-1,:) - [phi(i-1, end), phi(i-1, 1:end-1)])/dx;
    dphidx_right = ([phi(i-1, 2:end), phi(i-1, 1)] - phi(i-1,:))/dx;
    H_val = max(-dphidx_right, 0) + max(dphidx_left, 0);
    H_val = c_in_H.* H_val + f_in_H;
    phi(i,:) = phi(i-1,:) - dt * H_val;
end
end