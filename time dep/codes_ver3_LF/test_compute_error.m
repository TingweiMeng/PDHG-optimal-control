load("./results_L1Hamiltonian/true_sol_LO.mat");

phi_dense = phi_true;

nx_arr = [10; 20; 40];

err_fwd_l1 = zeros(length(nx_arr),1);
err_fwd_l1_rel = zeros(length(nx_arr),1);

err_bwd_l1 = zeros(length(nx_arr),1);
err_bwd_l1_rel = zeros(length(nx_arr),1);

for i = 1: length(nx_arr)
    nx = nx_arr(i);
    load("./results_L1Hamiltonian/pdhg_" + nx + ".mat")
    phi_pdhg = phi_PDHG_fwd_2;
    [~, err_fwd_l1(i,1), err_fwd_l1_rel(i,1)] = compute_error_HJ(phi_pdhg, phi_dense, T);
    phi_pdhg = phi_PDHG_bwd_2;
    [~, err_bwd_l1(i,1), err_bwd_l1_rel(i,1)] = compute_error_HJ(phi_pdhg, phi_dense, T);
end

save("./results_L1Hamiltonian/errors.mat", "err_fwd_l1", "err_fwd_l1_rel", "err_bwd_l1", "err_bwd_l1_rel");