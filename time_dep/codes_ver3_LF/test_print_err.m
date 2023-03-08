clear
clc
close all

%% problem setup: Hind = 1: transport, Hind = 2: Burgers; Hind = 3: L1 Hamiltonian
% one dimentional
dim = 1;

method = 1;
Hind = 3;

filename2 = "./prev_results_method1_nosq/burgers_LF_LO_comparison/sol_newtonLO.mat";
% load true solution
load(filename2);
phi_dense = phi_true_5000_newton;

nx_arr = [10; 20; 40];
if method == 1 || method == 2
    if_sq_arr = 1:2;
else
    if_sq_arr = [0];
end

for if_forward = 0:1
    for if_sq_ind = 1:length(if_sq_arr)
        if_square = if_sq_arr(if_sq_ind);
        for nx_ind = 1:length(nx_arr)
            nx = nx_arr(nx_ind);
            nt = nx + 1;
            method = 1;
            % PDHG filename
            filename2 = "./method" + method;
            if if_forward == 1 % forward
                filename2 = filename2 + "_fwd_";
            else % backward
                filename2 = filename2 + "_bwd_";
            end
            filename2 = filename2 + "sqr" + if_square + "_";
            filename2 = filename2 + "Hind" + Hind + "_nx" + nx + "_nt" + nt;
            load(filename2 + ".mat");
            phi_pdhg = phi_output;
            % compute error 
            [error, err_l1, err_l1_rel] = compute_error_HJ(phi_pdhg, phi_dense, T);
            fprintf(filename2 + " error %f\n", err_l1_rel);
            clearvars phi_pdhg phi_output error err_l1 err_l1_rel;
        end
    end
end



