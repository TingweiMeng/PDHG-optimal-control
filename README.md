# PDHG method for optimal control problem
This is a PDHG method for solving optimal control problem. 
For more details, see the paper [...].

Run the following code (this is for example 1, 1d, diffusion parameter = 0. For other examples, change the flags egno, ndim, epsl. For more details, see the comments in the last part of run_example.py)
CUDA_VISIBLE_DEVICES=0, python3 ./jaxsrc/run_example.py --ndim 1 --epsl 0 --egno 1 --nx 160 --nt 41 --stepsz_param 0.1 --save --notfboard --plot --plot_traj_num_1d 30 > out_1d_egno1_nx160_nt41.log 2>&1 &
