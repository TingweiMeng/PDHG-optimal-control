gpu_no=0

c=10.0
stepsz=0.9
use_prev=True

for v_method in 0 1 2 3
do
    for rho_method in 0
    do
        for updating_rho_first in 0
        do
            for egno in 0 1 2
            do
                CUDA_VISIBLE_DEVICES=${gpu_no}, python3 pdhg1d_v_2var.py --if_prev_codes=$use_prev --egno $egno --c_on_rho $c --stepsz_param $stepsz --v_method $v_method --rho_method $rho_method --updating_rho_first $updating_rho_first >out_eg${egno}_prev${use_prev}_v${v_method}_rho${rho_method}_update${updating_rho_first}_c${c}_step${stepsz}.log 2>&1
            done
        done
    done
done