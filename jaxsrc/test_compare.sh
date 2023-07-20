gpu_no=1

for nt in 21 41
do
    nx=$((($nt - 1) * 2))
    CUDA_VISIBLE_DEVICES=${gpu_no}, python3 compare_2methods_1d.py --egno 0 --nx $nx --nt $nt >out_eg0_1d_compare_${nx}_${nt}.log 2>&1
done



CUDA_VISIBLE_DEVICES=1, python3 compare_2methods_1d.py --egno 0 --nx 41 --nt 22 >out_eg0_1d_compare_41_22.log 2>&1
CUDA_VISIBLE_DEVICES=1, python3 compare_2methods_1d.py --egno 0 --nx 80 --nt 41 >out_eg0_1d_compare_80_41.log 2>&1
CUDA_VISIBLE_DEVICES=1, python3 compare_2methods_1d.py --egno 0 --nx 160 --nt 81 >out_eg0_1d_compare_160_81.log 2>&1