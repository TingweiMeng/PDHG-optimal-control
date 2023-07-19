gpu_no=1

for nt in 21 41
do
    nx=$((($nt - 1) * 2))
    CUDA_VISIBLE_DEVICES=${gpu_no}, python3 compare_2methods_1d.py --egno 0 --nx $nx --nt $nt >out_eg0_1d_compare_${nx}_${nt}.log 2>&1
done