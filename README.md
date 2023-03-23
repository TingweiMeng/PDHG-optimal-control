# PDHG_HJ


Run 
CUDA_VISIBLE_DEVICES=0, python3 ./jaxsrc/test_mtw.py --nx 200 --nt 101 --ifsave=false > ./jaxsrc/out_test_scaling.log 2>&1
(if do not want to save results)

or 
CUDA_VISIBLE_DEVICES=0, python3 ./jaxsrc/test_mtw.py --nx 200 --nt 101 > ./jaxsrc/out_test_scaling.log 2>&1
(if you want to save results)
