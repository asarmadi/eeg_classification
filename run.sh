LR=0.0001
WD=0.01
BATCH_SIZE=128
KERNEL_SIZE=5
NEPOCHS=41
MODEL_TYPE=capsnet

#for TARGET in 1 2 3 4 6 7 8 9 12 13 14 15 16 17 19; do
#TEST_TARGET=${TARGET}
#VALID_TARGET=${}

#python mat2h5Conversion1D.py --target_test ${TEST_TARGET}

python main.py --lr ${LR} --batch_size ${BATCH_SIZE} --wd ${WD} --model_type ${MODEL_TYPE} --n_epochs ${NEPOCHS} --kernel_size ${KERNEL_SIZE}

#python test_performance.py --model_type ${MODEL_TYPE} --batch_size ${BATCH_SIZE} --kernel_size ${KERNEL_SIZE}

#done
