LR=0.000625
WD=0.00
BATCH_SIZE=8
KERNEL_SIZE=5
NEPOCHS=201
MODEL_TYPE=eegnet

for TARGET in 1 2 3 4 6 7 8 9 12 13 14 15 16 17 19; do
TEST_TARGET=${TARGET}

python mat2h5Conversion1D.py --target_test ${TEST_TARGET}

python main.py --lr ${LR} --batch_size ${BATCH_SIZE} --wd ${WD} --model_type ${MODEL_TYPE} --n_epochs ${NEPOCHS} --kernel_size ${KERNEL_SIZE}

python test_performance.py --model_type ${MODEL_TYPE} --batch_size ${BATCH_SIZE} --kernel_size ${KERNEL_SIZE}

done
