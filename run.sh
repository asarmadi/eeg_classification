LR=0.001
WD=0.0
BATCH_SIZE=128
KERNEL_SIZE=9
NEPOCHS=51
MODEL_TYPE=eegnet
STFT=0


APPLY_STFT_ARG=""
if [ $STFT -eq 1 ]; then
  APPLY_STFT_ARG="--stft"
fi

for TARGET in 1 2 3 4 6 7 8 9 12 13 14 15 16 17 19 1; do
TEST_TARGET=${TARGET}

python mat2h5Conversion.py --target_test ${TEST_TARGET} ${APPLY_STFT_ARG}

python main.py --lr ${LR} --batch_size ${BATCH_SIZE} --wd ${WD} --model_type ${MODEL_TYPE} --n_epochs ${NEPOCHS} --kernel_size ${KERNEL_SIZE} ${APPLY_STFT_ARG}

python test_performance.py --model_type ${MODEL_TYPE} --batch_size ${BATCH_SIZE} --kernel_size ${KERNEL_SIZE} ${APPLY_STFT_ARG}

done
