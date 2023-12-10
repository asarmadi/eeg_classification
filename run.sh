LR=0.0001
WD=0.0
BATCH_SIZE=32
KERNEL_SIZE=3
NEPOCHS=11
MODEL_TYPE=combined
STFT=0
MAJVOTE=1


APPLY_STFT_ARG=""
if [ $STFT -eq 1 ]; then
  APPLY_STFT_ARG="--stft"
fi

MAJ_VOTE_ARG=""
if [ $MAJVOTE -eq 1 ]; then
  MAJ_VOTE_ARG="--maj_vote"
fi

#for TARGET in 1 2 3 4 6 7 8 9 12 13 14 15 16 17 19; do
TEST_TARGET=${TARGET}

#python mat2h5Conversion.py --target_test ${TEST_TARGET} ${APPLY_STFT_ARG}
#
#python main.py --lr ${LR} --batch_size ${BATCH_SIZE} --wd ${WD} --model_type ${MODEL_TYPE} --n_epochs ${NEPOCHS} --kernel_size ${KERNEL_SIZE} ${APPLY_STFT_ARG}

python -W ignore test_performance.py --model_type ${MODEL_TYPE} --batch_size ${BATCH_SIZE} --kernel_size ${KERNEL_SIZE} ${APPLY_STFT_ARG} ${MAJ_VOTE_ARG}

#done
