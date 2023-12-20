LR=0.0001
WD=0.0
BATCH_SIZE=128
KERNEL_SIZE=3
NEPOCHS=201
MODEL_TYPE=cnn
TRANSFORM=stockwell
MAJVOTE=1


MAJ_VOTE_ARG=""
if [ $MAJVOTE -eq 1 ]; then
  MAJ_VOTE_ARG="--maj_vote"
fi

#for TARGET in 1 2 3 4 6 7 8 9 12 13 14 15 16 17 19; do
#TEST_TARGET=${TARGET}
TEST_TARGET=12
#python -W ignore mat2h5Conversion.py --target_test ${TEST_TARGET} --apply_transform ${TRANSFORM}
#
python -W ignore main.py --lr ${LR} --batch_size ${BATCH_SIZE} --wd ${WD} --model_type ${MODEL_TYPE} --n_epochs ${NEPOCHS} --kernel_size ${KERNEL_SIZE} --transform ${TRANSFORM}

#python -W ignore test_performance.py --model_type ${MODEL_TYPE} --batch_size ${BATCH_SIZE} --kernel_size ${KERNEL_SIZE} --transform ${TRANSFORM} ${MAJ_VOTE_ARG}

#done
