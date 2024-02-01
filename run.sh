
LR=0.0001
WD=0.0
BATCH_SIZE=32
KERNEL_SIZE=5
NEPOCHS=201
MODEL_TYPE=mlp
TRANSFORM=aelinear
MAJVOTE=1


MAJ_VOTE_ARG=""
if [ $MAJVOTE -eq 1 ]; then
  MAJ_VOTE_ARG="--maj_vote"
fi

for TARGET in 1 2 3 4 6 7 8 9 10 12 13 14 15 16 17 18; do
#for TARGET in 7; do
TEST_TARGET=${TARGET}
TRANSFORM=1d
#python -W ignore mat2h5ConversionFull.py --target_test ${TEST_TARGET} --apply_transform ${TRANSFORM}
python -W ignore mat2h5Conversion.py --target_test ${TEST_TARGET} --apply_transform ${TRANSFORM}
#python -W ignore mat2h5ConversionSeparateFiles.py --target_test ${TEST_TARGET} --apply_transform ${TRANSFORM}
#done
#python csp_test.py --subject ${TEST_TARGET}
#done
#for TARGET in 1 2 3 4 6 7 8 9 12 13 14 15 16 17 19; do
#TEST_TARGET=${TARGET}
##### Training AE ######
NEPOCHS=31
MODEL_TYPE=atcnet
TRANSFORM=nothing
python -W ignore main.py --lr ${LR} --batch_size ${BATCH_SIZE} --wd ${WD} --model_type ${MODEL_TYPE} --n_epochs ${NEPOCHS} --kernel_size ${KERNEL_SIZE} --transform ${TRANSFORM} --subject ${TEST_TARGET}

#### Trainin MLP using AE
#NEPOCHS=201
#MODEL_TYPE=mlp
#TRANSFORM=aelinear
#python -W ignore main.py --lr ${LR} --batch_size ${BATCH_SIZE} --wd ${WD} --model_type ${MODEL_TYPE} --n_epochs ${NEPOCHS} --kernel_size ${KERNEL_SIZE} --transform ${TRANSFORM}


#### Testing MLP
#BATCH_SIZE=16
#for TARGET in 1 2 3 4 6 7 8 9 12 13 14 15 16 17 19; do
python -W ignore test_performance.py --model_type ${MODEL_TYPE} --batch_size ${BATCH_SIZE} --kernel_size ${KERNEL_SIZE} --transform ${TRANSFORM} ${MAJ_VOTE_ARG}
#python -W ignore voting.py --model_type ${MODEL_TYPE} --batch_size ${BATCH_SIZE} --kernel_size ${KERNEL_SIZE} --transform ${TRANSFORM} ${MAJ_VOTE_ARG} --target_test ${TEST_TARGET}
done

python plot_test_acc.py
