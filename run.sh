LR=0.0001
WD=0.1
BATCH_SIZE=32
NEPOCHS=200
MODEL_TYPE=cnnlstm

#for TARGET in 1 2 3 4 6 7 8 9 12 13 14 15 16 17 19; do
TEST_TARGET=16
#TEST_TARGET=${TARGET}

#python mat2h5Conversion1D.py --test_sub ${TEST_TARGET}

#python main.py --lr ${LR} --batch_size ${BATCH_SIZE} --wd ${WD} --model_type ${MODEL_TYPE} --n_epochs ${NEPOCHS}

python test_performance.py --model_type ${MODEL_TYPE} --batch_size ${BATCH_SIZE}
#done
