LR=0.0011
WD=0.000034
BATCH_SIZE=128
NEPOCHS=20
MODEL_TYPE=cnn1d

for TARGET in 1 2 3 4 6 7 8 9 12 13 14 15 16 17 19; do
#TEST_TARGET=${TARGET}
#VALID_TARGET=${}

python mat2h5Conversion1D.py

python main.py --lr ${LR} --batch_size ${BATCH_SIZE} --wd ${WD} --model_type ${MODEL_TYPE} --n_epochs ${NEPOCHS}

python test_performance.py --model_type ${MODEL_TYPE} --batch_size ${BATCH_SIZE}
done
