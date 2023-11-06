LR=0.0001
WD=0.01
BATCH_SIZE=32
NEPOCHS=600
MODEL_TYPE=lstm

#python mat2h5Conversion.py
#python mat2h5Conversion_ObsTrain.py

python main.py --lr ${LR} --batch_size ${BATCH_SIZE} --wd ${WD} --model_type ${MODEL_TYPE} --n_epochs ${NEPOCHS}

python test_performance.py --model_type ${MODEL_TYPE}
