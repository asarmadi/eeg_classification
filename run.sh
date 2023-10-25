LR=0.001
WD=0.0
BATCH_SIZE=32
MODEL_TYPE=capsnet

python mat2h5Conversion.py

#python main.py --lr ${LR} --batch_size ${BATCH_SIZE} --wd ${WD} --model_type ${MODEL_TYPE}

#python test_performance.py --model_type ${MODEL_TYPE}
