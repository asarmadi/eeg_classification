LR=0.001
WD=0.1
BATCH_SIZE=128

#python mat2h5Conversion.py

python main.py --lr ${LR} --batch_size ${BATCH_SIZE} --wd ${WD}
