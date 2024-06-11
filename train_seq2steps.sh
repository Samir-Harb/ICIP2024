#!/bin/sh 
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
#python /home/assale02/LandmarkDetector/test.py > /home/assale02/LandmarkDetector/log.log
echo "$1, $2"
python train_seq2steps.py "$1" "$2" > ./log/log_seq_"$1"_"$2"_2steps.txt 2>&1
# python train.py "$1" "$2" > log"$1"_"$2".txt 

#python /home/assale02/LandmarkDetector/test_complete_train.py