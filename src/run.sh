#!/bin/bash

test ! -z "$1" || { echo "Usage: $0 <type no.>"; exit 255; }

dim="500"
batch_size="10"
optimizer="StochasticGradientLangevinDynamics" #StochasticGradientLangevinDynamics, GradientDescent
learning_decay="1."
learning_decay_power="-0.55"
learning_rate="0.03"

typeno="$1"
case "$typeno" in
	0)
		hidden_dim="3"
		max_steps="100"
		;;
	1)
		hidden_dim="4"
		max_steps="100"
		;;
	2)
		hidden_dim="2"
		max_steps="20"
		;;
	3)
		hidden_dim="8 8"
		max_steps="2000"
		;;
	*)
		echo "Invalid type no."
		exit 255
		;;
esac

logdir="/tmp/tensorflow/Playground/playground_type$typeno"
rm -rf $logdir

python3 playground_example.py \
    --batch_size $batch_size \
    --dimension $dim \
    --data_type $typeno \
    --hidden_dimension $hidden_dim \
    --learning_decay $learning_decay \
    --learning_decay_power $learning_decay_power \
    --learning_rate $learning_rate \
    --max_steps $max_steps \
    --noise 0.1 \
    --optimizer $optimizer \
    --csv_file ../test.csv \
    --log_dir=$logdir
