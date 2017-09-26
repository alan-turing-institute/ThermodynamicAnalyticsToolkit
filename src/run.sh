#!/bin/bash

test ! -z "$1" || { echo "Usage: $0 <type no.>"; exit 255; }

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

python3 playground_example.py --dimension 500 --data_type $typeno --noise 0.1 --hidden_dimension $hidden_dim --learning_rate 0.03 --max_steps $max_steps --csv_file test.csv --log_dir=$logdir
