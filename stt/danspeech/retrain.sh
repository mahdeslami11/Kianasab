#!/bin/bash
. danspeech.config

python3 retrain.py -t $output_dir -l $output_dir/logs
