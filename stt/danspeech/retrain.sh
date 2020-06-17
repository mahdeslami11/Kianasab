#!/bin/bash
. danspeech.config

python3 retrain.py -t $output_dir -l /work1/s183921/preprocessed_data/danspeech/spraakbanken/original/logs
