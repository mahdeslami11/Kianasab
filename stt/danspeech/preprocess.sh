#!/bin/bash
. danspeech.config

python3 preprocess.py -m $meta_data -o $output_dir -c $csv_name -ow True
