#!/bin/bash
#Main shell script for running the preprocessing of speaker data for VAE
. preprocess.config

if [ $stage -le 0 ]; then
    python3 dataset/make.py $raw_data_dir $data_dir $n_out_speakers $test_prop $sample_rate $n_utt_attr
fi

if [ $stage -le 1 ]; then
    python3 dataset/reduce.py $data_dir/train.pkl $data_dir/train_$segment_size.pkl $segment_size
fi

if [ $stage -le 2 ]; then
    # sample training samples
    python3 dataset/segments.py $data_dir/train.pkl $data_dir/train_samples_$segment_size.json $training_samples $segment_size
fi
if [ $stage -le 3 ]; then
    # sample testing samples
    python3 dataset/segments.py $data_dir/in_test.pkl $data_dir/in_test_samples_$segment_size.json $testing_samples $segment_size
    python3 dataset/segments.py $data_dir/out_test.pkl $data_dir/out_test_samples_$segment_size.json $testing_samples $segment_size
fi
