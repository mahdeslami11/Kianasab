. preprocess.config

if [ $stage -le 0 ]; then
    python dataset/main.py -d $raw_data_dir -o $data_dir -vs $validation_speakers -ts $test_prop -sr $sample_rate -u $n_utt_attr
fi

if [ $stage -le 1 ]; then
    python dataset/reduce.py $data_dir/train.json $data_dir/train_$segment_size.json $segment_size
fi

if [ $stage -le 2 ]; then
    # sample training samples
    python dataset/segments.py $data_dir/train.json $data_dir/train_samples_$segment_size.json $training_samples $segment_size
fi
if [ $stage -le 3 ]; then
    # sample testing samples
    python dataset/segments.py $data_dir/in_test.json $data_dir/in_test_samples_$segment_size.json $testing_samples $segment_size
    python dataset/segments.py $data_dir/out_test.json $data_dir/out_test_samples_$segment_size.json $testing_samples $segment_size
fi
