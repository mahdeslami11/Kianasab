import os
import re
import numpy as np
import json
from tacotron.utils import get_spectrograms
from tinydb import TinyDB
import speaker
import features
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-data_dir', '-d', default='/work1/s183921/speaker_data/Spraakbanken-Corpus')
    parser.add_argument('-out_dir', '-o', default='/work1/s183921/preprocessed_data/vae/spraakbanken')
    parser.add_argument('-validation_speakers', '-vs', default=20, type=int)
    parser.add_argument('-test_proportion', '-tp', default=0.1, type=int)
    parser.add_argument('-sample_rate', '-sr', default=16000, type=int)
    parser.add_argument('-n_utts_attr', '-u', default=5000, type=int)
    parser.add_argument('-database_name', '-db', default='spectrograms.json')

    args = parser.parse_args()

    #Json database for saving speaker wav spectrograms
    db = TinyDB(os.path.join(args.out_dir, args.database_name))

    train_path_list, in_test_path_list, out_test_path_list = speaker.train_test_validation_split(args.data_dir, args.out_dir)

    features.extract(train_path_list, in_test_path_list, out_test_path_list, args.n_utts_attr, db)
