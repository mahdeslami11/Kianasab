import os
import re
import numpy as np
import json
from tacotron.utils import get_spectrograms
from tinydb import TinyDB
import speaker
from argparse import ArgumentParser


def spec_feature_extraction(wav_file):
    '''
    Extracts the mel and magnitude spectrogram from the given audio file

    :param wav_file:    The audio file to extract spectrogram from

    returns the mel and magnitude spectrogram of the given audio file
    '''
    mel, mag = get_spectrograms(wav_file)
    return mel, mag

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-data_dir', '-d', default='/work1/s183921/speaker_data/Spraakbanken-Corpus')
    parser.add_argument('-out_dir', '-o', default='/work1/s183921/preprocessed_data/vae/spraakbanken')
    parser.add_argument('-test_speakers', '-ts', default=20, type=int)
    parser.add_argument('-test_proportion', '-tp', default=0.1, type=int)
    parser.add_argument('-sample_rate', '-sr', default=16000, type=int)
    parser.add_argument('-n_utts_attr', '-u', default=5000, type=int)

######Feature extraction, mean and variance vectors, saved as pickle
    for dset, path_list in zip(['train', 'in_test', 'out_test'], \
            [train_path_list, in_test_path_list, out_test_path_list]):

        print(f'processing {dset} set, {len(path_list)} files')
        data = {}
        output_path = os.path.join(output_dir, f'{dset}.pkl')
        all_train_data = []
        for i, path in enumerate(sorted(path_list)):
            if i % 500 == 0 or i == len(path_list) - 1:
                print(f'processing {i} files')
            filename = path.strip().split('/')[-1]
            mel, mag = spec_feature_extraction(path)
            data[filename] = mel
            if dset == 'train' and i < n_utts_attr:
                all_train_data.append(mel)
        #Extrating mean and standard deviation for training data and saves it in .pkl
        if dset == 'train':
            all_train_data = np.concatenate(all_train_data)
            mean = np.mean(all_train_data, axis=0)
            std = np.std(all_train_data, axis=0)
            attr = {'mean': mean, 'std': std}
            with open(os.path.join(output_dir, 'attr.pkl'), 'wb+') as f:
                pickle.dump(attr, f)
        #Normalizing mel spectrogram data
        for key, val in data.items():
            val = (val - mean) / std
            data[key] = val
        with open(output_path, 'wb+') as f:
            pickle.dump(data, f)
#############################################################################################
