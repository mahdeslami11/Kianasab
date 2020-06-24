'''
The script is created from an original implementation by: 
https://github.com/jjery2243542/adaptive_voice_conversion

This is the main script for the VAE speaker preprocessing.
It selects .wav files from speakers in a given directory and
generates mel spectrograms for them. From the mel spectrograms
segments are selected to be used for the training of VAE.
This script is set up to work with Spraakbanken speaker data
which is 16 kHz recordings. It is important to adjust the 
sampling rate in /tacotron/hyperparams.py and
../preprocess.config if any other audio frequencies are used.

IMPORTANT:
Even though the script is run on Spraakbanken data it does not fit
the original folder structure of Spraakbanken. Instead it fits that
of VCTK (see https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)

See the script ../../files.py which can be used to select files from Spraakbanken
and move them to a folder structure similar to that of VCTK.
'''
import pickle 
import sys
import glob 
import random
import os
from collections import defaultdict
import re
import numpy as np
import json
from tacotron.utils import get_spectrograms

def read_speaker_info(speaker_info_path):
    '''
    Reads all lines of a text-file. Each line is regarded as a path to a speaker audio file.
    Each speaker file path is expected to have the speaker id in the beginning of the path.

    :param speaker_info_path:   Path to the text file containing the lines of speaker audio file paths.
    returns a collection of speaker ids
    '''
    speaker_ids = []
    for speaker_id in os.listdir(speaker_info_path):
        speaker_id = speaker_id.strip()
        speaker_ids.append(speaker_id)
    return speaker_ids

def read_filenames(root_dir):
    '''
    Creates a map with speaker ids as keys and a collection with the speaker's audio file paths.
    
    :param root_dir:   The root dir of the speaker corpus 

    returns a dictionary with speaker ids mapped to a collection of speaker audio file paths
    '''
    speaker2filenames = defaultdict(lambda : [])
    paths = sorted(glob.glob(os.path.join(root_dir, 'Stasjon*/*.wav')))
    for path in paths:
        speaker_id = path.strip().split('/')[-2]
        speaker2filenames[speaker_id].append(path)
    return speaker2filenames

def read_json_filenames(json_obj):
    '''
    Creates a map with speaker ids as keys and a collection with the speaker's audio file paths.
    
    :param json_obj:    A json object containing:
                        :data_dir:  Path to speaker data folder
                        :train:     Ids of speakers to train on
                        :test:      Ids of speakers to test on

    returns a dictionary with speaker ids mapped to a collection of speaker audio file paths
    '''
    data_dir = json_obj['data_dir']
    speaker2filenames = defaultdict(lambda : [])
    for speaker_id in json_obj['train']:
        speaker2filenames[speaker_id] = glob.glob(
                os.path.join(data_dir, speaker_id, '*.wav'))
    for speaker_id in json_obj['test']:
        speaker2filenames[speaker_id] = glob.glob(
                os.path.join(data_dir, speaker_id, '*.wav'))

    return speaker2filenames


def spec_feature_extraction(wav_file):
    '''
    Extracts the mel and magnitude spectrogram from the given audio file

    :param wav_file:    The audio file to extract spectrogram from

    returns the mel and magnitude spectrogram of the given audio file
    '''
    mel, mag = get_spectrograms(wav_file)
    return mel, mag

if __name__ == '__main__':
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    test_speakers = int(sys.argv[3])
    test_proportion = float(sys.argv[4])
    sample_rate = int(sys.argv[5])
    n_utts_attr = int(sys.argv[6])

    print('Reading speaker ids')
    print(data_dir[-5:])
    if not data_dir.endswith('.json'):
        speaker_ids = read_speaker_info(data_dir)
        random.shuffle(speaker_ids)

        train_speaker_ids = speaker_ids[:-test_speakers]
        test_speaker_ids = speaker_ids[-test_speakers:]

        print('Reading speaker2filenames')
        speaker2filenames = read_filenames(data_dir)
    else:
        with open(data_dir, 'r') as speakers:
            train_test = json.loads(speakers.read())
            train_speaker_ids = train_test['train']
            test_speaker_ids = train_test['test']
            speaker_ids = train_speaker_ids + test_speaker_ids
            speaker2filenames = read_json_filenames(train_test)

    #Speaker file extraction
    train_path_list, in_test_path_list, out_test_path_list = [], [], []

    #Randomly shuffling audio files for each speaker and extracting test and training data
    for speaker in train_speaker_ids:
        path_list = speaker2filenames[speaker]
        random.shuffle(path_list)
        test_data_size = int(len(path_list) * test_proportion)
        train_path_list += path_list[:-test_data_size]
        in_test_path_list += path_list[-test_data_size:]

    #Train speakers
    with open(os.path.join(output_dir, 'in_test_paths.txt'), 'w+') as f:
        for path in in_test_path_list:
            f.write(f'{path}\n')

    #Test speakers
    for speaker in test_speaker_ids:
        path_list = speaker2filenames[speaker]
        out_test_path_list += path_list

    with open(os.path.join(output_dir, 'out_test_paths.txt'), 'w+') as f:
        for path in out_test_path_list:
            f.write(f'{path}\n')

    #Feature extraction, mean and variance vectors, saved as pickle
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

