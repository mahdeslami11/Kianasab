import random
import os
import json
from collections import defaultdict
import glob

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
    for path in sorted(glob.glob(os.path.join(root_dir, '*/*.wav'))):
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

def load_from_dir(data_dir, test_speakers):
    print('Reading speaker ids')
    speaker_ids = read_speaker_info(data_dir)
    random.shuffle(speaker_ids)

    print('Splitting into train and test speakers')
    train_speaker_ids = speaker_ids[:-test_speakers]
    test_speaker_ids = speaker_ids[-test_speakers:]

    print('Reading speaker2filenames')
    speaker2filenames = read_filenames(data_dir)

    return train_speaker_ids, test_speaker_ids, speaker2filenames

def load_from_json(data_dir):
    with open(data_dir, 'r') as speakers:
        print('Reading train and test speakers')
        train_test = json.loads(speakers.read())
        train_speaker_ids = train_test['train']
        test_speaker_ids = train_test['test']
        print('Combining speaker ids')
        speaker_ids = train_speaker_ids + test_speaker_ids
        print('Reading speaker2filenames')
        speaker2filenames = read_json_filenames(train_test)

    return train_speaker_ids, test_speaker_ids, speaker2filenames
