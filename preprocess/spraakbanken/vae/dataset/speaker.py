import os
import utils
import random

def __train_test_split(train_speaker_ids, speaker2filenames, test_proportion):
    train_path_list, test_path_list = [], []

    #Randomly shuffling audio files for each speaker and extracting test and training data
    for speaker in train_speaker_ids:
        path_list = speaker2filenames[speaker]
        random.shuffle(path_list)
        test_data_size = int(len(path_list) * test_proportion) #test_proportion defined in config file
        train_path_list += path_list[:-test_data_size]
        test_path_list += path_list[-test_data_size:]

    return train_path_list, test_path_list

def __validation_data(test_speaker_ids, speaker2filenames):
    out_test_path_list = []
    #Validation speakers, never seen by the model before
    for speaker in test_speaker_ids:
        path_list = speaker2filenames[speaker]
        out_test_path_list += path_list

    return out_test_path_list


def train_test_validation_split(data_dir, path_output_dir, test_proportion):
    '''
    Reads data from a data location with speaker audio files and 
    splits it into training, test, and validation speaker data.

    :param data_dir:        The location of the speaker audio data to split
                            into training, test, and validation data
    :param path_out_dir:    The out put location for where to write text files with
                            the paths of the test and validation data

    :returns:               
    '''
    if data_dir.endswith('.json'):
        print(f'Reading from json train_test file: {data_dir}')
        train_speaker_ids, test_speaker_ids, speaker2filenames = utils.load_from_json(data_dir)
    else:
        print(f'Reading from directory: {data_dir}')
        train_speaker_ids, test_speaker_ids, speaker2filenames = utils.load_from_dir(data_dir)

    print('Splitting training data into training and test')
    print(f'Using {1-test_proportion} training and {test_proportion} test proportions')
    train_data_paths, test_data_paths = __train_test_split(
            train_speaker_ids, speaker2filenames, test_proportion)

    print('Finding validation speaker paths')
    validation_data_paths = __validation_data(test_speaker_ids, speaker2filenames)

    #Test speakers
    with open(os.path.join(path_output_dir, 'in_test_files.txt'), 'w+') as f:
        for path in test_data_paths:
            f.write(f'{path}\n')

    #Validation speakers
    with open(os.path.join(path_output_dir, 'out_test_files.txt'), 'w+') as f:
        for path in validation_data_paths:
            f.write(f'{path}\n')

    return train_data_paths, test_data_paths, validation_data_paths
