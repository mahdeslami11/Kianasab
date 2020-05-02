import sys
from os import listdir, sep
from os.path import isdir, join

def save_path_if_valid(root_folder_path:str):
    return None #TODO implement

def preprocess(data_path:str):
    '''
    Preprocess function for extracting usable wav speaker files from Spraakbanken Danish speech data.
    The function expects the data_path to reference a folder containing Stasjon folders from Spraakbanken.
    This function will create a new folder and copy valid speaker wav files into the folder in the structure:
    Spraakbanken-Corpus/<speaker_name>/<wav_file>

    :param data_path:   The path to the folder containing the Statsjon speech data.
    '''
    log = Logger()
    out_put_folder = join(data_path.rsplit(sep, 1)[0], 'Spraakbanken-Corpus')
    log.write(f'# Preprocessing of {data_path}')
    log.write(f'Danish preprocessed data from Spraakbanken is output to: {out_put_folder}')

    stasjon_folders = [f for f in listdir(data_path) if isdir(f) and 'stasjon' in f.lower()]
    log.write(f'Found folders {stasjon_folders} for extracting speech data')

    for f in stasjon_folders:
        save_path_if_valid(f)


if __name__ == '__main__':
    '''
    Expects this script to be located on the DTU HPC Server with access to the
    work1 scratch
    '''
    speaker_data_path = join('work1', 's183921', 'speaker_data', 'Spraakbanken-Raw')
    preprocess(speaker_data_path)
