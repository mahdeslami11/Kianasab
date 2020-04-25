import sys
from os import listdir
from os.path import join, isdir

def save_path_if_valid(root_folder_path:str):
    return None #TODO implement

def preprocess_vctk(data_path:str):
    '''
    Preprocess function for extracting usable wav speaker files from VCTK speech data.
    The function expects the data_path to reference a folder containing the unzip VCTK speaker data folder.

    :param data_path:   The path to the folder containing the VCTK speech data.
    '''
    log = Logger()
    log.write(f'# Preprocessing of {data_path}')
    log.write(f'A json file of the VCTK preprocessed data is output to: ./vctk')

    speaker_folders = [f for f in listdir(data_path) if isdir(f) and 'p' in f]
    log.write(f'Found speaker folders {speaker_folders} for extracting speech data')

    for f in speaker_folders:
        save_path_if_valid(f)

if __name__ == '__main__':
    data_path = sys.argv[1]
    preprocess_vctk(data_path)
