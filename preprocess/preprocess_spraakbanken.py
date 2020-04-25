import sys
from os import listdir
from os.path import isdir

def save_path_if_valid(root_folder_path:str):
    return None #TODO implement

def preprocess_spraakbanken(data_path:str):
    '''
    Preprocess function for extracting usable wav speaker files from Spraakbanken Danish speech data.
    The function expects the data_path to reference a folder containing Stasjon folders from Spraakbanken.

    :param data_path:   The path to the folder containing the Statsjon speech data.
    '''
    log = Logger()
    log.write(f'# Preprocessing of {data_path}')
    log.write(f'Danish preprocessed data from Spraakbanken is output to: ./spraakbanken')

    stasjon_folders = [f for f in listdir(data_path) if isdir(f) and 'stasjon' in f.lower()]
    log.write(f'Found folders {stasjon_folders} for extracting speech data')

    for f in stasjon_folders:
        save_path_if_valid(f)


if __name__ == '__main__':
    data_path = sys.argv[1]
    preprocess_spraakbanken(data_path)
