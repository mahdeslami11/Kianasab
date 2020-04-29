import sys
from logger import Logger
from os import listdir
from os.path import join, isdir

def save_path_if_valid(log:Logger, folder_path:str):
    num_of_wav = 0

    with open(join('vctk', 'speaker_paths_vctk.txt'), 'w+') as sp:
        for wav in listdir(folder_path):
            if wav[-4:] == '.wav':
                sp.write(f'{join(folder_path, wav)}\n')
                num_of_wav += 1

    log.write_line(f'Found {num_of_wav} wav files for speaker')


def preprocess_vctk(data_path:str):
    '''
    Preprocess function for extracting usable wav speaker files from VCTK speech data.
    The function expects the data_path to reference a folder containing the unzip VCTK speaker data folder.

    :param data_path:   The path to the folder containing the VCTK speech data.
    '''
    log = Logger()
    log.write_line(f'# Preprocessing of {data_path}')
    log.write_line(f'A txt file of the VCTK preprocessed data is output to: ./vctk\n')

    num_of_speakers = 0

    log.write_line('## Speaker Data')
    for f in listdir(data_path):
        f = join(data_path, f)
        if isdir(f) and 'p' in f:
            log.write_line(f'Found speaker folder {f}')
            num_of_speakers += 1
            save_path_if_valid(log, f)

    log.write_line(f'Found {num_of_speakers} speakers in total')
    log.write_line(f'Preprocessing ended')

if __name__ == '__main__':
    data_path = sys.argv[1]
    preprocess_vctk(data_path)
