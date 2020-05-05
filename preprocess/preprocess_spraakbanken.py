import sys
import os
import shutil
from os import listdir, sep
from os.path import isdir, join, isfile

def use_path_if_valid(station:str, root_folder_path:str, out_put_folder:str, log:Logger):
    '''
    Recursive function which drills down into folders, searching for wav files.
    When found will copy the wav files to a destination defined by the station
    and out_put_folder params.
    The function is desgined for the speech data from Spraakbanken

    :param station:             The station folder from Spraakbanken speech data
    :param root_folder_path:    The current root folder in the recursive process
    :param out_put_folder:      The root folder for copying wav files to.
                                This folder will be where files are copyied
                                to create a new file structure, better suited for
                                doing voice convertion.
    :param log:                 The Logger object used to log what is happening
                                during the copying of wav files
    '''
    log.write_line(f'Searching folder {root_folder_path}', verbose=True)
    dir_content = listdir(root_folder_path)
    log.write_line(f'Folder size: {len(dir_content}', verbose=True)
    for i, f in enumerate(dir_content):
        f_path = join(root_folder_path, f)
        if isdir(f_path):
            use_path_if_valid(station, f_path, out_put_folder, log)
        elif isfile(f_path) and f[-4:] == '.wav' and len(dir_content) > 1:
            root_folder = root_folder_path.split(sep)[-1]
            save_folder = join(out_put_folder, station, root_folder_path)
            if not isdir(save_folder):
                log.write_line(f'Found new wav files', verbose=True)
                log.write_line(f'Copying to {save_folder}...', verbose=True)
                os.mkdir(save_folder)
            shutil.copy(f_path, join(save_folder, f))
            print(f'Copying file {i+1}...', end='\r')
            



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
    if not isdir(out_put_folder):
        os.mkdir(out_put_folder)
    log.write(f'# Preprocessing of {data_path}')
    log.write(f'Danish preprocessed data from Spraakbanken is output to: {out_put_folder}')

    stasjon_folders = [f for f in listdir(data_path) if isdir(f) and 'stasjon' in f.lower()]
    log.write(f'Found folders {stasjon_folders} for extracting speech data')

    for f in stasjon_folders:
        use_path_if_valid(station=f, root_folder_path=join(data_path, f), 
                out_put_folder=out_put_folder, log=log)


if __name__ == '__main__':
    '''
    Expects this script to be located on the DTU HPC Server with access to the
    work1 scratch
    '''
    speaker_data_path = join('work1', 's183921', 'speaker_data', 'Spraakbanken-Raw')
    preprocess(speaker_data_path)
