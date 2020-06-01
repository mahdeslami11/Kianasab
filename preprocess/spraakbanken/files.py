import sys
import os
import shutil
from os import listdir, sep
from os.path import isdir, join, isfile
from logger import Logger
import glob
import json
import librosa

def is_valid_wav(fpath):
    '''
    Test if a given wav file is valid by testing if scipy can read the file.

    :param fpath:   Path of the wav file to validate

    Returns:        True if the wav file can be opened without any exception errors.
                    False otherwise.
    '''
    try:
        librosa.load(fpath)
        return True
    except:
        return False

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
    log.write_line(f'# Preprocessing of {data_path}')
    log.write_line(f'Danish preprocessed data from Spraakbanken is output to: {out_put_folder}')

    #Search for all speaker audio files
    speaker_paths = glob.glob('/work1/s183921/speaker_data/Spraakbanken-Raw/*/*/*/speech/*/*/*/r*') 
    for sp in speaker_paths:
        speaker_id = sp.split('/')[-1]
        out_speaker = join(out_put_folder, speaker_id)
        if not isdir(out_speaker):
            log.write_line(f'Found new speaker {speaker_id}', verbose=True)
            wav_files = glob.glob(join(sp, '*.wav'))
            if len(wav_files) > 1:
                #Save speaker and utterance meta data as json
<<<<<<< HEAD
                meta_data = meta.read_spl_file(speaker_id)
                if 'region of dialect' not in meta_data.keys():
                    log.write_line(f'Speaker {speaker_id} did not have a registered dialect')
                    break
                else:
                    json.dumps(meta_data, join(out_speaker, 'meta.json'))
                    log.write_line(f'Copying to {out_speaker}...', verbose=True)
                    os.mkdir(out_speaker)
                    for i, wav in enumerate(wav_files):
                        if is_valid_wav(wav):
                            shutil.copy(wav, join(out_speaker, f))
                            print(f'Copying file {i+1}...', end='\r')
=======
                with open(join(out_speaker, 'meta.json'), 'w+') as meta_file:
                    meta_file.write(json.dumps(meta.read_spl_file(speaker_id)))
                for i, wav in enumerate(wav_files):
                    if is_valid_wav(wav):
                        shutil.copy(wav, join(out_speaker, f))
                        print(f'Copying file {i+1}...', end='\r')
>>>>>>> 7b858080d10cd1fd3a156cdf9a3284efe1b69c0b


if __name__ == '__main__':
    '''
    Expects this script to be located on the DTU HPC Server with access to the
    work1 scratch
    '''
    speaker_data_path = join(f'{sep}work1', 's183921', 'speaker_data', 'Spraakbanken-Raw')
    preprocess(speaker_data_path)
