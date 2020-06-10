import sys
import os
import shutil
from os import listdir, sep
from os.path import isdir, join, isfile
from logger import Logger
import glob
import json
import librosa
import meta
from argparse import ArgumentParser

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

def preprocess(args):
    '''
    Preprocess function for extracting usable wav speaker files from Spraakbanken Danish speech data.
    The function expects the data_path to reference a folder containing Stasjon folders from Spraakbanken.
    This function will create a new folder and copy valid speaker wav files into the folder in the structure:
    Spraakbanken-Corpus/<speaker_name>/<wav_file>

    :param data_path:   The path to the folder containing the Statsjon speech data.
    '''
    log = Logger()
    out_put_folder = args.out_dir
    if not isdir(out_put_folder):
        os.mkdir(out_put_folder)
    log.write_line(f'# Preprocessing of {data_path}')
    log.write_line(f'Danish preprocessed data from Spraakbanken is output to: {out_put_folder}')

    #Search for all speaker audio files
    speaker_paths = glob.glob(join(args.data_dir, f'*{sep}*{sep}*{sep}speech{sep}*{se[}*{sep}*{sep}r*'))
    for sp in speaker_paths:
        speaker_id = sp.split(sep)[-1]
        out_speaker = join(out_put_folder, speaker_id)
        if not isdir(out_speaker):
            log.write_line(f'Found new speaker {speaker_id}', verbose=True)
            wav_files = glob.glob(join(sp, '*.wav'))
            if len(wav_files) > 1:
                #Change the path to fit your own file structure if needed
                spl_file = glob.glob(join(args.data_dir, f'*{sep}*{sep}*{sep}data{sep}*{sep}*{sep}*{sep}{speaker_id}.spl'))[0]
                #Save speaker and utterance meta data as json
                meta_data = meta.read_spl_file(speaker_id, spl_file)
                if 'dialect' not in meta_data.keys():
                    log.write_line(f'Speaker {speaker_id} did not have a registered dialect')
                    continue
                else:
                    log.write_line(f'Copying to {out_speaker}...', verbose=True)
                    os.mkdir(out_speaker)
                    with open(join(out_speaker, f'{speaker_id}_meta.json'), 'w+') as meta_file:
                        meta_file.write(json.dumps(meta_data, indent=4))

                    count = 0
                    for wav in wav_files:
                        if is_valid_wav(wav):
                            count += 1
                            filename = wav.split(sep)[-1]
                            shutil.copy(wav, join(out_speaker, f'{speaker_id}_{filename}'))
                            print(f'Copying file {count}...', end='\r')
                        else:
                            print(f'File {wav} was invalid. Could not be opened by librosa')


if __name__ == '__main__':
    '''
    Preprocessing speaker data for wav files stored in the same directory structure as Spraakbanken NST.
    '''
    parser =  ArgumentParser()
    parser.add_argument('-data_dir', '-d', default='/work1/s183921/speaker_data/Spraakbanken-Raw')
    parser.add_argument('-out_dir', '-o', default='/work1/s183921/speaker_data/Spraakbanken-Corpus')
    args = parser.parse_args()
    preprocess(args)
