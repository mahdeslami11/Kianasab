import sys
import os
from os import listdir, sep
from os.path import isdir, join, isfile
from logger import Logger
import glob
import json
import librosa
import soundfile as sf
import meta
from argparse import ArgumentParser

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
    log.write_line(f'# Preprocessing of {args.data_dir}')
    log.write_line(f'Danish preprocessed data from Spraakbanken is output to: {out_put_folder}')

    #Search for all speaker audio files
    speaker_paths = glob.glob(join(args.data_dir, f'*{sep}*{sep}*{sep}speech{sep}*{sep}*{sep}*{sep}r*'))
    for sp in speaker_paths:
        split_path = sp.split(sep)
        station = split_path[5]
        substation = split_path[6]
        speaker_id = split_path[-1]
        out_speaker = join(out_put_folder, f'{station}_{substation}_{speaker_id}')
        if not isdir(out_speaker):
            log.write_line(f'Found new speaker {speaker_id}', verbose=True)
            wav_files = glob.glob(join(sp, '*.wav'))
            if len(wav_files) > 1:
                print(f'{station}{sep}{substation}{sep}*{sep}data{sep}*{sep}*{sep}*{sep}{speaker_id}.spl')
                #Change the path to fit your own file structure if needed
                spl_file = glob.glob(join(args.data_dir, 
                    f'{station}{sep}{substation}{sep}*{sep}data{sep}*{sep}*{sep}*{sep}{speaker_id}.spl'))[0]
                #Save speaker and utterance meta data as json
                meta_data = meta.read_spl_file(speaker_id, spl_file)
                if 'dialect' not in meta_data.keys():
                    log.write_line(f'Speaker {speaker_id} did not have a registered dialect')
                    continue
                else:
                    log.write_line(f'Copying to {out_speaker}...', verbose=True)
                    os.mkdir(out_speaker)
                    with open(join(out_speaker, f'{station}_{substation}_{speaker_id}_meta.json'), 'w+') as meta_file:
                        meta_file.write(json.dumps(meta_data, indent=4))

                    count = 0
                    for wav in wav_files:
                        try:
                            count += 1
                            filename = wav.split(sep)[-1]
                            x, _ = librosa.load(wav, sr=16000)
                            sf.write(join(out_speaker, f'{station}_{substation}_{speaker_id}_{filename}'), x, 16000)
                            print(f'Copying file {count}...', end='\r')
                        except:
                            print(f'File {wav} was invalid')


if __name__ == '__main__':
    '''
    Preprocessing speaker data for wav files stored in the same directory structure as Spraakbanken NST.
    '''
    parser =  ArgumentParser()
    parser.add_argument('-data_dir', '-d', default='/work1/s183921/speaker_data/Spraakbanken-Raw')
    parser.add_argument('-out_dir', '-o', default='/work1/s183921/speaker_data/Spraakbanken-Corpus')
    args = parser.parse_args()
    preprocess(args)
