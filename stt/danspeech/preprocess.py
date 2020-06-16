from argparse import ArgumentParser, Namespace
import os
import meta
import audio
         

def __create_project_dirs(dirs):
    for d in dirs:
        if not os.path.isdir(d):
            os.mkdir(d)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-meta_data',       '-m' , default='/work1/s183921/speaker_data/Spraakbanken-Selected/all_json')
    parser.add_argument('-out_dir',         '-o' , defalut='/work1/s183921/preprocessed_data/danspeech/spraakbanken/original')
    parser.add_argument('-csv',             '-c' , default'training.csv')     
    parser.add_argument('-overwrite',       '-ow', default=True, type=bool)

    args = parser.parse_args()

    print('Creating folders')
    __create_project_dirs([args.out_dir])

    print('Preprocessing meta data')
    meta.preprocess(args)
    print('Preprocessing audio files')
    audio.preprocess(args)
