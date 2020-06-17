import sys
sys.path.append('danspeech_training/deepspeech')
from train import train_new
from argparse import ArgumentParser, Namespace
import torch
import os

if torch.cuda.is_available():
    print('Using cuda')
    print(f'Cuda device count: {torch.cuda.device_count()}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-training_path', '-t', default='/work1/s183921/preprocessed_data/danspeech/spraakbanken/original')
    parser.add_argument('-log_dir', '-l', default='/work1/s183921/preprocessed_data/danspeech/spraakbanken/original/logs')
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)

    train_new(model_id='original_retrain', 
            train_data_path=args.training_path, 
            validation_data_path=args.training_path,
            model_save_dir=args.training_path,
            tensorboard_log_dir=args.log_dir)
