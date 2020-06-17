import sys
sys.path.append('danspeech_training/deepspeech')
from train import train_new
from argparse import ArgumentParser, Namespace
import torch
if torch.cuda.is_available():
    print('Using cuda')
    print(f'Cuda device count: {torch.cuda.device_count()}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-training_path', '-t', default='/work1/s183921/preprocessed_data/danspeech/spraakbanken')
    args = parser.parse_args()

    train_new(model_id=None, 
            train_data_path=args.training_path, 
            validation_data_path=args.training_path)
