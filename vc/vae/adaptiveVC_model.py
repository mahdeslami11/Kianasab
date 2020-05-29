import torch
import numpy as np
import sys
import os 
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pickle
from vclab.vc_models.adaptiveVC_model_files.model import AE
from vclab.vc_models.adaptiveVC_model_files.utils import *
from functools import reduce
import json
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from scipy.io.wavfile import write
import random
from vclab.vc_models.adaptiveVC_model_files.preprocess.tacotron.utils import melspectrogram2wav
from vclab.vc_models.adaptiveVC_model_files.preprocess.tacotron.utils import get_spectrograms
import librosa 
from vclab.vc_models.test_model import TestModel

class Inference(TestModel):
    """
    Class representing a rewrite of the inference.py from 
    the voice conversion model in the Adaptive Voice Conversion Paper:
    https://github.com/jjery2243542/adaptive_voice_conversion
    """
    def __init__(self,sample_rate:int=24000):
        """
        Initialization method
        
        :param config_file:     Path to yaml configuration file which will be used to load the settings for 
                                the voice conversion. 
        :param attr_file:       Path to the pickle file generated from the model training used to set attributes
                                for the voice conversion
        :param model_path:      Path to the trained adaptive voice conversion model to use for the voice conversion
        :param sample_rate:     The sample rate to use when saving the converted voice to a wav file
        """
        self.sample_rate = sample_rate
        self.model_path = os.path.join(os.path.join(os.path.dirname(__file__), 'adaptiveVC_model_files'), 'models.ckpt')
        config_file = os.path.join(os.path.join(os.path.dirname(__file__), 'adaptiveVC_model_files'), 'models.config.yaml')
        attr_file = os.path.join(os.path.join(os.path.dirname(__file__), 'adaptiveVC_model_files'), 'attr.pkl')

        with open(config_file) as f:
            config = yaml.load(f)
        # config store the value of hyperparameters, turn to attr by AttrDict
        self.config = config
        print(config)

        # init the model with config
        self.build_model()

        # load model
        self.load_model()

        with open(attr_file, 'rb') as f:
            self.attr = pickle.load(f)

    def load_model(self):
        print(f'Load model from {self.model_path}')
        self.model.load_state_dict(torch.load(self.model_path))

    def build_model(self): 
        # create model, discriminator, optimizers
        self.model = cc(AE(self.config))
        print(self.model)
        self.model.eval()

    def utt_make_frames(self, x):
        frame_size = self.config['data_loader']['frame_size']
        remains = x.size(0) % frame_size 
        if remains != 0:
            x = F.pad(x, (0, remains))
        out = x.view(1, x.size(0) // frame_size, frame_size * x.size(1)).transpose(1, 2)
        return out

    def inference_one_utterance(self, x, x_cond):
        x = self.utt_make_frames(x)
        x_cond = self.utt_make_frames(x_cond)
        dec = self.model.inference(x, x_cond)
        dec = dec.transpose(1, 2).squeeze(0)
        dec = dec.detach().cpu().numpy()
        dec = self.denormalize(dec)
        wav_data = melspectrogram2wav(dec)
        return wav_data, dec

    def denormalize(self, x):
        m, s = self.attr['mean'], self.attr['std']
        ret = x * s + m
        return ret

    def normalize(self, x):
        m, s = self.attr['mean'], self.attr['std']
        ret = (x - m) / s
        return ret

    def write_wav_to_file(self, wav_data, output_path):
        write(output_path, rate=self.sample_rate, data=wav_data)

    def convert(self, source:str, target:str, output_path:str):
        src_mel, _ = get_spectrograms(source)
        tar_mel, _ = get_spectrograms(target)
        src_mel = torch.from_numpy(self.normalize(src_mel)).cuda()
        tar_mel = torch.from_numpy(self.normalize(tar_mel)).cuda()
        conv_wav, conv_mel = self.inference_one_utterance(src_mel, tar_mel)
        print(f'Writing wav file to: {output_path}')
        self.write_wav_to_file(conv_wav, output_path)
