
'''
Dette script indeholder en funktion find_test_data_v3()
Denne funktion tager 2 paths som indput, en til en mappe med lyd data og en til en mappe med de tilhørende
transkriptioner i. Transkriptionerne skal være i et meget specifikt json format. Disse filer kan genereres med
filen meta.py
Når man kører funktionen outputter den en sorteret liste i en txt fil som efterfølgende kan inspires
for at se om der er overenstemmelse mellem lyd of textfiler.



'''


import jiwer
from danspeech import Recognizer
from danspeech.pretrained_models import DanSpeechPrimary
from danspeech.language_models import DSL3gram
from danspeech.audio import load_audio
import os
import json
import regex
from more_itertools import sort_together


import librosa
import soundfile as sf

import warnings
warnings.filterwarnings("ignore")

def find_test_data_v3(voice_data_dir, json_data_dir):
    model = DanSpeechPrimary()
    recognizer = Recognizer(model=model)
    try:
        lm = DSL3gram()
        recognizer.update_decoder(lm=lm, alpha=0.65, beta=0.65, beam_width=227)
    except ImportError:
        print("ctcdecode not installed. Using greedy decoding.")

    # given the path to a folder containing all speakers of interest find there corrospoding json meta data file

    n_possible_audio_files = 0
    possiible_true_transcript = []
    possiible_danspeech_transcript = []
    possiible_audio_file = []
    WER_list = []
    speaker_id_list = []
    sentence_lenght = []

    skipped = 0

    for c, audio_file in enumerate(sorted([path for path in os.listdir(voice_data_dir) if not path.endswith("001.wav")])):
        if c%10 == 0:
            print(c)

        speaker_id = audio_file[:-13] + "_meta.json"
        audio_file_json_key = audio_file[-12:]

        # find corrosponding jsonfile
        json_file_location = os.path.join(json_data_dir, speaker_id)

        # open jsonfile and filter all non audiofiles out
        with open(json_file_location, 'r') as f:
            transcript_dict = json.load(f)

        # load and resave the audiofile so danspeech can read it
        x, _ = librosa.load(os.path.join(voice_data_dir, audio_file), sr=16000)
        sf.write('tmp.wav', x, 16000)

        # load audio and make speech to text with danspeech
        danspeech_transcript = recognizer.recognize(load_audio("tmp.wav"))

        try:
            true_transcript = transcript_dict[audio_file_json_key]
            true_transcript = true_transcript.lower()
            true_transcript = true_transcript.replace(".", "")
            true_transcript = true_transcript.replace("é", "e")
            true_transcript = true_transcript.replace("\\", " ")
            true_transcript = true_transcript.replace(",", "")
            true_transcript = true_transcript.replace("?", "")
            true_transcript = true_transcript.replace("!", "")
            true_transcript = regex.sub(' +', ' ', true_transcript)

            WER = jiwer.wer(true_transcript, danspeech_transcript)
            WER_list.append(WER)
            sentence_lenght.append(len(true_transcript.split()))
            possiible_true_transcript.append(true_transcript)
            possiible_danspeech_transcript.append(danspeech_transcript)
            possiible_audio_file.append(audio_file)
            speaker_id_list.append(speaker_id[:-5])

        except KeyError:
            skipped +=1
            print("KEY ERROR HAPPENED")
            print(audio_file_json_key)
            print(speaker_id)


    print("skipped: ", skipped)

    # sorting the data according to biggest WER
    sorted_according_to_WER = sort_together([WER_list,
                                             possiible_true_transcript,
                                             possiible_danspeech_transcript,
                                             possiible_audio_file,speaker_id_list],
                                            reverse=True)

    WER_list = sorted_according_to_WER[0]
    possiible_true_transcript = sorted_according_to_WER[1]
    possiible_danspeech_transcript = sorted_according_to_WER[2]
    possiible_audio_file = sorted_according_to_WER[3]
    speaker_id_list = sorted_according_to_WER[4]


    print("possiible_audio_file: ", possiible_audio_file)
    print("sentence_lenght: ", sentence_lenght)
    print("possiible_true_transcript: ", possiible_true_transcript)
    print("possiible_danspeech_transcript: ", possiible_danspeech_transcript)

    # saving possible audio files for futher inspection
    file = open("ISDANSPEECHANYGOOD.txt", "a+")

    file.write(str(speaker_id_list) + "\n")
    file.write(str(possiible_audio_file) + "\n")
    file.write(str(WER_list) + "\n")
    file.write(str(sentence_lenght)+ "\n")
    file.write(str(possiible_true_transcript) + "\n")
    file.write(str(possiible_danspeech_transcript) + "\n")
    file.write("################################" + "\n")


    for i in range(len(WER_list)):
        file.write(str(speaker_id_list[i]) + "\n")
        file.write(str(possiible_audio_file[i])+"\n")
        file.write(str(WER_list[i])+"\n")
        file.write(str(sentence_lenght[i]) + "\n")
        file.write(str(possiible_true_transcript[i])+"\n")
        file.write(str(possiible_danspeech_transcript[i])+"\n")
        file.write("################################"+"\n")
    file.close()



voice_data_dir = "/home/karl/Desktop/train/test_samples"
json_data_dir = "/home/karl/Desktop/train/VC_training_all_json"

find_test_data_v3(voice_data_dir, json_data_dir)






