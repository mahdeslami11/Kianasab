"""
Are you interested in the word error rate, (wer) of a set of voice files?

call the function: weighted_mass_wer(transcription_dir, voice_dir, sphinx = True, getdata = False)

transcription_dir is a string containing the path to the true transcriptions of the voice clips.
The transcriptions MUST be contained as plain text in .txt files. one .txt file per voice clip.
The .txt files must be sorted in same alphabetic order as the voice clips.
Just name the .txt files the same as their corrosponding voice clip.

voice_dir is a string containing the path to the voice clip folder. The voice clip must be of type .wav
The voice clips MUST be sorted in same alphabetic order as their transcriptions.

If sphinx = True the speech to text convertion is carried out but sphinx.
If sphinx = False the conversion will be done by danspeech

if getdata = True: the true transcription aswell as the speech to text conversion will be returned
as 2 strings.
"""

"""
Are you interested in comparing two set of voice conversions against each other?
How about a p-value stating how likely the two voice conversions are to be different?

call the function: mcnemar_v2(transcription_dir, voice_dir1, voice_dir2, sphinx = True)

transcription_dir is the same as above

voice_dir1 and voice_dir2 is the same as above, you you just have one for the location of the first 
set of voice conversions and for the location of the second set of voice conversions.

sphinx = True/False again the same as above. 
"""


import os
import jiwer
from scipy.stats import chi2

from danspeech import Recognizer
from danspeech.pretrained_models import TestModel
from danspeech.language_models import DSL3gram
from danspeech.audio import load_audio
import librosa
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")

#transcriptions location
transcription_dir = "/home/karl/Desktop/auto_examples_python/transcriptions"

#voice data
voice_dir = "/home/karl/Desktop/auto_examples_python/voice_data"

# helper function for mcnemar test. calculates intersection including duplicates between two lists:
def intersec(true_transcript, proposed_transcript):

    intersec_ = list()
    if type(true_transcript) == str: true_transcript = true_transcript.split(" ")
    if type(proposed_transcript) == str: proposed_transcript = proposed_transcript.split(" ")

    for x in true_transcript:
        if x in proposed_transcript:
            proposed_transcript.remove(x)
            intersec_.append(x)

    return intersec_

# helper function to mcnemar test. calculates number of uniques in list1 with duplicates
def uniques(list1, list2):
    list1_count = []

    if type(list1) == str: list1 = list1.split(" ")
    if type(list2) == str: list2 = list2.split(" ")

    for word in list1:
        if word in list2:
            list2.remove(word)
        else:
            list1_count.append(word)

    return len(list1_count)

def mcnemar(true_transcript, prosed1, prosed2):
    prosed1_true = intersec(true_transcript, prosed1)
    prosed2_true = intersec(true_transcript, prosed2)
    b = uniques(prosed1_true, prosed2_true)
    c = uniques(prosed2_true, prosed1_true)

    test_statistic = (abs(b-c)-1)**2 / (b+c)

    return (1 - chi2.cdf(test_statistic, df = 1))

# just some test data
#transcript = ['this', 'what', 'a', 'list', 'of', 'words', 'become', 'will', 'soon', 'become', 'and', 'bonus', 'list']
#list1 = ['this', 'is', 'a', 'list', 'of', 'words', 'but', 'will', 'soon', 'become', 'and', 'actual', 'list']
#list2 = ['of', 'words', 'will', 'soon', 'become', 'and', 'list', 'but', 'this', 'some', 'words', 'aswell', "bonus"]

#print(intersec(transcript, list1))
#print(intersec(transcript, list2))
#print(mcnemar(transcript, list1, list2))

# helper function to only find males or females
def find_male_female(l1, string):
    for keyword in l1:
        if keyword in string:
            return True
    return False

# calculating wer as a weighted sum of all individual wers from each transctiption/vc pair
# region ["fyn", "jylland", "lolland", "sjælland", "sønderjylland"]
def weighted_mass_wer(transcription_dir,
                      voice_dir,
                      danspeechpro = False,
                      getdata = False,
                      print_transcipt_and_vc = False,
                      region = None,
                      sex = None):

    t_dir_list = sorted(os.listdir(transcription_dir))
    t_dir_list = [t_file for t_file in t_dir_list if t_file[-3:] == "txt"]

    vc_dir_list = sorted(os.listdir(voice_dir))
    vc_dir_list = [vc_file for vc_file in vc_dir_list if vc_file[-3:] == "wav"]

    if region is not None:
        t_dir_list = [t_file for t_file in t_dir_list if t_file[:3] == region[:3]]
        vc_dir_list = [vc_file for vc_file in vc_dir_list if vc_file[:3] == region[:3]]

    if sex in ["k", "k1", "female", "kvinde"]:
        t_dir_list = [t_file for t_file in t_dir_list if find_male_female(["k1","k2","k3","k4"], t_file)]
        vc_dir_list = [vc_file for vc_file in vc_dir_list if find_male_female(["k1","k2","k3","k4"], vc_file)]

    if sex in ["m", "m1", "male", "mand"]:
        t_dir_list = [t_file for t_file in t_dir_list if find_male_female(["m1","m2","m3","m4"], t_file)]
        vc_dir_list = [vc_file for vc_file in vc_dir_list if find_male_female(["m1","m2","m3","m4"], vc_file)]



    voice_conversion_total = ""
    ground_truth_total = ""

    weight_accumulated = 0
    WER = 0

    model = TestModel()
    recognizer = Recognizer(model=model)

    if danspeechpro:
        try:
            lm = DSL3gram()
            recognizer.update_decoder(lm=lm, alpha=1.2, beta=0.15, beam_width=10)
        except ImportError:
            print("ctcdecode not installed. Using greedy decoding.")
            return None

    for t_file, vc_file in zip(t_dir_list, vc_dir_list):
        t_file_location = os.path.join(transcription_dir, t_file)
        ground_truth = open(t_file_location, 'r').read()[:-1]
        ground_truth = ground_truth.lower()
        ground_truth = ground_truth.replace(",", "")
        ground_truth = ground_truth.replace(".", "")
        ground_truth_total = ground_truth_total + " " + ground_truth
        weight = len(ground_truth.split())
        weight_accumulated += weight

        # load audio to danspeech
        vc_file_location = os.path.join(voice_dir, vc_file)

        x, _ = librosa.load(vc_file_location, sr=16000)
        sf.write(vc_file_location, x, 16000)
        audio = load_audio(path=vc_file_location)

        # do the text to speech with danspeech
        voice_conversion = recognizer.recognize(audio, show_all=False)
        voice_conversion_total = voice_conversion_total + " " + voice_conversion

        if print_transcipt_and_vc:
            print(t_file, ground_truth)
            print(vc_file, voice_conversion)
            print("********************")

        WER += jiwer.wer(ground_truth, voice_conversion) * weight
        
    voice_conversion_total = voice_conversion_total[1:]
    ground_truth_total = ground_truth_total[1:]
    
    if getdata:
        return WER/weight_accumulated, ground_truth_total, voice_conversion_total

    else:
        return WER/weight_accumulated


def mcnemar_v2(transcription_dir, orginial, converted, danspeechpro=True, region=None, print_transcipt_and_vc = True, sex = None):
    wer1, true_transcript, prosed1 = weighted_mass_wer(transcription_dir,
                                                       orginial,
                                                       danspeechpro=danspeechpro,
                                                       getdata= True,
                                                       print_transcipt_and_vc = print_transcipt_and_vc,
                                                       region=region,
                                                       sex = sex)

    wer2, _, prosed2 = weighted_mass_wer(transcription_dir,
                                         converted,
                                         danspeechpro=danspeechpro,
                                         getdata=True,
                                         print_transcipt_and_vc =
                                         print_transcipt_and_vc,
                                         region=region,
                                         sex = sex)

    print(wer1)
    print(wer2)
    print(mcnemar(true_transcript, prosed1, prosed2))

#print(weighted_mass_wer(transcription_dir, voice_dir))
#print(mass_wer(transcription_dir, voice_dir))

original = "/home/karl/Desktop/train/Testdata/VC_original"
converted = "/home/karl/Desktop/train/Testdata/VC_converted_StarGAN_r6110050_25-Test-Final-Training"
true_transcriptions = "/home/karl/Desktop/train/Testdata/VC_transcriptions"

mcnemar_v2(true_transcriptions,
           original,
           converted,
           danspeechpro=True,
           print_transcipt_and_vc=True,
           region=None,
           sex=None
           )

