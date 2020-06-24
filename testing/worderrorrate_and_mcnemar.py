"""
dette script indeholder en række hjælpe- og hovedfunktunktioner.

generelt om de stiger som de forskellige funktioner tager som inputs.
Alle stiger til mappen med lyddata samt stigen til med transckriptions filer bgives som strings. transkriptionsfilerne
skal være txt filer. en per lydfil. lyd og transkriptions filer skal have samme alferbetiske rækkefølge således de
matches korrekt af funktionerne. lydfilernme skal være 16 khz og a typen .wav


1
weighted_mass_wer()
denne funktion beregner den samlede vægtede WER for en mappe med lydfiler.
Den tager 2 ikke standart inputs.
Den tager som input stigen til en mappe med lydfiler og stigen til mappen med de tilhørende transkriptionsfiler.

funktionen bruges primækrt som hjælpe funktion men kan godt høres for sig selv.

2
mcnemar_v2()
denne funktion sammenligner 1 mappe med orginale lyd filder med dens tilhørende mappe med VC lydfiler.
Den beregner WER for begge mapper med lydfiler samt udføres en Mcnemar test mellem de 2.

den tager tre stiger. stigen til mappen med originale filer. stigen til mappen med de converterede filer.
stigen til mappen med de fælles transkriptionsfiler


3
WER_all_combiantions()
denne funktion bruges til at generere data til de plots der i rapporten udgør resultaterne til
research question 2. dataen gemmes i en txt fil iterativt, samt printes den i konsollen.
dataen skal manuelt overføres til plot funktionen ved navn plot_WER()
funktionen tager de samme argumenter som mcnemar_v2.

4
plot_WER()
denne funktionen laver de plots der i rapporten findes under resultaterne til
research question 2.
funktionen tager ingen inputs.
dataen fra WER_all_combiantions()'s udprint eller txt fil skal overføres manuelt ind i denne funktion.

5
WER_all_combiantions_retrained()
denne funktion bruges til at generere data til de plots der i rapporten udgør resultaterne til
research question 3. dataen gemmes i en txt fil iterativt, samt printes den i konsollen.
dataen skal manuelt overføres til plot funktionen ved navn plot_WER_part2()
funktionen tager de samme argumenter som mcnemar_v2.
funktionen tager yderligere 2 stigerne til henholdsvis den originalt danspeech retrænede model
og den VC-convertede retrænede danspeech model. stigerne skal være strings.

6
plot_WER_part2()
denne funktionen laver de plots der i rapporten findes under resultaterne til
research question 3.
funktionen tager ingen inputs.
dataen fra WER_all_combiantions_retrained()'s udprint eller txt fil skal overføres manuelt ind i denne funktion.




"""


import os
import jiwer
from scipy.stats import chi2

from danspeech.deepspeech.model import DeepSpeech
from danspeech.utils.data_utils import get_model
from danspeech import Recognizer
from danspeech.pretrained_models import DanSpeechPrimary
from danspeech.pretrained_models import TestModel
from danspeech.language_models import DSL3gram
from danspeech.audio import load_audio
import librosa
import soundfile as sf
import warnings
import matplotlib.pyplot as plt
import numpy as np


plt.style.use('ggplot')
warnings.filterwarnings("ignore")


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
                      sex = None,
                      specific_model = None
                      ):

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

    weight_accumulated = 1
    WER = 1

    if specific_model in ["TestModel", "testModel"] or specific_model == None:
        model = DanSpeechPrimary()

    elif specific_model in ["DanSpeechprimary", "danspeechprimary", "DanSpeechPrimary"]:
        model = DanSpeechPrimary()

    else:
        model = DeepSpeech.load_model(specific_model)

    recognizer = Recognizer(model=model)
    if danspeechpro:
        try:
            lm = DSL3gram()
            recognizer.update_decoder(lm=lm, alpha=0.65, beta=0.65, beam_width=227)
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

# mcnemar_v2(true_transcriptions,
#            original,
#            converted,
#            danspeechpro=True,
#            print_transcipt_and_vc=True,
#            region="lolland",
#            sex=None
#            )

transcription_path = "/home/karl/Desktop/train/Testdata/VC_transcriptions"
original_path = "/home/karl/Desktop/train/Testdata/VC_original"
stargan_path = "/home/karl/Desktop/train/Testdata/VC_converted_StarGAN_r6110050_25-Test-Final-Training"
VAE_path = "/home/karl/Desktop/train/Testdata/VC_converted_VAE"

space_of_intereste = ["fyn", "jylland", "lolland", "sjælland", "sønderjylland", "female", "male", "all"]

def WER_all_combiantions(transcription_path, original_path, stargan_path, VAE_path, space_of_intereste):

    original_WER = []
    stargan_WER = []
    VAE_WER = []

    for element in space_of_intereste:
        region = None
        sex = None

        if element in ["fyn", "jylland", "lolland", "sjælland", "sønderjylland"]:
            region = element
            print("--------------\n",region, "\n--------------")

        if element in ["m", "m1", "male", "mand", "k", "k1", "female", "kvinde"]:
            sex = element
            print("--------------\n", sex, "\n--------------")

        print("ORIGINAL")
        original_WER.append(weighted_mass_wer(transcription_path,
                          original_path,
                          danspeechpro=True,
                          getdata= False,
                          print_transcipt_and_vc = True,
                          region=region,
                          sex = sex))

        print("STARGAN")
        stargan_WER.append(weighted_mass_wer(transcription_path,
                          stargan_path,
                          danspeechpro=True,
                          getdata= False,
                          print_transcipt_and_vc = True,
                          region=region,
                          sex = sex))

        print("VAE")
        VAE_WER.append(weighted_mass_wer(transcription_path,
                          VAE_path,
                          danspeechpro=True,
                          getdata= False,
                          print_transcipt_and_vc = True,
                          region=region,
                          sex = sex))

        print(space_of_intereste, original_WER, stargan_WER, VAE_WER)
    
        file = open("WER_all.txt", "a+")
        file.write("------------------------------------\n")
        file.write(str(space_of_intereste) + "\n")
        file.write(str(original_WER) + "\n")
        file.write(str(stargan_WER) + "\n")
        file.write(str(VAE_WER) + "\n")
        file.write("------------------------------------\n")
        file.close()

    return (space_of_intereste, original_WER, stargan_WER, VAE_WER)

# WER_all_combiantions(transcription_path, original_path, stargan_path, VAE_path, space_of_intereste)




def plot_WER():

    space_of_intereste = ['Nord- og midtjylland', 'Fyn', 'Syd- og vestsjælland', 'Storkøbenhavn', 'Sønderjylland', 'Female', 'Male', 'Combined']
    original_WER = [0.6263736263736264, 0.5783582089552238, 0.4218009478672986, 0.37790697674418605, 0.6201550387596899, 0.5211267605633803, 0.5657439446366782, 0.5419407894736842]
    stargan_WER = [0.7967032967032966, 0.7350746268656716, 0.5781990521327014, 0.6976744186046512, 0.7467700258397932,0.7230046948356808, 0.7058823529411765, 0.7146381578947368]
    VAE_WER = [0.9505494505494505, 0.9440298507462687, 0.8720379146919431, 1.0, 1.0, 0.918622848200313, 0.9377162629757786,0.9276315789473685]




    locations = np.arange(len(space_of_intereste))
    width = 0.25
    width_adj = 0.03

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.bar(locations[:-2]-width, original_WER[:-3] + [original_WER[-1]], width=width-width_adj, label="Original", color="coral")
    ax.bar(locations[:-2], stargan_WER[:-3] + [stargan_WER[-1]], width=width-width_adj, label= "StarGan" ,color ="cornflowerblue")
    ax.bar(locations[:-2]+width, VAE_WER[:-3] + [VAE_WER[-1]], width=width-width_adj, label = "VAE", color = "lightsteelblue")
    ax.set_aspect(2)
    ax.set_xticks(locations)
    ax.set_xticklabels(space_of_intereste[:-3]+[space_of_intereste[-1]])
    ax.legend(borderpad=1,framealpha=10,edgecolor= "dimgray")
    ax.set_ylim(0, 1.4)
    ax.set_xlim(0-0.5, len(space_of_intereste[:-3]+[space_of_intereste[-1]])-0.5)
    ax.axhline(1, linestyle= "--", color= "dimgray")
    ax.set_title("WER across dialects")
    ax.set_ylabel("WER")
    plt.show()



    locations = np.arange(3)
    width = 0.25
    width_adj = 0.03

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.bar(locations-width, original_WER[-3:], width=width-width_adj, label="Original", color="coral")
    ax.bar(locations, stargan_WER[-3:], width=width-width_adj, label= "StarGan" ,color ="cornflowerblue")
    ax.bar(locations+width, VAE_WER[-3:], width=width-width_adj, label = "VAE", color = "lightsteelblue")
    ax.set_aspect(2)
    ax.set_xticks(locations)
    ax.set_xticklabels(space_of_intereste[-3:])
    ax.legend(borderpad=1,framealpha=10,edgecolor= "dimgray")
    ax.set_ylim(0, 1.4)
    ax.set_xlim(0-0.5, 2.5)
    ax.axhline(1, linestyle= "--", color= "dimgray")
    ax.set_title("WER across gender")
    ax.set_ylabel("WER")
    plt.show()

# plot_WER()











transcription_path = "/home/karl/Desktop/train/Testdata/VC_transcriptions"
original_path = "/home/karl/Desktop/train/Testdata/VC_original"
stargan_path = "/home/karl/Desktop/train/Testdata/VC_converted_StarGAN_r6110050_25-Test-Final-Training"
# VAE_path = "/home/karl/Desktop/train/Testdata/VC_converted_VAE"

space_of_intereste = ["jylland", "fyn" ,"lolland", "sjælland", "sønderjylland", "female", "male", "all"]


def WER_all_combiantions_retrained(transcription_path,
                                   original_path,
                                   stargan_path,
                                   space_of_intereste,
                                   original_model,
                                   StarGAN_model
                                   ):
    original_WER = []
    stargan_WER = []

    for element in space_of_intereste:
        region = None
        sex = None

        if element in ["fyn", "jylland", "lolland", "sjælland", "sønderjylland"]:
            region = element
            print("--------------\n", region, "\n--------------")

        if element in ["m", "m1", "male", "mand", "k", "k1", "female", "kvinde"]:
            sex = element
            print("--------------\n", sex, "\n--------------")

        print("ORIGINAL")
        original_WER.append(weighted_mass_wer(transcription_path,
                                              original_path,
                                              danspeechpro=True,
                                              getdata=False,
                                              print_transcipt_and_vc=True,
                                              region=region,
                                              sex=sex,
                                              specific_model=original_model))

        print("STARGAN")
        stargan_WER.append(weighted_mass_wer(transcription_path,
                                             stargan_path,
                                             danspeechpro=True,
                                             getdata=False,
                                             print_transcipt_and_vc=True,
                                             region=region,
                                             sex=sex,
                                             specific_model=StarGAN_model))

        print(space_of_intereste, original_WER, stargan_WER)

        file = open("Q3_WER_all.txt", "a+")
        file.write("------------------------------------\n")
        file.write(str(space_of_intereste) + "\n")
        file.write(str(original_WER) + "\n")
        file.write(str(stargan_WER) + "\n")
        file.write("------------------------------------\n")
        file.close()

    return (space_of_intereste, original_WER, stargan_WER)


# WER_all_combiantions_retrained(transcription_path,
#                                original_path,
#                                stargan_path,
#                                space_of_intereste,
#                                "/home/karl/Downloads/newest_models/william_models_wed/danspeech_baseline_william_cleaned_original.pth",
#                                "/home/karl/Downloads/newest_models/william_models_wed/danspeech_william_cleaned_converted.pth"
#                                )


def plot_WER_part2():
    # OLD
    # space_of_intereste = ['Nord- og midtjylland', 'Fyn', 'Syd- og vestsjælland', 'Storkøbenhavn', 'Sønderjylland',
    #                       'Female', 'Male', 'Combined']
    # original_WER = [0.8846153846153846, 0.8805970149253731, 0.6540284360189573, 0.7325581395348837, 0.8217054263565892, 0.7715179968701096, 0.8356401384083045, 0.8018092105263158]
    # stargan_WER = [0.9725274725274725, 0.914179104477612, 0.8056872037914692, 0.9011627906976745, 0.9276485788113695, 0.9107981220657277, 0.9013840830449827, 0.90625]
    # VAE_WER = [1.0, 1.0, 0.990521327014218, 1.0, 1.0, 0.9906103286384976, 1.0, 0.9950657894736842]


    space_of_intereste = ['Nord- og midtjylland', 'Fyn', 'Syd- og vestsjælland', 'Storkøbenhavn', 'Sønderjylland',
                          'Female', 'Male', 'Combined']

    original_WER = [0.7197802197802198, 0.6044776119402985, 0.47393364928909953, 0.43023255813953487, 0.6330749354005168, 0.5727699530516432, 0.5934256055363322, 0.5822368421052632]
    stargan_WER = [0.8626373626373627, 0.7350746268656716, 0.5213270142180095, 0.5872093023255814, 0.7545219638242894, 0.6917057902973396, 0.71280276816609, 0.7014802631578947]



    locations = np.arange(len(space_of_intereste))
    width = 0.25
    width_adj = 0.03
    spacing = 0.6

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.bar(locations[:-2] - width*spacing*0.985, original_WER[:-3] + [original_WER[-1]], width=width - width_adj, label="Conventional STT setup",
           color="coral")
    ax.bar(locations[:-2] + width*spacing, stargan_WER[:-3] + [stargan_WER[-1]], width=width - width_adj, label="Implemented StarGan VC STT setup",
           color="cornflowerblue")

    ax.set_aspect(2)
    ax.set_xticks(locations)
    ax.set_xticklabels(space_of_intereste[:-3] + [space_of_intereste[-1]])
    ax.legend(borderpad=1, framealpha=10, edgecolor="dimgray")
    ax.set_ylim(0, 1.4)
    ax.set_xlim(0 - 0.5, len(space_of_intereste[:-3] + [space_of_intereste[-1]]) - 0.5)
    ax.axhline(1, linestyle="--", color="dimgray")
    ax.set_title("WER of conventional and implemented VC STT setup")
    ax.set_ylabel("WER")
    plt.show()

    locations = np.arange(3)
    width = 0.25
    width_adj = 0.03
    spacing = 0.65

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.bar(locations - width*spacing*0.992, original_WER[-3:], width=width - width_adj, label="Conventional STT setup", color="coral")
    ax.bar(locations + width*spacing, stargan_WER[-3:], width=width - width_adj, label="Implemented StarGan VC STT setup", color="cornflowerblue")

    ax.set_aspect(2)
    ax.set_xticks(locations)
    ax.set_xticklabels(space_of_intereste[-3:])
    ax.legend(borderpad=1, framealpha=10, edgecolor="dimgray")
    ax.set_ylim(0, 1.4)
    ax.set_xlim(0 - 0.5, 2.5)
    ax.axhline(1, linestyle="--", color="dimgray")
    ax.set_title("WER of conventional and implemented VC STT setup")
    ax.set_ylabel("WER")
    plt.show()

# plot_WER_part2()




