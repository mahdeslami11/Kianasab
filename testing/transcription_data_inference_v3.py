'''

dette scrip har 4 funktioner som når de bliver kørt i forlængelse af hinanden
tilsammen laver den type plot der findes i projektets data afsnit.

Det eneste eksterne der skal til for at køre funktiorne er "json_path" som er string med stigen til
en mappe med transkriptioner (i form af json filer) som man ønsker at plotte. Transkriptionerne skal være i et
meget specifikt json format. Disse filer kan genereres med filen meta.py

funktionerne i den korrekt forlængelse af hinannden ser således ud:

figure, axes  = plt.subplots(2,2)
trans_data    = load_json(json_path)
figure, axes  = age_sex_dialect_distribution(trans_data, ["age", "sex", "dialect"], figure, axes)
all_sentences = word_and_sentence_lists(trans_data)
word_and_sentence_distribution(all_sentences, figure, axes, filter_sentences=False, print_most_frequent=False)


'''




import os
import json
import numpy as np
import matplotlib.pyplot as plt
import regex
import pandas as pd
# plotting style
plt.style.use('ggplot')

# helper function to make plt.text text look sweet
def equallenght(t1,t2):
    if len(t1) > len(t2):
        while len(t1) != len(t2):
            t2 = t2.replace(" ", "  ",1)
    else:
        while len(t2) != len(t1):
            t1 = t1.replace(" ", "  ",1)

    return t1 + "\n" + t2

# loading all json to a list of dictionaries
def load_json(json_path, exclude = [], only_include = []):

    # regarding excluding and including
    # Define a list of files names to be excluded or included. Only one OR the other, not both.

    json_files = os.listdir(json_path)
    json_files = sorted([file for file in json_files if file[-4:] == "json"])
    trans_data = []

    # loading all json to a list of dictionaries
    for json_file in json_files:
        if len(only_include) != 0 and len(exclude) != 0:
            print("You are both trying to include and explude, this is not allowed. No data was loaded")
            return None

        # use all but the excludes
        if len(exclude) != 0:
            if json_file not in exclude:
                sample = os.path.join(json_path, json_file)
                with open(sample, 'r') as f:
                    trans_data.append(json.load(f))

        # only use includes
        if len(only_include) != 0:
            if json_file in only_include:
                sample = os.path.join(json_path, json_file)
                with open(sample, 'r') as f:
                    trans_data.append(json.load(f))

        # no excludes or specifc includes
        if len(only_include) == 0 and len(exclude) == 0:
            sample = os.path.join(json_path, json_file)
            with open(sample, 'r') as f:
                trans_data.append(json.load(f))

    return trans_data

# get age, sex and dialect distribution
# key is a list containing any or all three keywords ["age", "sex", "dialect"]
def age_sex_dialect_distribution(trans_data, keys, figure, axes):

    for keyword in keys:
        if keyword not in ["age", "sex", "dialect"]:
            print("Invalid keyword. Valid keywords are age, sex and dialect")
            return None

        data = []
        unknown_count = 0
        for trans_dict in trans_data:
            if keyword == "age":
                try:
                    data.append(int(trans_dict[keyword]))

                except ValueError:
                    unknown_count += 1

            if keyword == "sex" or keyword == "dialect":
                value = trans_dict[keyword]
                if len(value) != 0 and type(value) == str:
                    data.append(value)
                else:
                    unknown_count += 1


        # plot of age distribtuion
        if keyword == "age":
            i = 22.5
            data = np.asarray(data)
            counts = np.array([])
            bins = np.array([])
            while i < 85:

                count = np.sum(data < i) - np.sum(counts)
                counts = np.append(counts, count)
                bins = np.append(bins, i-2.5)
                i += 5

            counts = np.append(counts, unknown_count)
            bin_labels = np.append(bins.astype(int), "Unknown")
            bin_labels[-2] = " "
            bins = np.append(bins, bins[-1]+5)

            barplot = axes[1,0].bar(bins, counts, color="cornflowerblue", width=2.5)
            barplot[-1].set_color('slategray')
            axes[1,0].set_xticks(bins)
            axes[1,0].set_xticklabels(bin_labels)
            axes[1,0].set_title("Age distribution")
            # axes[1,0].gcf().subplots_adjust(bottom=0.30)
            # plt.xlabel("Age")
            # plt.show()

        # plot of sex distributions
        # if keyword == "sex":
        #     counts = [data.count("male"), data.count("female"), unknown_count]
        #     bins = [5,10,15]
        #
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111)
        #     ax.set_aspect(aspect=2.5)
        #
        #     barplot = plt.bar(bins, counts, color="cornflowerblue", width=2.5)
        #     barplot[-1].set_color('slategray')
        #     plt.xticks(bins, ["Male", "Female", "Unknown"], rotation=45)
        #     plt.title("Gender distribution")
        #     plt.xlim(0,20)
        #     plt.gcf().subplots_adjust(bottom=0.15)
        #     plt.show()

        if keyword == "dialect":
            bin_labels = list(set(data))
            # bin_labels.remove("østjylland")
            # bin_labels.remove("nordjylland")
            # bin_labels.remove("vestjylland")
            # bin_labels.append("Midt- og nordjylland")
            bin_labels.append("Unknown")
            bin_labels = [label.capitalize() for label in bin_labels]
            bins = [x*5 for x in list(range(1,len(bin_labels)+1))]

            counts = []
            # nord_øst_jylland = 0
            for dialect in list(set(data)):
                    counts.append(data.count(dialect))
            # counts.append(nord_øst_jylland)
            counts.append(unknown_count)

            barplot = axes[1,1].bar(bins, counts, color="cornflowerblue", width=1.5)
            barplot[-1].set_color('slategray')
            axes[1,1].set_xticks(bins)
            axes[1,1].set_xticklabels(bin_labels, rotation= 45)
            axes[1,1].set_title("Dialect distribution")
            # axes[1,1].gcf().subplots_adjust(bottom=0.30)
            # plt.show()
    return figure, axes

# distribution of words and sentences
def word_and_sentence_lists(trans_data):
    all_sentences = []
    all_words = []

    for trans in trans_data:
        sentences = list(trans.values())[6:]
        for sentence in sentences:
            sentence = sentence.lower()
            sentence = sentence.replace(".", "")
            sentence = sentence.replace("é", "e")
            sentence = sentence.replace("\\", " ")
            sentence = sentence.replace(",", "")
            sentence = sentence.replace("?", "")
            sentence = sentence.replace("!", "")
            sentence = sentence.replace("\"", "")
            sentence = regex.sub(' +', ' ', sentence)
            all_sentences.append(sentence)

    return all_sentences

def word_and_sentence_distribution(all_sentences, figure, axes, filter_sentences=False, print_most_frequent=True):
    min_words_in_sentence = 3
    num_most_common_words_to_print = 10
    num_most_sentences_words_to_print = 10

    most_commen = ['adskillige af de huse som kommunen byggede sidste år er blevet beskadiget under den sidste tids stormvejr',
                'jeg er ikke særlig interesseret i at stå og fryse her mens de andre sidder og hygger sig derhjemme foran pejsen',
                'der var både blodpølse og sylte på bordet men han sagde at han bare ville spise varm mad såsom suppe',
                'tester en to tre fire fem seks syv otte',
                'gaden er lukket for gennemgående færdsel ti måneder om året men man kan køre udenom hvis man drejer til højre i krydset',
                'tror du at han kan flytte den store kasse og dunkene med vand ned i kælderen uden at tabe det hele',
                'hvad synes du om den nye bank der åbnede i går dernede på hjørnet over for kirken',
                'anden gang jeg var i hjørring mødte jeg både troels og per hos fiskehandlerne',
                'han løb ud i det lille køkken for at sige tak for kaffe til pers kone men hun var ikke til at se',
                'dan havde lavet en flot tegning af en sol og en måne med hat og tre øjne som blev hængt op i skolen så alle kunne se den',
                'da de var kommet ud til den gamle mølle uden for byen vendte de om og kørte hjem igen']

    # filter sentences:
    if filter_sentences:
        print("Sentences will be filtered")
        temp = []
        c = 0
        for i in all_sentences:
            # sentence must not be among the common ones
            # sentence must contain more than 2 words.
            # sentence must have more than 3 characters per space
            if i not in most_commen and i.count(" ") >= min_words_in_sentence and len(i) / i.count(" ") > 3:
                temp.append(i)
        print( len(all_sentences)- len(temp), "sentences omitted\n")
        all_sentences = temp

    all_words = ' '.join(all_sentences)
    all_words = all_words.split()

    print("Number of words:", len(all_words))
    print("Number of sentences:", len(all_sentences))
    print("Number of unique words:", len(set(all_words)))
    print("Number of unique sentences:", len(set(all_sentences)),"\n")

    word_count = pd.Series(all_words).value_counts()
    word_count_list = list(word_count.items())
    sentence_count = pd.Series(all_sentences).value_counts()
    sentence_count_list = list(sentence_count.items())

    if print_most_frequent:
        if num_most_common_words_to_print > 0:
            print(num_most_common_words_to_print, "Most common words")
            for i in word_count_list[:num_most_common_words_to_print]:
                print(i)
            print("\n")

        if num_most_sentences_words_to_print > 0:
            print(num_most_sentences_words_to_print, "Most common sentences")
            for i in sentence_count_list[:num_most_sentences_words_to_print]:
                print(i)

    # plotting words
    bin_locs = []
    bin_labels = []
    counts = []
    for index in range(num_most_common_words_to_print):
        bin_labels.append(word_count_list[index][0])
        counts.append(word_count_list[index][1])
        bin_locs.append((index+1)*5)

    axes[0,0].bar(bin_locs, counts, color="cornflowerblue", width=2.5)
    axes[0,0].set_xticks(bin_locs)
    axes[0,0].set_xticklabels(bin_labels)
    axes[0,0].set_title("Most common words")
    text = "total words:  " + str(len(all_words))+"\n" + "unique words:  " + str(len(set(all_words)))
    axes[0,0].text(.80,.90,
             text,
             bbox = dict(boxstyle='round', facecolor='cornflowerblue', alpha=0.5), ha='center', va='center', transform=axes[0,0].transAxes)
    # plt.show()

    # plotting sentence lenght:
    sentence_lenght = []
    for sentence in (all_sentences):
        # setting a limit for sentence lenght to make the plot look nice
        if sentence.count(" ")+1 < 35:
            sentence_lenght.append(sentence.count(" ")+1)


    axes[0,1].hist(sentence_lenght, bins=(max(sentence_lenght) - min(sentence_lenght)), rwidth=0.85, color="cornflowerblue")

    bin_labels = [x for x in range(0,max(sentence_lenght)+1) if x%5 == 0]
    bin_locs = [x+.5 for x in bin_labels]

    axes[0,1].set_xticks(bin_locs)
    axes[0,1].set_xticklabels(bin_labels)
    axes[0,1].set_title("Sentence length distribution")
    text = ("total sentences:    " + str(len(all_sentences)) + "\n" +  "unique sentences: " + str(len(set(all_sentences))))
    axes[0,1].text(.80,.90,
             text,
             bbox = dict(boxstyle='round', facecolor='cornflowerblue', alpha=0.5),
             ha='center', va='center', transform=axes[0,1].transAxes)

    # figure.subplots_adjust(hspace=0.25)
    # figure.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.05, bottom=0.15, right=0.95, top=0.95)

    plt.show()

# json folder path
json_path = "/home/karl/Desktop/train/VC_training_all_json"
# json_path = "/home/karl/Desktop/train/VC_training_all_json"
json_path = "/home/karl/Desktop/train/danspeech_training_all_json"
json_path = "/home/karl/Desktop/train/json_final"


figure, axes = plt.subplots(2,2)
trans_data = load_json(json_path)
figure, axes = age_sex_dialect_distribution(trans_data, ["age", "sex", "dialect"], figure, axes)
all_sentences = word_and_sentence_lists(trans_data)
word_and_sentence_distribution(all_sentences, figure, axes, filter_sentences=False, print_most_frequent=False)












