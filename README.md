# Voice Conversion - DTU Project Work 4th Semester
This repo contains the code for the project undertaken 
as part of the course Project Work in Artificial Intelligence at the Technical University of Denmark.<br>
For English voice conversion the [VCTK Multi-speaker Corpus](https://datashare.is.ed.ac.uk/handle/10283/3443) is used.<br>
For Danish voice conversion the [Språkbankens ressurskatalog](https://www.nb.no/sprakbanken/show?serial=oai%3Anb.no%3Asbr-19&lang=en) is used.

## Project Statement
This project specifically looks at, to which degree voice conversion technologies can be utilized for transforming dialect heavy speech into a standard voice, to improve the performance of the existingDanish state-of-the-art speech to text system by danspeech.

### Research Questions
* Is it possible to reproduce test results achieved by the state-of-the-art voice conversion models: [StarGAN-VC](https://arxiv.org/abs/1806.02169) and [Instance-normalization](https://arxiv.org/abs/1904.05742)?
* How does the [danspeech](https://github.com/danspeech) speech to text translation perform when applying voice conversion models compared to using no voice conversion?
* Which pipe line architecture seems to be preferable when introducing voice conversion in the danspeech speech to text framework?

# Overall Architecture
![The overall architecture of the process supported by the code in this repository](/img/architecture.png)

# Repository Content
The repository is structured as follows

### Voice Conversion Models Folder
The following models are used with custom implementations, such that the preprocessing of data and training of the models are customized to the specific topic of this research project. 

* StarGAN
    - Implementation of the [StarGAN Voice Conversion Project](https://github.com/liusongxiang/StarGAN-Voice-Conversion) from the [StarGAN-VC Paper] (https://arxiv.org/abs/1806.02169).
* SSCR
    - Implementation of the [One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization](https://github.com/jjery2243542/adaptive_voice_conversion) from the [Voice Conversion Paper](https://arxiv.org/abs/1904.05742).

### Preprocess Folder
Module to preprocess wav file data for the StarGAN and SSCR models to consume.
* VCTKPreprocessing
    - Preprocessing of data from the [VCTK Multi-speaker Corpus](https://datashare.is.ed.ac.uk/handle/10283/3443).
* DanishPreprocessing
    - Preprocessing of Danish multi speaker data from the Norwegian National Library [Språkbankens ressurskatalog](https://www.nb.no/sprakbanken/show?serial=oai%3Anb.no%3Asbr-19&lang=en).

### Speech to Text Models Folder
The following models are used for the transformation of original and voice converted speech data to produce a text signal which is used to evalute the speech recognition accuracy when using voice conversion as opposed to no conversion.
* CMU PocketSphinx
    - Implementation of the [CMU PocketSphinx Framework](https://cmusphinx.github.io/) using the Python module [SpeechRecognition](https://pypi.org/project/SpeechRecognition/)
* danspeech
    - Implementation of the [danspeech](https://github.com/danspeech) using the Python module [danspeech](https://pypi.org/project/danspeech/)


### Testing Folder
This module includes code for evaluating and comparing the accuracy of the Speech to Text Models for original and voice converted speech input. 
* McNemar
    - Comparing the final text out put from each speech to text framework when voice input is converted or not using the statistical [McNemar's test](https://en.wikipedia.org/wiki/McNemar%27s_test)
* Word Error Rate
    - Comparing the final text out put from each speech to text framework when voice input is converted or not using the performance metric [Word Error Rate](https://en.wikipedia.org/wiki/Word_error_rate).


### Using StarGAN
There are three runnable scripts, which have been used for this project. 
These are modified of the StarGAN cloned scripts preprocess.py, main.py and convert.py:  
* /fagprojekt2020/preprocess/stargan/stargan_preprocess_spraakbanken.py
    - This script is build to preprocess Spraakbanken speaker data for later training of StarGAN.
    - The script converts 48 kHz wav audio files to 16 kHz, unless data is already 16 kHz. 
    It then extracts the acoustic features (MCEPs, F0) and compute the corresponding stats (means, stds) and saves 
    these to */mc/train and */mc/test.
    - The different speakers must be in seperate folders and speaker_used must designate which to prerpocess.
    - Run in terminal to preprocess speaker_used list inside the script.
    ```
    python3 stargan_preprocess_spraakbanken.py 
    ```
* /fagprojekt2020/vc/stargan/main_spraakbanken.py  
    - This script is used for training the StarGAN-VC model.  
    - List of speakers included in training needs to be designated in /fagprojekt2020/vc/stargan/data_loader.py. 
    These have to be preprocessed.
    - Directories can be designated in terminal using the parser, but this is more easely done in the script.
    Change the defaults to accomodate for training data placement and designated folder for saving models.
    - Run in terminal designating number of training speakers at xx.
    ```
    python3 main_spraakbanken.py --num_speakers xx
    ```
* /fagprojekt2020/vc/stargan/convert_noTrain.py  
    - This script is used for converting .wav files directly, meaning preprocess and converting happens together and no training data for StarGAN model training is produces.  
    - The default iteration model used is 200,000 and its path is needed.
    - The mc folder needs to include the _stats.npz file of target speaker, and the same list of training speakers used for training the model need to be included under the class TestDataset as self.speakers.  
    - Every speaker included in the origin_wavpath will be converted.
    - Run in terminal
    ```
    python3 convert_noTrain.py 
    ```

