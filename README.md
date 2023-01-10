# Voice Conversion - DTU Project Work 4th Semester
This repo contains the code for the project undertaken 
as part of the course Project Work in Artificial Intelligence at the Technical University of Denmark.<br>
For English voice conversion the [VCTK Multi-speaker Corpus](https://datashare.is.ed.ac.uk/handle/10283/3443) was used for preliminary experiments.<br>
For Danish voice conversion the [Språkbankens ressurskatalog](https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-19/) is used.<br>
To run the full project, you need GPU hardware. This project has been run using the gpuv100 server on the **DTU HPC Server**.


## Project Statement
This project specifically looks at, to which degree voice conversion technologies can be utilized fortransforming dialect heavy speech into a standard voice, to improve the performance of the existingDanish state-of-the-art speech to text system danspeech.

### Research Questions
* How well can state-of-the-art voice conversion results from [StarGAN-VC](https://arxiv.org/abs/1806.02169) and [Instance-normalization](https://arxiv.org/abs/1904.05742) VC models be reproduced for many-to-one, zero-shot voice conversion scenarios?
* How does the [danspeech](https://github.com/danspeech) speech to text translation perform when applying voice conversion models compared to using no voice conversion?
* How does the danspeech speech to text translation perform when voice converted input isprovided to a pretrained danspeech model compared to a danspeech model retrained on voiceconverted data?

# Overall Architecture
![The overall architecture of the process supported by the code in this repository](/img/architecture.png)
The architecture incorperates voice conversion as part of training the STT model and as an added step in the speech to text process to try and create a common voice with better speech to text translation accuracy.

# Repository Content
The repository is structured as follows

### Voice Conversion Models Folder
The following models are used with custom implementations, such that the preprocessing of data and training of the models are customized to the specific topic of this research project. The two VC models used can be found in the /vc folder

* StarGAN
    - Implementation of the [StarGAN Voice Conversion Project](https://github.com/liusongxiang/StarGAN-Voice-Conversion) from the [StarGAN-VC Paper] (https://arxiv.org/abs/1806.02169).
* SSCR
    - Implementation of the [One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization](https://github.com/jjery2243542/adaptive_voice_conversion) from the [Voice Conversion Paper](https://arxiv.org/abs/1904.05742).

### Preprocess Folder
<span style="color.red">IMPORTANT: Due to limitations in the code, you have to create the following files manually in the preprocessed folder before running the preprocess. attr.pkl, in_test.pkl, out_test.pkl, train.pkl. (maybe others as well. The error will show. Sorry for that</span>
<br/>
Module to preprocess wav file data for the StarGAN and SSCR models to consume.
* VCTKPreprocessing <span style="color.red"> DISCLAIMER - The preprocessing of VCTK is not supported as Spraakbanken is the focus of this project</span>
    - Preprocessing of data from the [VCTK Multi-speaker Corpus](https://datashare.is.ed.ac.uk/handle/10283/3443).
* DanishPreprocessing
    - Preprocessing of Danish multi speaker data from the Norwegian National Library [Språkbankens ressurskatalog](https://www.nb.no/sprakbanken/show?serial=oai%3Anb.no%3Asbr-19&lang=en).

### Speech to Text Models Folder
The following models are used for the transformation of original and voice converted speech data to produce a text signal which is used to evalute the speech recognition accuracy when using voice conversion as opposed to no conversion.
* danspeech
    - Implementation of the [danspeech](https://github.com/danspeech) using the Python module [danspeech](https://pypi.org/project/danspeech/)
    - Also include the danspeech_training module for retraining danspeech models. This module is said by danspeech to be on an experimental stage, and as such not always reliable.


### Testing Folder
This module includes code for evaluating and comparing the accuracy of the Speech to Text Models for original and voice converted speech input. 
* McNemar
    - Comparing the final text out put from each speech to text framework when voice input is converted or not using the statistical [McNemar's test](https://en.wikipedia.org/wiki/McNemar%27s_test)
* Word Error Rate
    - Comparing the final text out put from each speech to text framework when voice input is converted or not using the performance metric [Word Error Rate](https://en.wikipedia.org/wiki/Word_error_rate).

### Moving Spraakbanken Files to VCTK Structure
To move files from a Spraakbanken folder structure (nested structure with StasjonXX folders)
use the script preprocess/spraakbanken/files.py. Meta .json files will also be created for each speaker in the process.
```
files.py -data_dir <Spraakbanken Directory Path> -out_dir <Path to create the new file structure>
```

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
* /fagprojekt2020/vc/stargan/convertnew.py  
    - This script is used for converting .wav files directly, meaning preprocess and converting happens together and no training data for StarGAN model training is produces.  
    - The default iteration model used is 200,000 and its path is needed.
    - The mc folder needs to include the _stats.npz file of target speaker, and the same list of training speakers used for training the model need to be included under the class TestDataset as self.speakers.  
    - Every speaker included in the origin_wavpath will be converted.
    - Run in terminal
    ```
    python3 convertnew.py 
    ```

### Usin VAE
As with StarGAN there are three runnable scripts which have been used for running VAE in this project.
* preprocess/spraakbanken/vae/run.sh
    - When running the preprocessing the provided python virtual environment **preprocess_vae** can be used to 
      avoid installing a lot of modules in specific versions.
    - The script preprocess .wav files into a format, that can be used by VAE for training.
      It is important that the script is run on audio data, that is placed in a folder structure
      adhearing to the one created by /preprocess/spraakbanken/files.py
    - Configurations are made in the /preprocess/spraakbanken/vae/preprocess.config file. The most important are:
        - segment_size: How many segments the .wav files should at least contain.
          .wav files with fewer segments will be filtered out.
        - data_dir: The directory where the preprocessed data will be written to
        - raw_data_dir: The directory containing the speaker data to preprocess (must follow structure, see above)
        - training_samples: How many segments to randomly sample from the .wav files to use for training
    - Finally run the script
    ```
    sh run.sh 
    ```
* vc/vae/train.sh
    - Used for training the VAE model. Run it using the **train_vae** as python virtual environment
      to avoid installing a lot of python packages.
    ```
    train.sh -d <location of preprocessed data> -train_set <leave as is> -train_index_file <leave as is> -store_model_path <location    of where to save the trained model and dependencies> -t <name of tensorboard log folder> -iters <training iterations> -summary_step <how often to save a log of the training loss to tensorboard>
    ```

* vc/vae/infer.sh
    - Used to perform conversions of unseen source speakers to a target speaker
        - source_folder: Folder containing source speakers to convert
        - target: location of specific .wav file for the target speaker. This is used for the conversion.
          Better quality = better conversion.
        - inference.py arguments:
            * -a: directory of attr.pkl file, should be in the preprocessed data folder
            * -c: location of model.config.yaml file, should be in the trained model folder
            * -m: location of model.ckpt file, should be in the trained model folder
            * -s: don't change. It is set from source_folder
            * -t: don't change. It is set from target
            * -o: output directory. You have to create the directory.
    ```
    infer.sh
    ```
}


1- target

Voice conversion (VC) is a technique for converting para/nonlinguistic information contained in a given utterance while preserving linguistic information. This technique can be applied to various tasks such as speaker-identity modification for text-to-speech (TTS) systems [1], speaking assistance , speech enhancement , and pronunciation conversion.
One successful VC framework involves statistical methods based on Gaussian mixture models (GMMs). Recently, a neural network (NN)-based framework based on feed-forward deep NNs , recurrent NNs , and generative adversarial nets (GANs) , and an exemplarbased framework based on non-negative matrix factorization (NMF) have also proved successful.
