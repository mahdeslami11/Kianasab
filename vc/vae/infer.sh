#!/bin/bash
source_folder="/work1/s183921/speaker_data/Validation-Corpus/VC_original"
target="/work1/s183921/speaker_data/Spraakbanken-Corpus-Test/r6110032/r6110032_u0032133.wav" #female target speaker
#target="/work1/s183921/speaker_data/Spraakbanken-Corpus-Test/r6110050/r6110050_u0050891.wav" #male target speaker

for source in $(ls $source_folder/*.wav)
    do
        echo "Converting $source to $target..."
        python3 inference.py    -a /work1/s183921/preprocessed_data/vae/spraakbanken/sanity_check/attr.pkl \
                                -c /work1/s183921/trained_models/vae/sanity_check_first/model.config.yaml \
                                -m /work1/s183921/trained_models/vae/sanity_check_first/model.ckpt\
                                -s $source\
                                -t $target\
                                -o /work1/s183921/converted_speakers/vae/sanit_check_final/female_r6110032
        echo "Completed conversion.."
    done
