#!/bin/bash
source_folder="/work1/s183921/speaker_data/source_speakers/william_test"
target="/work1/s183921/speaker_data/Spraakbanken-Corpus-Test/r6110050/r6110050_u0050891.wav"

for source in $(ls $source_folder/*.wav)
    do
        echo "Converting $source to $target..."
        python3 inference.py    -a /work1/s183921/preprocessed_data/vae/spraakbanken/preprocessed_danish_10_test/attr.pkl \
                                -c /work1/s183921/trained_models/vae/model_danish_10_test/model.config.yaml \
                                -m /work1/s183921/trained_models/vae/model_danish_10_test/model.ckpt\
                                -s $source\
                                -t /work1/s183921/speaker_data/target_speakers/william_test/$target\
                                -o /work1/s183921/converted_speakers/vae/$source_to_$target.wav
        echo "Completed conversion.."
    done
