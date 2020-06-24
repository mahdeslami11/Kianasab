#!/bin/bash
#Main shell script for running speaker conversions with VAE
source_folder="/work1/s183921/speaker_data/Validation-Corpus/VC_original"
target="/work1/s183921/speaker_data/Spraakbanken-Corpus-Test/r6110032/r6110032_u0032133.wav"

for source in $(ls $source_folder/*.wav)
    do
        echo "Converting $source to $target..."
        python3 inference.py    -a /work1/s183921/preprocessed_data/vae/spraakbanken/sanity_check/attr.pkl \
                                -c /work1/s183921/trained_models/vae/sanity_check/model.config.yaml \
                                -m /work1/s183921/trained_models/vae/sanity_check/model.ckpt\
                                -s $source\
                                -t $target\
                                -o /work1/s183921/converted_speakers/vae/sanity_check/female_r6110032
        mv /work1/s183921/emb.npy /work1/s183921/$source_emb.npy
        mv /work1/s183921/mu.npy /work1/s183921/$source_mu.npy
        mv /work1/s183921/sigma.npy /work1/s183921/$source_sigma.npy
        echo "Completed conversion.."
    done
