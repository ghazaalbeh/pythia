#!/bin/bash
ln -s /home/behboud/projects/def-agullive/behboud/myData/data/ data

mkdir -p pythia/.vector_cache
cd pythia/.vector_cache
wget https://dl.fbaipublicfiles.com/pythia/pretrained_models/fasttext/wiki.en.bin 
wget http://nlp.stanford.edu/data/glove.6B.zip
