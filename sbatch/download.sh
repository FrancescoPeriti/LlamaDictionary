#!/bin/bash
#SBATCH -A NAISS2024-5-148 -p alvis
#SBATCH -t 00:20:00
#SBATCH -C NOGPU

echo "Download data used in https://aclanthology.org/2023.acl-long.176/"
wget http://www.tkl.iis.u-tokyo.ac.jp/~ishiwatari/naacl_data.zip
mkdir -p data/raw_data
unzip naacl_data.zip -d data/raw_data/
rm naacl_data.zip

mkdir -p data/raw_data/CoDWoE
wget https://codwoe.atilf.fr/data/trial-data_all.zip
unzip trial-data_all.zip -d data/raw_data/CoDWoE/trial-data_all
rm trial-data_all.zip

wget https://codwoe.atilf.fr/data/train-data_all.zip
unzip train-data_all.zip -d data/raw_data/CoDWoE/train-data_all
rm train-data_all.zip

wget https://codwoe.atilf.fr/data/test-data_all.zip
unzip test-data_all.zip -d data/raw_data/CoDWoE/test-data_all
rm test-data_all.zip

wget https://codwoe.atilf.fr/data/full_dataset.zip
unzip full_dataset.zip -d data/raw_data/CoDWoE/
rm full_dataset.zip

## uniform the format of different datasets
python src/format_dataset.py --dataset "codwoe" "oxford" "wordnet" "slang" "wiki"

# remove data dir
rm -rf data

echo "Download data for WiC"
wget https://pilehvar.github.io/wic/package/WiC_dataset.zip
unzip WiC_dataset.zip -d wic
rm WiC_dataset.zip
python src/wic_preprocessing.py -f wic/dev/dev.data.txt # line by line
python src/wic_preprocessing.py -f wic/test/test.data.txt # line by line

echo "Download data for LSC"
wget https://zenodo.org/records/7387261/files/dwug_en.zip?download=1
unzip dwug_en.zip?download=1
rm dwug_en.zip?download=1
python src/lsc_preprocessing.py # line by line
