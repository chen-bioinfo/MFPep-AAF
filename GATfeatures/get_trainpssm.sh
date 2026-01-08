#! /bin/bash

train=/geniusland/home/huangjinpeng/MyModel/data/MFTP_train

python generate_pssm.py -pssm_ifasta $train/train.fasta -pssm_opssm $train/pssm/

