#! /bin/bash

train=/geniusland/home/huangjinpeng/MyModel/data/MFTP_train

python generate_hhm_npz.py -hhm_ifasta $train/train.fasta -hhm_oa3m $train/a3m/ -hhm_ohhm $train/hhm/ -hhm_tmp $train/tmp/ -tr_ia3m $train/a3m/ -tr_onpz $train/npz/

