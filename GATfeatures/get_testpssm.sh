#! /bin/bash

test=/geniusland/home/huangjinpeng/MyModel/data/MFTP_test

python generate_pssm.py -pssm_ifasta $test/test.fasta -pssm_opssm $test/pssm/

