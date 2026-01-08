#! /bin/bash

test=/geniusland/home/huangjinpeng/MyModel/data/MFTP_test

python generate_hhm_npz.py -hhm_ifasta $test/test.fasta -hhm_oa3m $test/a3m/ -hhm_ohhm $test/hhm/ -hhm_tmp $test/tmp/ -tr_ia3m $test/a3m/ -tr_onpz $test/npz/

