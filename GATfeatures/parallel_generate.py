# for i in range(11):
#     with open(f'get_trainpssm_{i}.sh', 'w') as file:
#         file.write(f'#! /bin/bash\n\ntrain=data/train_data_parallel\n\npython getpssm.py -pssm_ifasta $train/train_{i}.fasta -pssm_opssm $train/pssm/')

for i in range(11):
    with open(f'get_trainfeatures_{i}.sh', 'w') as file:
        file.write(f'#! /bin/bash\n\ntrain=data/train_data\n\npython generate_features.py -hhm_ifasta $train/train_{i}.fasta -hhm_oa3m $train/a3m_{i}/ -hhm_ohhm $train/hhm/ -hhm_tmp $train/tmp/ -tr_ia3m $train/a3m_{i}/ -tr_onpz $train/npz/')