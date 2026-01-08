import os
import sys

from pathlib import Path
project_root = Path(__file__).resolve().parent.parent  # 上两级目录
sys.path.append(str(project_root))

import utils.psiblast_search as psi
import yaml
import argparse

with open("config.yaml", 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

psiblast = cfg['psiblast']

# Databases and model
nrdb90 = cfg['nrdb90']
nr = cfg['nr']

def generate_pssm(args):
    psi.run(psiblast, args.pssm_ifasta, args.pssm_opssm, nrdb90, nr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate PSSM files using psiblast search')
    # PSSM parameters
    parser.add_argument('-pssm_ifasta', type=str, default='example/test.fasta', help='Input .fasta file for psiblast search')
    parser.add_argument('-pssm_opssm', type=str, default='example/pssm/', help='Output folder saving .pssm files')

    args = parser.parse_args()

    generate_pssm(args)