# MFPep-AAF
📋Amino Acid-level Multimodal Fusion Framework for Multi-functional Peptide Identification by Integrating Protein Language Models and Graph Attention Networks

## 📘 Abstract
&nbsp;&nbsp;&nbsp;&nbsp; Multifunctional peptide identification is a challenging task due to the complex sequence- structure-function relationships. Existing methods often rely on single-modal features, limiting their ability to comprehensively model the intricate interplay between sequence and structural information. In this study, we propose MFPep-AAF, a novel amino acid (AA)-level multimodal fusion framework for multi-functional peptide identification by integrating sequential information and structural features. MFPep-AAF harnesses a cross-modal attention mechanism to dynamically fuse AA-level semantics from a fine-tuned protein language model and AA-level structural constraints from a graph attention network. This fine-grained fusion strategy enables the model to effectively capture both local residue interactions and global sequence-structure relationships for functional prediction. Experimental results on benchmark datasets demonstrate that MFPep-AAF achieves state-of-the-art performance in terms of absolute true metric. These results underscore the advantages of integrating multimodal features, providing a robust and reliable framework for multifunctional peptide prediction. 

## 🧬 Model Structure
&nbsp;&nbsp;&nbsp;&nbsp; MFPep-AAF is designed to identify multifuntional peptides by integrating complementary information from protein sequence and structure at the amino acid level. As illustrated in Figure 1, it comprises two primary phases: multimodal feature extraction and feature fusion. In the feature extraction phase, two distinct modalities of amino acid-level features are extracted. The first modality captures evolutionary sequence information using a fine-tuned ESM2 model. The second modality encodes structural information using GATs, where nodes correspond to amino acids, initialized from a combination of one-hot encoding, HMM encoding, and position encoding. Edges are constructed based on inter-residue contact maps predicted by Rosetta, allowing the GAT to model the structural context of the sequence. Together, these two modalities provide a complementary and fine-grained representation at the amino acid level. In the multimodal fusion phase, a cross-modality attention mechanism is applied to integrate the extracted and GAT features at amino acid-level. The fused feature representation is then concatenated with the original modality-specific features to form a comprehen- sive final sequence representation. This enriched representation effectively leverages both sequence and structural information, subsequently used for multifunctional pep- tide identification. 
<div align=center><img src=img/framework.png></div>

## Requirements
The majoy dependencies used in this project are as following:

```
python  3.7
numpy 1.21.6
tqdm  4.64.1
pyyaml  6.0
scikit-learn  1.0.2
torch  1.11.0+cu113
torch-cluster  1.6.0
torch-scatter  2.0.9
torch-sparse  0.6.15
torch-geometric  1.7.2
tensorflow  1.14.0
tensorboardX  2.5.1
```

More detailed python libraries used in this project are referred to `requirements.txt`. 
Check your CPU device and install the pytorch and pyG (torch-cluster, torch-scatter, torch-sparse, torch-geometric) according to your CUDA version.
> **Note** that torch-geometric 1.7.2 and tensorflow 1.14.0 are required, becuase our trained model does not support the `torch-geometric` with higher version , and the model from trRosetta does not support the `tensorflow` with higher version.
> 
The The installed pyG (torch-cluster, torch-scatter, torch-sparse, torch-geometric) must be a GPU version according to your CUDA. If you installed a wrong vesion, there will be some unexpected errors like https://github.com/rusty1s/pytorch_scatter/issues/248 and https://github.com/pyg-team/pytorch_geometric/issues/2040. We provide the installation process of pytorch and pyG in our environment for reference:

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
```
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric==1.7.2 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
```

## Tools
Two multiple sequence alignment tools and three databases are required: 
```
psi-blast 2.12.0
hhblits 3.3.0
```
Databases:
```
nrdb90(http://bliulab.net/sAMPpred-GAT/static/download/nrdb90.tar.gz)
NR(https://ftp.ncbi.nlm.nih.gov/blast/db/)
uniclust30_2018_08(https://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08_hhsuite.tar.gz)
```
**nrdb90**: We have supplied the nrdb90 databases on our webserver. You need to put it into the `utils/psiblast/` directoy and decompress it. 

**NR**:You can download NR dababase from `https://ftp.ncbi.nlm.nih.gov/blast/db/`. Note that only the files with format `nr.*` are needed. You need to download them can put them into the `utils/psiblast/nr/` directory. The `utils/psiblast/nr/` folder should contain `nr.00.psq`, `nr.00.ppi`, ..., `nr.54.phd`, etc..

**uniclust30_2018_08**:You can download it dababase from `https://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08_hhsuite.tar.gz`. Just decompress it in the directory `utils/hhblits/` and rename this database folder to `uniclust30_2018_08`.

**trRosetta**: The structures are predicted by trRosetta(https://github.com/gjoni/trRosetta), you need to download and put the trRosetta pretrain model(https://files.ipd.uw.edu/pub/trRosetta/model2019_07.tar.bz2) and decompress it into `utils/trRosetta/`.

**ESM2**: The ESM2 model is used to extract sequence-level features. You can download it from `https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt`. Just put it into the `MFPep-AAF` directory.

> **Note** that all the defalut paths of the tools and databases are shown in `GATfeatures/config.yaml`. You can change the paths of the tools and databases by configuring `GATfeatures/config.yaml` as you need. 


`psi-blast` and `hhblist` are recommended to be configured as the system envirenment path. Your can follow these steps to install them:
### How to install psiblast

Download 

```
wget ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.12.0/ncbi-blast-2.12.0+-x64-linux.tar.gz
tar zxvf ncbi-blast-2.12.0+-x64-linux.tar.gz
```

Add the path to system envirenment in `~/.bashrc`.

```
export BLAST_HOME={your_path}/ncbi-blast-2.12.0+
export PATH=$PATH:$BLAST_HOME/bin
```

Finally, reload the system envirenment and check the psiblast command:

```
source ~/.bashrc
psiblast -h
```
## Feature extraction

#GAT features
`GATfeatures/generate_features.py` is the entry of feature extraction process. An usage example is shown in `GATfeatures/generate_features_example.sh`. 

Run the example by: 
```
chmod +x GATfeatures/generate_features_example.sh
./GATfeatures/generate_features_example.sh
```
The features of the examples will be genrerated if your tools and databases are configured correctly. 
Some common errors:
+ `BLAST Database error` means the nrdb90 or NR is failed to found.
+ `ERROR:   could not open file ... uniclust30_2018_08_cs219.ffdata` means the uniclust30_2018_08 is failed to found.

If you want generate the features using your own file in fasta format, just follow the `generate_features_example.sh` and change the pathes into yours.

#ESM2 features
`ESMfeatures/trainESM2.py` is the entry of feature extraction process. An usage example is shown in `ESMfeatures/generate_features_example.sh`. 


## Usage
It takes 3 steps to train/test our model:
(1) copy the train/test soucre files in fasta format, which is  supplied in `datasets` folder, into the `data` folder.
(2) generate features, including the ESM2 features, predicted sturctures and the sequential features.
(3) train / test.

`train.py` and `test.py` are used for training and testing, respectively. 
Running `python train.py -h` and `python test.py -h` to learn the meaning of each parameter.


