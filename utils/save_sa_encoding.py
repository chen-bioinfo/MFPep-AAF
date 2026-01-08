import pickle
import os
import sys
sys.path.append("..")
from data_processing import load_seqs
from transformers import EsmTokenizer, EsmForMaskedLM
import torch
from encoding_methods import load_sa

current_path = os.getcwd()
train_fasta_path = os.path.join(current_path, "data/train_data/train.fasta")
train_sa_path = os.path.join(current_path, "data/train_data/sa")
test_fasta_path = os.path.join(current_path, "data/test_data/test.fasta")
test_sa_path = os.path.join(current_path, "data/test_data/sa")

model_path = os.path.join(current_path, "SaProt_650M_AF2")

tokenizer = EsmTokenizer.from_pretrained(model_path)
model = EsmForMaskedLM.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 改一下下面的train_fasta_path和train_sa_path
ids, seqs, labels = load_seqs(test_fasta_path)
res = []
for id in ids:
    """
    parser sa features
    """
    seq = load_sa(id, test_sa_path)
    # print(seq)
    inputs = tokenizer(seq, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    logits = outputs.logits[0][1:-1].cpu().detach().numpy() # 446维
    res.append(logits)

# 将列表保存到文件
with open(os.path.join(test_sa_path, 'sa_encoding.pkl'), 'wb') as f:
    pickle.dump(res, f)
