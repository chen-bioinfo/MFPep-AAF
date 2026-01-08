import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from transformers import EsmTokenizer, EsmForMaskedLM, EsmModel

class ESM2(nn.Module):
    def __init__(self, plm_path, hidden_dim, output_dim):
        super(ESM2, self).__init__()
        

        self.tokenizer = EsmTokenizer.from_pretrained(plm_path)
        self.plm_model = EsmModel.from_pretrained(plm_path)
        self.output_dim = output_dim
            
        self.lin0 = Linear(self.plm_model.config.hidden_size, hidden_dim)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)
        
    def forward(self, hidden_states):
        protein_rep = hidden_states.mean(dim=0)  # 全局平均池化

        logits = self.lin0(protein_rep)
        logits = F.relu(logits)
        logits = self.lin1(logits)
        logits = F.relu(logits)
        logits = self.lin2(logits)

        return logits
    
