from transformers import BertModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv, LayerNorm
from torch_geometric.nn import TopKPooling
from torch.nn import Linear
from transformers import EsmTokenizer, EsmForMaskedLM, EsmModel
from torch_geometric.nn import global_mean_pool

class Attention(nn.Module):
    def __init__(self, input_dim, dense_dim, n_heads):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        self.n_heads = n_heads
        self.fc1 = nn.Linear(self.input_dim, self.dense_dim)
        self.fc2 = nn.Linear(self.dense_dim, self.n_heads)

    def forward(self, input):  # input.shape = (batch_size, seq_len, input_dim)
        x = torch.tanh(self.fc1(input))  # x.shape = (batch_size, seq_len, dense_dim)
        x = self.fc2(x)  # x.shape = (batch_size, seq_len, n_heads)
        x = torch.softmax(x, dim=1)  # 在第1维度上进行softmax操作
        attention = x.transpose(1, 2)  # attention.shape = (batch_size, n_heads, seq_len)
        return attention
    
class GAT_Model(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, drop, nheads=8, k=4):
        super(GAT_Model, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop = drop
        self.heads = nheads
        self.k = k


        self.conv1 = GATConv(node_feature_dim, hidden_dim, heads=nheads)
        self.conv2 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads)
        self.conv3 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads, concat=False)

        self.norm1 = LayerNorm(nheads * hidden_dim)
        self.norm2 = LayerNorm(nheads * hidden_dim)
        self.norm3 = LayerNorm(hidden_dim)

        self.lin0 = Linear(hidden_dim, hidden_dim)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin = Linear(hidden_dim, output_dim)

        self.topk_pool = TopKPooling(hidden_dim, ratio=k)

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, edge_index, batch,extra_f=None):
        x = self.conv1(x, edge_index)
        x = self.norm1(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.norm2(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.norm3(x, batch)

        features = x

        # x = self.topk_pool(x, edge_index, batch=batch)[0]
        # x = x.view(batch[-1] + 1, -1)
        
        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=self.drop, training=self.training)

        x = self.lin0(x)
        x = F.relu(x)

        x = self.lin1(x)
        x = F.relu(x)

        x = self.lin(x)

        return x,features
    

class GATModel(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, drop, nheads=1, k=4):
        super(GATModel, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop = drop
        self.heads = nheads
        self.k = k

        self.conv1 = GATConv(node_feature_dim, hidden_dim, heads=nheads)
        self.conv2 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads)
        self.conv3 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads, concat=False)

        self.norm1 = LayerNorm(nheads * hidden_dim)
        self.norm2 = LayerNorm(nheads * hidden_dim)
        self.norm3 = LayerNorm(hidden_dim)

        # self.lin0 = Linear(k * hidden_dim, hidden_dim)
        # self.lin1 = Linear(hidden_dim, hidden_dim)
        # self.lin = Linear(hidden_dim, output_dim)

        self.topk_pool = TopKPooling(hidden_dim, ratio=k)

        self.attention = Attention(hidden_dim, hidden_dim, nheads * 8)

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.norm1(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.norm2(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.norm3(x, batch)

        # print(x.shape) [num_nodes, hidden_dim]
        # print(batch.shape) [num_nodes]
        pooled_x, _, _, _, _, _ = self.topk_pool(x, edge_index, batch=batch)
        pooled_x = pooled_x.view(batch[-1] + 1, -1)

        node_feature_embedding_list = []

        for i in range(batch[-1] + 1):
            sample_x = x[batch == i]
            sample_x_unsqueezed = sample_x.unsqueeze(0)
            attention = self.attention(sample_x_unsqueezed)
            node_feature_embedding = attention.squeeze(0) @ sample_x_unsqueezed.squeeze(0)
            node_feature_embedding_avg = torch.sum(node_feature_embedding, dim=1) / self.attention.n_heads
            node_feature_embedding_list.append(node_feature_embedding_avg.squeeze(0))
        
        node_feature_embedding = torch.stack(node_feature_embedding_list)

        # print(pooled_x.shape)
        # print(node_feature_embedding.shape)
        concat_output = torch.cat((pooled_x, node_feature_embedding), dim=1)

        # x = F.dropout(x, p=self.drop, training=self.training)

        # x = self.lin0(x)
        # x = F.relu(x)

        # x = self.lin1(x)
        # x = F.relu(x)

        # x = self.lin(x)

        return concat_output

class GAT_plm(nn.Module):
    def __init__(self, use_plm, use_esm, plm_path, node_feature_dim, hidden_dim, output_dim, drop, nheads=1, k=4):
        super(GAT_plm, self).__init__()
        
        self.use_plm = use_plm
        self.use_esm = use_esm
        if use_plm:
            self.tokenizer = EsmTokenizer.from_pretrained(plm_path)
            self.plm_model = EsmModel.from_pretrained(plm_path)
            node_feature_dim += self.plm_model.config.hidden_size
        
        self.GAT = GATModel(node_feature_dim, hidden_dim, output_dim, drop, nheads, k=k)

        if use_plm:
            self.lin0 = Linear(k * hidden_dim + nheads * 8 + self.plm_model.config.hidden_size, hidden_dim)
        else:
            self.lin0 = Linear(k * hidden_dim + nheads * 8, hidden_dim)
        
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)
        
    def forward(self, x, sa_seq, sequences, edge_index, batch):
        if self.use_plm:
            plm_encoding = []
            plm_logits = []

            if self.use_esm:
                seqs = sequences
            else:
                seqs = sa_seq
            for seq in seqs:
                inputs = self.tokenizer(seq, return_tensors="pt").to(x.device)
                
                outputs = self.plm_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1][0][1:-1]
                protein_representation = hidden_states.mean(dim=0)
                plm_encoding.append(hidden_states)
                plm_logits.append(protein_representation.unsqueeze(0))

            plm_encoding = torch.cat(plm_encoding, dim=0)
            plm_logits = torch.cat(plm_logits, dim=0)
            x = torch.cat([x, plm_encoding], dim=1)

        logits = self.GAT(x, edge_index, batch)

        if self.use_plm:
            logits = torch.cat([logits, plm_logits], dim=1)

        logits = self.lin0(logits)
        logits = F.relu(logits)
        
        logits = self.lin1(logits)
        logits = F.relu(logits)

        logits = self.lin2(logits)

        return logits