import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, LayerNorm
from torch_geometric.nn import TopKPooling
import torch.nn as nn
from torch.nn import Linear
from transformers import EsmTokenizer, EsmForMaskedLM
from utils.encoding_methods import load_sa

"""
Appling pyG lib
"""


# class GATModel(nn.Module):
#     def __init__(self, node_feature_dim, hidden_dim, output_dim, drop, nheads=1, device='cpu', k=4):
#         super(GATModel, self).__init__()

#         # 加载蛋白质语言模型
#         # self.tokenizer = EsmTokenizer.from_pretrained(model_path)
#         # self.plm_model = EsmForMaskedLM.from_pretrained(model_path)

#         self.node_feature_dim = node_feature_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.drop = drop
#         self.heads = nheads
#         self.device = device
#         self.k = k

#         # self.embedding = nn.Embedding(20, 20) # 将独热编码的节点特征映射到embedding空间
#         # self.fc = nn.Linear(446, 20) # 446是SaProt的特征维度

#         self.conv1 = GATConv(node_feature_dim, hidden_dim, heads=nheads)
#         # self.conv2 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads)
#         # self.conv3 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads, concat=False)
#         self.conv2 = GATConv((nheads * hidden_dim) + node_feature_dim, hidden_dim, heads=nheads)
#         self.conv3 = GATConv((nheads * hidden_dim) + (nheads * hidden_dim) + node_feature_dim, hidden_dim, heads=nheads, concat=False)

#         # self.norm0 = LayerNorm(nheads * hidden_dim)
#         self.norm1 = LayerNorm(nheads * hidden_dim)
#         self.norm2 = LayerNorm(nheads * hidden_dim)
#         self.norm3 = LayerNorm(hidden_dim)

#         self.lin0 = Linear(k * hidden_dim, hidden_dim)
#         self.lin1 = Linear(hidden_dim, hidden_dim)
#         self.lin = Linear(hidden_dim, output_dim)

#         self.topk_pool = TopKPooling(hidden_dim, ratio=k)

#         self.residual1 = Linear(node_feature_dim, nheads * hidden_dim)
#         self.residual2 = Linear(nheads * hidden_dim, nheads * hidden_dim)
#         self.residual3 = Linear(nheads * hidden_dim, hidden_dim)

#         self.residual1 = Linear(node_feature_dim, nheads * hidden_dim)
#         self.residual2 = Linear(nheads * hidden_dim, nheads * hidden_dim)
#         self.residual3 = Linear(nheads * hidden_dim, hidden_dim)

#         self._reset_parameters()

#     def _reset_parameters(self):
#         r"""Initiate parameters in the transformer model."""

#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     def forward(self, x, ids, sa_dir, edge_index, batch):
#         # print(type(ids)) # list
#         # print(ids.shape)

#         # 先使用embedding将独热编码的节点特征映射到embedding空间
#         # one_hot_encoding = x[:, :20]
#         # indices = torch.argmax(one_hot_encoding, dim=1)
#         # embedding = self.embedding(indices)
#         # x = torch.cat([embedding, x[:, 20:]], dim=1)

#         # 将SaProt的特征转化为20维
#         # plm_encoding = []
#         # for id in ids:
#         #     seq = load_sa(id, sa_dir)
#         #     inputs = self.tokenizer(seq, return_tensors="pt")
#         #     inputs.to(self.device)

#         #     outputs = self.plm_model(**inputs)
#         #     logits = outputs.logits[0][1:-1]
#         #     # print(logits.shape)
#         #     plm_encoding.append(logits)
#         # plm_encoding = torch.cat(plm_encoding, dim=0)
#         # plm_encoding = self.fc(plm_encoding)
#         # x = torch.cat([x, plm_encoding], dim=1)
#         # x = plm_encoding

#         x1 = self.conv1(x, edge_index)
#         x1 = self.norm1(x1, batch)
#         x1 = F.relu(x1)
#         x1 = F.dropout(x1, p=self.drop, training=self.training)

#         x1_res = torch.cat([x, x1], dim=-1)
        
#         x2 = self.conv2(x1_res, edge_index)
#         x2 = self.norm2(x2, batch)
#         x2 = F.relu(x2)
#         x2 = F.dropout(x2, p=self.drop, training=self.training)

#         x2_res = torch.cat([x1_res, x2], dim=-1)

#         x3 = self.conv3(x2_res, edge_index)
#         x3 = self.norm3(x3, batch)
#         x3 = F.relu(x3)

#         x = self.topk_pool(x3, edge_index, batch=batch)[0]
#         x = x.view(batch[-1] + 1, -1) # 这个地方batch从512变成511了

#         x = F.dropout(x, p=self.drop, training=self.training)

#         x = self.lin0(x)
#         x = F.relu(x)

#         x = self.lin1(x)
#         x = F.relu(x)

#         z = x  # extract last layer features

#         x = self.lin(x)
#         y = x
#         x = F.sigmoid(x)

#         return x, y, z



