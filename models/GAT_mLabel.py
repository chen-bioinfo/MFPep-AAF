from transformers import BertModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv, LayerNorm
from torch_geometric.nn import TopKPooling
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
def adjConcat(a, b):
    """
    Combine the two matrices a,b diagonally along the diagonal direction and fill the empty space with zeros
    """
    lena = len(a)
    lenb = len(b)
    left = np.row_stack((a, np.zeros((lenb, lena))))  
    right = np.row_stack((np.zeros((lena, lenb)), b))  
    result = np.hstack((left, right))
    return result


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime), attention
        else:
            return h_prime, attention

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions1 = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions1):
            self.add_module('attention1_{}'.format(i), attention)
        self.attentions2 = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions2):
            self.add_module('attention2_{}'.format(i), attention)


    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        attention_matrix_list = list()   # Lists for saving attention: len(attention_matrix_list) = layer_num * head
        # First layer gat
        for idx, att in enumerate(self.attentions1):
            x1, attention_matrix = att(x, adj)
            attention_matrix_list.append(attention_matrix)
            if idx == 0:
                x_tmp = x1
            else:
                x_tmp = torch.cat((x_tmp, x1), dim=1)
        x = F.dropout(x_tmp, self.dropout, training=self.training)
        x = F.elu(x)

        # Second layer gat
        for idx, att in enumerate(self.attentions2):
            x2, attention_matrix = att(x, adj)
            attention_matrix_list.append(attention_matrix)
            if idx == 0:
                x_tmp = x2
            else:
                x_tmp = torch.cat((x_tmp, x2), dim=1)
        x = F.dropout(x_tmp, self.dropout, training=self.training)
        x = F.elu(x)

        return x, attention_matrix_list

# class GAT_Model(nn.Module):
#     def __init__(self, node_feature_dim, hidden_dim, output_dim, drop, nheads=8, k=4):
#         super(GAT_Model, self).__init__()

#         self.node_feature_dim = node_feature_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.drop = drop
#         self.heads = nheads
#         self.k = k


#         self.conv1 = GATConv(node_feature_dim, hidden_dim, heads=nheads)
#         self.conv2 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads)
#         self.conv3 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads, concat=False)

#         self.norm1 = LayerNorm(nheads * hidden_dim)
#         self.norm2 = LayerNorm(nheads * hidden_dim)
#         self.norm3 = LayerNorm(hidden_dim)

#         self.topk_pool = TopKPooling(hidden_dim, ratio=k)

#         self._reset_parameters()

#     def _reset_parameters(self):
#         r"""Initiate parameters in the transformer model."""

#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     def forward(self, x, edge_index, batch):
#         x = self.conv1(x, edge_index)
#         x = self.norm1(x, batch)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.drop, training=self.training)

#         x = self.conv2(x, edge_index)
#         x = self.norm2(x, batch)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.drop, training=self.training)

#         x = self.conv3(x, edge_index)
#         x = self.norm3(x, batch)

#         x = self.topk_pool(x, edge_index, batch=batch)[0]
#         # print(x.shape)
#         x = x.view(batch[-1] + 1, -1)
#         # print(x.shape)
#         x = F.dropout(x, p=self.drop, training=self.training)

#         return x

class ModalityAttention(nn.Module):
    def __init__(self, modal_dims, hidden_dim=1024):
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(dim, hidden_dim) for dim in modal_dims])
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.weight_fc = nn.Linear(hidden_dim, 1)  # 用于计算每个模态的权重分数

    def forward(self, modalities):  # list of tensors, each [batch_size, modal_dim]
        # 1. 投影到同一维度
        projected = [proj(mod) for proj, mod in zip(self.projs, modalities)]  # list of [batch, hidden]
        tokens = torch.stack(projected, dim=1)  # [batch, n_modality, hidden]
        out, _ = self.attn(tokens, tokens, tokens)
        return out.mean(dim=1)  # 或用 weighted sum

class GAT_Model(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, drop, nheads=8, k=4,extra_hidden_dim=0,att_hidden_dim=512,modal_dims=[64]):
        super(GAT_Model, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop = drop
        self.heads = nheads
        self.k = k
        self.modal_dims = modal_dims
        final_dim=hidden_dim+extra_hidden_dim

        self.conv1 = GATConv(node_feature_dim, hidden_dim, heads=nheads)
        self.conv2 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads)
        self.conv3 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads, concat=False)

        self.norm1 = LayerNorm(nheads * hidden_dim)
        self.norm2 = LayerNorm(nheads * hidden_dim)
        self.norm3 = LayerNorm(hidden_dim)

        self.ModalAtt=ModalityAttention(modal_dims,att_hidden_dim)
        self.norm4 = LayerNorm(att_hidden_dim)

        self.lin0 = Linear(att_hidden_dim, 768)
        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, edge_index, batch,extra_f=None,seqs=None):
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


        x=torch.cat([x, extra_f], dim=-1)
        # print("拼接后特征",x.shape)
        modalities = []
        start = 0
        for dim in self.modal_dims:
            end = start + dim
            modalities.append(x[:, start:end])  # 切分并加入列表
            start = end
        x=self.ModalAtt(modalities)

        x = global_mean_pool(x, batch)
 
        x = F.dropout(x, p=self.drop, training=self.training)

        x=self.norm4(x)
        # x=self.lin0(x)
        return x  
    
class BertForMultiLabelSequenceClassification(nn.Module):
    def __init__(self,num_labels, node_feature_dim, hidden_dim, output_dim, drop, nheads=8, k=5):
        super(BertForMultiLabelSequenceClassification, self).__init__()
        extra_hidden_dim=4392
        att_hidden_dim=768
        modal_dims=[64,20,20,1280,3072]
        # self.GAT0 = GAT_Model(node_feature_dim, hidden_dim, output_dim, drop, nheads, k) 
        self.GAT0 = GAT_Model(node_feature_dim, hidden_dim, output_dim, drop, nheads,extra_hidden_dim=extra_hidden_dim,att_hidden_dim=att_hidden_dim,modal_dims=modal_dims)

        # self.bert = BertModel.from_pretrained(bert_pretrained_model_dir, output_attentions=True)
        # self.config = config
        self.num_labels = num_labels
        
        self.gat = GAT(nfeat=768, nhid=128, dropout=0.2, alpha=0.2,nheads=6)   # dropout=0.2

        self.pooling = 'pooler'
        for i in range(self.num_labels):    
            setattr(self, "FC%d" %i, nn.Sequential(
                                      nn.Linear(in_features=768,out_features=768),
                                      nn.Dropout()))

        for i in range(self.num_labels):  
            setattr(self, "CLSFC%d" %i, nn.Sequential(
                                      nn.Linear(in_features=768,out_features=1),
                                      nn.Dropout(),
                                      )) 

    def forward(self,
                x,edge_index,batch,
                adj_matrix = None, # [num_labels, num_labels]
                extra_f=None,
                ):
        
        # output = self.bert(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids,
        #     position_ids=position_ids)
        # attention_matrix = output[-1]  # [12, 1, 12, n, n]

        output = self.GAT0(x, edge_index, batch,extra_f)
        # print("-----1-----")
        # print(adj_matrix)
        # if self.pooling == 'cls':
        #     pred_emb = output.last_hidden_state[:, 0]  # [batch, 768]
        # if self.pooling == 'pooler':
        #     pred_emb = output.pooler_output  # [batch, 768]
        # if self.pooling == 'last-avg':
        #     last = output.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
        #     pred_emb = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]        
        pred_emb = output
        outs = []

        for i in range(self.num_labels):
            FClayer = getattr(self, "FC%d" %i)
            y = FClayer(pred_emb)
            y = torch.squeeze(y, dim=-1)
            outs.append(y)
        
        outs = torch.stack(outs, dim=0).transpose(0, 1)  # [batch, num_labels, 768]
        outs = outs.reshape(-1, 768)   # !!!
        for i in range(pred_emb.shape[0]):
            if i == 0:
                end_adj_matrix = adj_matrix.cpu().numpy()
            else:
                end_adj_matrix = adjConcat(end_adj_matrix, adj_matrix.cpu().numpy())    # [batch_size x num_label, batch_size x num_label]

        end_adj_matrix = torch.tensor(end_adj_matrix).to(outs.device)
        gat_embedding, _ = self.gat(outs, end_adj_matrix)
        gat_embedding = gat_embedding.reshape(-1, self.num_labels, 768)
    
        prediction_scores = list()
        for i in range(self.num_labels):
            CLSFClayer = getattr(self, "CLSFC%d" %i)
            y = CLSFClayer(gat_embedding[:,i,:])
            prediction_scores.append(y)

        prediction_res = torch.stack(prediction_scores, dim=1).reshape(-1,self.num_labels)

        # if labels is not None:
        #     loss_fct = nn.BCEWithLogitsLoss()
        #     loss = loss_fct(prediction_res.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        #     return loss, prediction_res, pred_emb, attention_matrix, gat_attention
        # else:
        #     return prediction_res, pred_emb, attention_matrix, gat_attention # [src_len, batch_size, num_labels]
        return prediction_res

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True