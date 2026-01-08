import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, LayerNorm
from torch_geometric.nn import TopKPooling
from torch.nn import Linear
from transformers import EsmTokenizer, EsmForMaskedLM, EsmModel
import sys
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import TransformerConv
# from models.GAT_mLabel import GAT
import numpy as np
    
class ModalityAttention(nn.Module):
    def __init__(self, modal_dims, hidden_dim=512):
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(dim, hidden_dim) for dim in modal_dims])
        # self.attns = nn.ModuleList([nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=False) for dim in modal_dims])
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.weight_fc = nn.Linear(hidden_dim, 1)  # 用于计算每个模态的权重分数

    def forward(self, modalities,batch=None):  # list of tensors, each [batch_size, modal_dim]
        # 1. 投影到同一维度
        projected = [proj(mod) for proj, mod in zip(self.projs, modalities)] 
        tokens = torch.stack(projected, dim=1)
        attention_weights=None  # [batch, n_modality, hidden]
        out, attention_weights = self.attn(tokens, tokens, tokens)

        out=out.mean(dim=1)
        output_tensor=torch.cat((out,projected[0],projected[1]),dim=1)
        # return output_tensor,attention_weights # 或用 weighted sum
        return output_tensor,attention_weights
        
class StableModalityAttention(nn.Module):
    def __init__(self, modal_dims, hidden_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.modal_dims = modal_dims
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 1. 稳定的投影层
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for dim in modal_dims
        ])
        
        # 2. 稳定的注意力层
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            batch_first=True,
            dropout=dropout
        )
        
        # 3. 简单的层归一化（可选，用于稳定性）
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """稳定的权重初始化"""
        for proj in self.projs:
            for module in proj:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.1)  # 使用小的正值
        
        # 初始化注意力层
        for name, param in self.attn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)

    def forward(self, modalities, batch=None):
        # 1. 投影到同一维度
        projected = []
        for proj, mod in zip(self.projs, modalities):
            mod_proj = proj(mod)
            # 数值稳定性保护
            mod_proj = torch.clamp(mod_proj, -10, 10)
            projected.append(mod_proj)
        
        # 2. 准备注意力输入 [batch_size, n_modality, hidden_dim]
        tokens = torch.stack(projected, dim=1)
        
        # 3. 注意力计算（带残差连接）
        # 保存原始tokens用于残差连接
        residual = tokens
        
        # 注意力计算
        attn_output, attention_weights = self.attn(
            tokens, tokens, tokens,
            need_weights=True
        )
        
        # 残差连接 + dropout + 层归一化
        attn_output = self.layer_norm(tokens + self.dropout(attn_output))
        
        # 4. 平均注意力输出（保持原始逻辑）
        out = attn_output.mean(dim=1)
        
        # 5. 直接拼接三个特征（原始逻辑）
        # [out, projected[0], projected[1]]
        output_tensor = torch.cat((out, projected[0], projected[1]), dim=1)
        
        # 数值稳定性保护
        # output_tensor = torch.clamp(output_tensor, -10, 10)
        
        return output_tensor, attention_weights
class MultiHeadAttentionBlock(nn.Module):
    """多头自注意力模块"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # self.norm1 = nn.LayerNorm(embed_dim)
        # self.norm2 = nn.LayerNorm(embed_dim)
        # self.ffn = nn.Sequential(
        #     nn.Linear(embed_dim, 4*embed_dim),
        #     nn.GELU(),
        #     nn.Linear(4*embed_dim, embed_dim),
        #     nn.Dropout(dropout))
        
    def forward(self, x, batch):
        if batch is not None:
            mask = (batch.unsqueeze(1) != batch.unsqueeze(0))  # [num_nodes, num_nodes]
        else:
            mask = None
        # 自注意力
        # print(x.shape)
        attn_output, _ = self.attention(query=x,
            key=x,
            value=x,
            attn_mask=mask,
            need_weights=False)
        # x = self.norm1(x + attn_output)
        
        # # 前馈网络
        # ffn_output = self.ffn(x)
        # return self.norm2(x + ffn_output)
        return attn_output
    
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
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
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
        for idx, att in enumerate(self.attentions1):
            x1, attention_matrix = att(x, adj)
            attention_matrix_list.append(attention_matrix)
            if idx == 0:
                x_tmp = x1
            else:
                x_tmp = torch.cat((x_tmp, x1), dim=1)
        x = F.dropout(x_tmp, self.dropout, training=self.training)
        x = F.elu(x)
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
        self.att_hidden_dim = att_hidden_dim
        final_dim=hidden_dim+extra_hidden_dim
        self.useGAT=True
        self.mLabel=False
        self.uesESM=False

        if self.uesESM:
            plm_path='/geniusland/home/huangjinpeng/MyModel/ESM_650M'
            self.tokenizer = EsmTokenizer.from_pretrained(plm_path)
            self.plm_model = EsmModel.from_pretrained(plm_path)
        if self.useGAT:
            self.conv1 = GATConv(node_feature_dim, hidden_dim, heads=nheads)
            self.conv2 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads)
            self.conv3 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads, concat=True)

            self.norm1 = LayerNorm(nheads * hidden_dim)
            self.norm2 = LayerNorm(nheads * hidden_dim)
            self.norm3 = LayerNorm(nheads *hidden_dim)

        self.norm = LayerNorm(1280)
        self.ModalAtt=ModalityAttention(modal_dims,att_hidden_dim)
        # self.ModalAtt=StableModalityAttention(modal_dims,att_hidden_dim)

        if self.mLabel: 
            att_hidden_dim=att_hidden_dim*3      
            self.gat = GAT(nfeat=att_hidden_dim, nhid=384, dropout=0.2, alpha=0.2,nheads=8)
            for i in range(output_dim):    
                setattr(self, "FC%d" %i, nn.Sequential(
                                        nn.Linear(in_features=att_hidden_dim,out_features=att_hidden_dim),
                                        nn.Dropout()))

            for i in range(output_dim):  
                setattr(self, "CLSFC%d" %i, nn.Sequential(
                                        nn.Linear(in_features=att_hidden_dim,out_features=1),
                                        nn.Dropout(),
                                        )) 
        else:
            att_hidden_dim=att_hidden_dim*3
            self.lin0 = Linear(att_hidden_dim, att_hidden_dim)
            self.lin1 = Linear(att_hidden_dim, att_hidden_dim)
            self.lin = Linear(att_hidden_dim, output_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, edge_index, batch, extra_f=None, seqs=None, adj_matrix=None):

        if self.uesESM:
            ESM_features=[]
            for seq in seqs:
                inputs = self.tokenizer(seq, return_tensors="pt").to(x.device)
                outputs = self.plm_model(**inputs)
                hidden_states = outputs.last_hidden_state[0][1:-1]
                ESM_features.append(hidden_states)
            ESM_features=torch.cat(ESM_features, dim=0)
            extra_f[:,-1280:] = ESM_features

        if self.useGAT:
            # x=torch.cat([x, extra_f], dim=-1)
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

            # x=torch.cat([x, extra_f], dim=-1)
        else:
            x=extra_f
        extra_f = self.norm(extra_f)
        modalities=[x,extra_f]
        x,attention_weights=self.ModalAtt(modalities,batch)
      
        features=x
        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=self.drop, training=self.training)

        
        if self.mLabel:
            pred_emb = x
            outs = []
            att_hidden_dim = self.att_hidden_dim*3
            for i in range(self.output_dim):
                FClayer = getattr(self, "FC%d" %i)
                y = FClayer(pred_emb)
                y = torch.squeeze(y, dim=-1)
                outs.append(y)
            
            outs = torch.stack(outs, dim=0).transpose(0, 1) 
            outs = outs.reshape(-1, att_hidden_dim)  
            for i in range(pred_emb.shape[0]):
                if i == 0:
                    end_adj_matrix = adj_matrix.cpu().numpy()
                else:
                    end_adj_matrix = adjConcat(end_adj_matrix, adj_matrix.cpu().numpy())    # [batch_size x num_label, batch_size x num_label]

            end_adj_matrix = torch.tensor(end_adj_matrix).to(outs.device)
            gat_embedding, _ = self.gat(outs, end_adj_matrix)
            gat_embedding = gat_embedding.reshape(-1, self.output_dim, att_hidden_dim)
        
            prediction_scores = list()
            for i in range(self.output_dim):
                CLSFClayer = getattr(self, "CLSFC%d" %i)
                y = CLSFClayer(gat_embedding[:,i,:])
                prediction_scores.append(y)

            x = torch.stack(prediction_scores, dim=1).reshape(-1,self.output_dim)
        else:
            x = self.lin0(x)
            x = F.relu(x)

            x = self.lin1(x)
            x = F.relu(x)

            x = self.lin(x)
        return x,features,attention_weights

