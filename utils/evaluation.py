"""https://github.com/xialab-ahu/MLBP/blob/master/MLBP/evaluation.py"""
from sklearn.metrics import confusion_matrix,roc_auc_score,matthews_corrcoef,roc_curve,auc
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,precision_recall_curve
import torch
import torch.nn.functional as F
import numpy as np

def scores(y_test,y_pred,th=0.5):
    y_predlabel=[(0 if item<th else 1) for item in y_pred]
    tn,fp,fn,tp=confusion_matrix(y_test,y_predlabel).flatten()
    SPE=tn*1./(tn+fp)
    MCC=matthews_corrcoef(y_test,y_predlabel)
    Recall=recall_score(y_test, y_predlabel)
    Precision=precision_score(y_test, y_predlabel)
    F1=f1_score(y_test, y_predlabel)
    Acc=accuracy_score(y_test, y_predlabel)
    AUC=roc_auc_score(y_test, y_pred)
    precision_aupr, recall_aupr, _ = precision_recall_curve(y_test, y_pred)
    AUPR = auc(recall_aupr, precision_aupr)
    return [Recall,SPE,Precision,F1,MCC,Acc,AUC,AUPR,tp,fn,tn,fp]



def Aiming(y_hat, y):
    '''
    the “Aiming” rate (also called “Precision”) is to reflect the average ratio of the
    correctly predicted labels over the predicted labels; to measure the percentage
    of the predicted labels that hit the target of the real labels.
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / sum(y_hat[v])
    return sorce_k / n


def Coverage(y_hat, y):
    '''
    The “Coverage” rate (also called “Recall”) is to reflect the average ratio of the
    correctly predicted labels over the real labels; to measure the percentage of the
    real labels that are covered by the hits of prediction.
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / sum(y[v])

    return sorce_k / n


def Accuracy(y_hat, y):
    '''
    The “Accuracy” rate is to reflect the average ratio of correctly predicted labels
    over the total labels including correctly and incorrectly predicted labels as well
    as those real labels but are missed in the prediction
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / union
    return sorce_k / n


def AbsoluteTrue(y_hat, y):
    '''
    same
    '''

    n, m = y_hat.shape
    sorce_k = 0
    for v in range(n):
        if list(y_hat[v]) == list(y[v]):
            sorce_k += 1
    return sorce_k/n


def AbsoluteFalse(y_hat, y):
    '''
    hamming loss
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v,h] == 1 or y[v,h] == 1:
                union += 1
            if y_hat[v,h] == 1 and y[v,h] == 1:
                intersection += 1
        sorce_k += (union-intersection)/m
    return sorce_k/n


def evaluate(y_hat, y):
    aiming = Aiming(y_hat, y)
    coverage = Coverage(y_hat, y)
    accuracy = Accuracy(y_hat, y)
    absolute_true = AbsoluteTrue(y_hat, y)
    absolute_false = AbsoluteFalse(y_hat, y)
    return aiming, coverage, accuracy, absolute_true, absolute_false

def f1_max(pred, target):
    """
    F1 score with the optimal threshold.

    This function first enumerates all possible thresholds for deciding positive and negative
    samples, and then pick the threshold with the maximal F1 score.

    Parameters:
        pred (Tensor): predictions of shape :math:`(B, N)`
        target (Tensor): binary targets of shape :math:`(B, N)`
    """
    # 按预测值降序排序
    order = pred.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    
    # 计算累计精确率和召回率
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    
    # 初始化is_start，用于标记每行的第一个元素
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)

    # 展平所有元素并重新排序
    all_order = pred.flatten().argsort(descending=True)
    order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    
    # 计算累计精确率和召回率的差值
    all_precision = precision[all_order] - \
                    torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - \
                 torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    
    # 计算F1分数
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return all_f1.max()

def check_model(model, data_loader, device):
    model.eval()
    ALLFeatures = {}
    with torch.no_grad():
        y_true = []
        y_hat = []
        for data in data_loader:
            data = data.to(device)
            logits,features = model(data.x, data.edge_index,data.batch)
            # print(features.shape)
            sequence_features = [features[data.batch == i] for i in range(data.batch.max().item() + 1)]
            # print(len(sequence_features),len(sequence_features[0]))
            ALLFeatures.update({data.seq[i]:sequence_features[i].cpu().numpy() for i in range(len(data.seq))})

            y_hat.extend(F.sigmoid(logits).cpu().detach().data.numpy()>0.5)
            y_true.extend(data.y.cpu().detach().data.numpy())
        y_hat = np.hstack(y_hat).reshape(-1, 5)
        y_true = np.hstack(y_true).reshape(-1, 5)
        aiming, coverage, accuracy, absolute_true, absolute_false = evaluate(y_hat, y_true)
    
    return aiming, coverage, accuracy, absolute_true, absolute_false,ALLFeatures

def check_model2(model, data_loader, device,n_class=5,adj_matrix=None):
    model.eval()
    ALLy_hat = {}
    ALLFeatures = {}
    ALLAttention = {}
    with torch.no_grad():
        y_true = []
        y_hat = []
        for data,xs in data_loader:
            data = data.to(device)
            xs=xs.to(device)
            # seq = seq.to(device)
            logits,features,attention_weights = model(data.x, data.edge_index,data.batch,xs.x,data.seq,adj_matrix)
            # logits = model(xs)
            y_hat.extend(F.sigmoid(logits).cpu().detach().data.numpy()>0.5)
            y_true.extend(data.y.cpu().detach().data.numpy())
            batch_seqs = data.seq
            for i, seq in enumerate(batch_seqs):
                # 保存当前序列的预测结果
                ALLy_hat[seq] = y_hat[i]

            sequence_features = [features[data.batch == i] for i in range(data.batch.max().item() + 1)]
            ALLFeatures.update({data.seq[i]:sequence_features[i].cpu().numpy() for i in range(len(data.seq))})
            
            sequence_att=[attention_weights[data.batch == i] for i in range(data.batch.max().item() + 1)]
            ALLAttention.update({data.seq[i]:sequence_att[i].cpu().numpy() for i in range(len(data.seq))})
            
        y_hat = np.hstack(y_hat).reshape(-1,n_class)
        y_true = np.hstack(y_true).reshape(-1, n_class)
        aiming, coverage, accuracy, absolute_true, absolute_false = evaluate(y_hat, y_true)
        ALLy_hat.update({data.seq[i]:y_hat[i] for i in range(len(data.seq))})
        print(len(ALLAttention))
        if 'AATENM' in ALLAttention.keys():
            print(ALLAttention['AATENM'])
        # exit(0)
    return aiming, coverage, accuracy, absolute_true, absolute_false,ALLy_hat,ALLAttention,ALLFeatures

def check_model3(model, data_loader, device,n_class=5,adj_matrix=None):
    model.eval()
    with torch.no_grad():
        y_true = []
        y_hat = []
        for data,xs,seq in data_loader:
            data = data.to(device)
            xs=xs.to(device)
            seq = seq.to(device)
            logits = model(data.x, data.edge_index,data.batch,xs.x,seq,adj_matrix)
            # logits = model(xs)
            y_hat.extend(F.sigmoid(logits).cpu().detach().data.numpy()>0.5)
            y_true.extend(data.y.cpu().detach().data.numpy())

        y_hat = np.hstack(y_hat).reshape(-1,n_class)
        y_true = np.hstack(y_true).reshape(-1, n_class)
        aiming, coverage, accuracy, absolute_true, absolute_false = evaluate(y_hat, y_true)
    
    return aiming, coverage, accuracy, absolute_true, absolute_false
    
def check_model4(model, data_loader, device,tokenizer):
    model.eval()
    features={}
    with torch.no_grad():
        y_true = []
        y_hat = []
        y_prob=[]
        for batch in data_loader:
            logits = []
            batch.to(device)
            for i,seq in enumerate(batch.seq):
                inputs = tokenizer(seq, return_tensors="pt").to(device)
                outputs = model.plm_model(**inputs)
                hidden_states = outputs.last_hidden_state[0][1:-1]
                # hidden_states = outputs.pooler_output
                # print(hidden_states.shape)
                # print(batch.id[i])
                features[seq]=hidden_states.cpu().numpy()
                logit = model(hidden_states)
                # print(logit.shape)
                logits.append(logit)
            logits = torch.cat(logits, dim=0)
            # print(logits.shape)
            pred_probs = F.sigmoid(logits).cpu().detach().numpy()
            # print(F.sigmoid(logits).cpu().detach().numpy())
            y_hat.extend(F.sigmoid(logits).cpu().detach().data.numpy()>0.5)
            y_true.extend(batch.y.to(device).cpu().detach().data.numpy())
            y_prob.extend(pred_probs)

        y_hat = np.hstack(y_hat).reshape(-1, 5)
        y_true = np.hstack(y_true).reshape(-1, 5)
        y_prob = np.hstack(y_prob).reshape(-1, 5)
        print(y_prob[:,4])
        print(y_prob[:,0])
        aiming, coverage, accuracy, absolute_true, absolute_false = evaluate(y_hat, y_true)
    
    return aiming, coverage, accuracy, absolute_true, absolute_false,features,y_prob
def get_test_result(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        y_true = []
        y_hat = []
        for data in data_loader:
            data = data.to(device)

            logits = model(data.x, data.edge_index, data.batch)
            
            y_hat.extend(F.sigmoid(logits).cpu().detach().data.numpy())
            y_true.extend(data.y.cpu().detach().data.numpy())
        
        y_hat = np.hstack(y_hat).reshape(-1, 5)
        y_true = np.hstack(y_true).reshape(-1, 5)
    
    return y_hat, y_true