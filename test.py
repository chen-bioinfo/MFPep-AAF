import torch
import numpy as np
import sys
sys.path.append('..')
import sys
sys.path.append('/geniusland/home/huangjinpeng/MyModel')
import os
from utils.evaluation import evaluate
from models.GAT_plm import GAT_Model
from utils.data_processing import load_data,load_data2
import pickle
from torch_geometric.data import DataLoader
import torch.nn.functional as F
def get_MFBP_model(model_num):
    project_dir = os.path.abspath('..')
    print(project_dir)
    node_feature_dim = 60
    n_class = 5 # 我们是5分类
    extra_hidden_dim=1280
    att_hidden_dim=1024
    modal_dims=[512,1280]
    model = GAT_Model(node_feature_dim, 64, n_class, 0.5, 8,extra_hidden_dim=extra_hidden_dim,att_hidden_dim=att_hidden_dim,modal_dims=modal_dims)
    # MFBP_model_path = f'/geniusland/home/huangjinpeng/MyModel/saved_models/model_MFBP_ESMfine0+GAT60_ResCon_noFGM_{model_num}.pth'
    MFBP_model_path = f'/geniusland/home/huangjinpeng/MyModel/saved_models/model_MFBP_ESMfine0+GAT60_ResCon_noFGM_{model_num}.pth'
    # MFBP_model_path = f'/geniusland/home/huangjinpeng/MyModel/saved_models/model_MFBP_ESMfine0InGAT60_noFGM_{model_num}.pth'
    print(MFBP_model_path)

    if os.path.exists(MFBP_model_path):
        loaded_paras = torch.load(MFBP_model_path)
        model.load_state_dict(loaded_paras)
        print("load model successfully!")
    else:
        print("model not exist!")

    return model



def get_res(data_iter, adj_matrix, model, device, PAD_IDX, res_path, model_num):
    import numpy as np
    model.eval()
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        real_res = []
        pred_res = []
        for id, (x, y) in enumerate(data_iter):
            x, y = x.transpose(0,1).to(device), y.to(device)
            padding_mask = (x != PAD_IDX)
            
            logits, _, _, _ = model(
                input_ids=x,
                attention_mask=padding_mask,
                adj_matrix=adj_matrix)

            y_pred = logits.sigmoid()
            y_pred = y_pred.detach().cpu().numpy()
            label_ids = y.to('cpu').numpy()
            if id == 0:
                pred_res = y_pred
                real_res = label_ids
            else:
                pred_res = np.vstack((y_pred, pred_res))
                real_res = np.vstack((label_ids, real_res))

        aiming, coverage, accuracy, absolute_true, absolute_false = evaluate(pred_res>0.5, real_res)
        print('model_num:', model_num)
        print('aiming:', aiming)
        print('coverage:', coverage)
        print('accuracy:', accuracy)
        print('absolute_true:', absolute_true)
        print('absolute_false:', absolute_false)

    return pred_res, real_res

def get_dataloader():
    threshold=0
    test_path='/geniusland/home/huangjinpeng/MyModel/data/origin_test_data1/'
    # test_path='/geniusland/home/huangjinpeng/MyModel/data/origin_train_data1/'
    if os.path.exists(os.path.join(test_path, f"test_data_list3.pkl")):
        with open(os.path.join(test_path, "test_data_list3.pkl"), "rb") as f:
            test_data_list = pickle.load(f)
    else:
        test_fasta_path = os.path.join(test_path, 'YPVEPF_mutations.fasta')
        test_data_list, _ = load_data(test_fasta_path, threshold, 1) 
        with open(os.path.join(test_path, f"test_data_list3.pkl"), "wb") as f:
            pickle.dump(test_data_list, f)
    test_fasta_path = '/geniusland/home/huangjinpeng/MyModel/data/origin_test_data1/test.fasta'
    # test_fasta_path = '/geniusland/home/huangjinpeng/MyModel/data/origin_train_data1/train.fasta'
    trainESM2_path='/geniusland/home/huangjinpeng/MyModel/ESMsavemodel/feature/MFBP_ESM2_train0.npy'
    testESM2_path='/geniusland/home/huangjinpeng/MyModel/ESMsavemodel/feature/MFBP_ESM2_test0.npy'

    Xs_test,_=load_data2(test_fasta_path, trainESM2_path, testESM2_path)
    test_data_list=list(zip(test_data_list,Xs_test))
    test_dataloader = DataLoader(test_data_list, batch_size=128)

    return test_dataloader

def get_MFBP_res(model_num, test_iter):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = get_MFBP_model(model_num)
    model.to(device)
    graph_info = np.ones((5,5))
    x = np.diag([1 for i in range(5)])
    graph_info = graph_info - x
    adj_matrix = torch.from_numpy(graph_info).float().to(device)
    ALLFeatures={}
    ALLAttention={}
    model.eval()
    with torch.no_grad():
        y_true = []
        y_hat = []
        y_prob=[]
        thresholds = {
        0: 0.5,  
        1: 0.5,  
        2: 0.5,  
        3: 0.5,  
        4: 0.5   
        }
        for data,xs in test_iter:
            data = data.to(device)
            xs=xs.to(device)
            logits,features,attention_weights = model(data.x, data.edge_index,data.batch,xs.x)

            # y_hat.extend(F.sigmoid(logits).cpu().detach().data.numpy()>0.4)
            # y_true.extend(data.y.cpu().detach().data.numpy())
            sequence_features = [features[data.batch == i] for i in range(data.batch.max().item() + 1)]
            ALLFeatures.update({data.seq[i]:sequence_features[i].cpu().numpy() for i in range(len(data.seq))})
            
            sequence_att=[attention_weights[data.batch == i] for i in range(data.batch.max().item() + 1)]
            ALLAttention.update({data.seq[i]:sequence_att[i].cpu().numpy() for i in range(len(data.seq))})

            pred_probs = F.sigmoid(logits).cpu().detach().numpy()
            true_labels = data.y.cpu().detach().numpy()
            
            batch_predictions = []
            for i in range(len(pred_probs)):
                sample_pred = []
                for label_idx in range(5): 

                    threshold = thresholds[label_idx]
                    
                    sample_pred.append(pred_probs[i][label_idx] > threshold)
                
                batch_predictions.append(sample_pred)
        
            
            y_hat.extend(batch_predictions)
            y_true.extend(true_labels)
            y_prob.extend(pred_probs)
        y_hat = np.hstack(y_hat).reshape(-1, 5)
        y_true = np.hstack(y_true).reshape(-1, 5)
        y_prob = np.hstack(y_prob).reshape(-1, 5)

        aiming, coverage, accuracy, absolute_true, absolute_false = evaluate(y_hat>0.5, y_true)
        print('model_num:', model_num)
        print('aiming:', aiming)
        print('coverage:', coverage)
        print('accuracy:', accuracy)
        print('absolute_true:', absolute_true)
        print('absolute_false:', absolute_false)
    return y_hat,y_true,ALLFeatures,y_prob,ALLAttention

if __name__ == '__main__':
    test_iter = get_dataloader()
    
    res = []
    for i in range(5,6):

        pred_res, real_res,ALLFeatures,y_prob,ALLAttention = get_MFBP_res(i, test_iter)
        res.append(pred_res)
        # 保存结果
        np.save(f'./MFBP_ESMfine0+GAT60_ResCon_noFGM_{i}.npy', pred_res)

        np.save(f'MFBP_FusionAttention_test{i}.npy', ALLAttention, allow_pickle=True)
    np.save(f'./MFBP_real_res.npy', real_res)


    stacked_arrays = np.stack(res)
    # 计算沿着第一个维度的平均值
    average_array = np.mean(stacked_arrays, axis=0)
    np.save(f'./MFBP_ESMfine0+GAT60_ResCon_noFGM_average.npy', average_array)


    aiming, coverage, accuracy, absolute_true, absolute_false = evaluate(average_array>0.5, real_res)

    print('aiming:', aiming)
    print('coverage:', coverage)
    print('accuracy:', accuracy)
    print('absolute_true:', absolute_true)
    print('absolute_false:', absolute_false)