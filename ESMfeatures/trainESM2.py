import sys

sys.path.append('../')
import torch
from torch.utils.data import Dataset
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
from transformers import EsmTokenizer
from utils.loss import FocalLoss, multilabel_categorical_crossentropy
from models.ESM2 import ESM2
from utils.evaluation import evaluate, check_model4
from torch_geometric.data import Data
from utils.data_processing import load_data4
from torch_geometric.data import DataLoader
import torch.nn as nn
# 自定义数据集类
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=50):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        # self.max_length = max_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sequences[idx],
            # max_length=self.max_length,
            # padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.FloatTensor(self.labels[idx])
        }

# 训练函数
def train_esm(model, train_data, test_data, device, args,tokenizer):
    # 初始化数据加载器
    train_loader = DataLoader(train_data, batch_size=args['batch_size'])
    test_loader = DataLoader(test_data, batch_size=args['batch_size'])
    
    # 定义优化器和损失函数
    optimizer = AdamW([
        {'params': model.plm_model.parameters(), 'lr': 5e-5},
        # {'params': model.lin0.parameters(), 'lr': args['lr']},
        # {'params': model.lin1.parameters(), 'lr': args['lr']},
        # {'params': model.lin2.parameters(), 'lr': args['lr']}
    ])
    
    # criterion = multilabel_categorical_crossentropy
    criterion = nn.BCEWithLogitsLoss()
    best_f1 = 0
    best_aim = 0
    best_coverage = 0
    best_accuracy = 0
    best_absolute_true = 0
    best_absolute_false = 1

    best_epoch = 0
    # 训练循环
    for epoch in range(args['epochs']):
        model.train()
        total_loss = 0
        # 训练阶段
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # inputs = {
            #     'input_ids': batch['input_ids'].to(device),
            #     'attention_mask': batch['attention_mask'].to(device),
            # }
            batch.to(device)
            logits=[]
            for seq in batch.seq:
                inputs = tokenizer(seq, return_tensors="pt").to(device)
                outputs = model.plm_model(**inputs)
                hidden_states = outputs.last_hidden_state[0][1:-1]
                # hidden_states = outputs.last_hidden_state[0]
                # hidden_states = outputs.pooler_output
                # print(hidden_states.shape)
                logit = model(hidden_states)
                logits.append(logit)
                # exit(0)
            logits = torch.cat(logits, dim=0)
            # print(logits.shape,batch.y.shape)
            optimizer.zero_grad()
            # outputs = model.plm_model(**inputs)
            # hidden_states = outputs.last_hidden_state
            # logits = model(hidden_states)  # 修改forward以适应批量处理
            logits = logits.view(-1)
            loss = criterion(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        # 评估阶段
        avg_loss = total_loss / len(train_loader)
        # test_f1, test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        aiming, coverage, accuracy, absolute_true, absolute_false,trainfeatures = check_model4(model, train_loader, device,tokenizer)
        print("train_data")
        print(F"aiming: {aiming}")
        print(F"coverage: {coverage}")
        print(F"accuracy: {accuracy}")
        print(F"absolute_true: {absolute_true}")
        print(F"absolute_false: {absolute_false}")
        aiming, coverage, accuracy, absolute_true, absolute_false,testfeatures = check_model4(model, test_loader, device,tokenizer)
        print("test_data")
        print(F"aiming: {aiming}")
        print(F"coverage: {coverage}")
        print(F"accuracy: {accuracy}")
        print(F"absolute_true: {absolute_true}")
        print(F"absolute_false: {absolute_false}")
        # 保存最佳模型
        if absolute_true > best_absolute_true and absolute_true<0.75: # 记录绝对真最好的时候的模型
            best_absolute_true = absolute_true
            best_aim = aiming
            best_coverage = coverage
            best_accuracy = accuracy
            best_absolute_false = absolute_false
            best_epoch = epoch
            # 保存最好的模型
            torch.save(model.state_dict(), "model/fineESM2_MFBP6.pth")
            np.save('feature/MFBP_ESM2_train6.npy', trainfeatures, allow_pickle=True)
            np.save('feature/MFBP_ESM2_test6.npy', testfeatures, allow_pickle=True)
    print("当absolute_true最高的时候的模型的结果：")
    print(f'best_epoch: {best_epoch + 1}')
    print(f'aim: {best_aim}')
    print(f'coverage: {best_coverage}')
    print(f'accuracy: {best_accuracy}')
    print(f'absolute_true: {best_absolute_true}')
    print(f'absolute_false: {best_absolute_false}')
    print("model/ESM2_MFBP6.pth")
import torch.nn.functional as F 
def test_esm(model, train_dataset, test_dataset, device, train_args, tokenizer):
    model.eval()
    # train_loader = DataLoader(train_data, batch_size=args['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=32)
    with torch.no_grad():
        y_true = []
        y_hat = []
        y_prob=[]
        for batch in tqdm(test_loader, desc="Testing"):
            batch.to(device)
            logits=[]
            real_res = []
            pred_res = []
            for seq in batch.seq:
                inputs = tokenizer(seq, return_tensors="pt").to(device)
                outputs = model.plm_model(**inputs)
                hidden_states = outputs.last_hidden_state[0][1:-1]
                logit = model(hidden_states)
                logits.append(logit)
            logits = torch.cat(logits, dim=0)
            # logits = logits.view(-1)
            # y_pred = logits.sigmoid()
            # y_pred = y_pred.detach().cpu().numpy()
            # label_ids = batch.y.to('cpu').numpy()
            # real_res.append(label_ids)
            # pred_res.append(y_pred)
            # if id == 0:
            #     pred_res = y_pred
            #     real_res = label_ids
            # else:
            #     pred_res = np.vstack((y_pred, pred_res))
            #     real_res = np.vstack((label_ids, real_res))
            y_hat.extend(F.sigmoid(logits).cpu().detach().data.numpy()>0.5)
            y_true.extend(batch.y.cpu().detach().data.numpy())

        y_hat = np.hstack(y_hat).reshape(-1, 5)
        y_true = np.hstack(y_true).reshape(-1, 5)
        # print(y_hat.shape, y_true.shape)
        # exit(0)
        aiming, coverage, accuracy, absolute_true, absolute_false,testfeatures,y_prob = check_model4(model, test_loader, device,tokenizer)
        print("test_data")
        print(F"aiming: {aiming}")
        print(F"coverage: {coverage}")
        print(F"accuracy: {accuracy}")
        print(F"absolute_true: {absolute_true}")
        print(F"absolute_false: {absolute_false}")
        # np.save('feature/YPVEPF_ESMfine0_noFGM.npy', testfeatures, allow_pickle=True)
    return y_hat, y_true,y_prob
# 使用示例
if __name__ == "__main__":
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型和分词器
    # model = ESM2(plm_path="ESM_650M", 
    #             hidden_dim=512, 
    #             output_dim=5).to(device)
    esm_path='/geniusland/home/huangjinpeng/MyModel/ESM_650M'
    model = ESM2(plm_path=esm_path,hidden_dim=1024, 
                output_dim=5).to(device)
    model.load_state_dict(torch.load("/geniusland/home/huangjinpeng/MyModel/ESMsavemodel/model/fineESM2_MFBP4.pth"))
    # model.load_state_dict(torch.load("/geniusland/home/huangjinpeng/MyModel/ESMsavemodel/model/fineESM2_MFTP8.pth"))
    print(model)
    for param in model.plm_model.parameters():
        param.requires_grad = True
    # 创建数据集
    train_fasta_path = '/geniusland/home/huangjinpeng/MyModel/data/origin_train_data1/train.fasta'
    test_fasta_path = '/geniusland/home/huangjinpeng/MyModel/data/origin_test_data1/test.fasta'
    # train_fasta_path = '/geniusland/home/huangjinpeng/MyModel/data/MFTP_train_data/train.fasta'
    # test_fasta_path = '/geniusland/home/huangjinpeng/MyModel/data/MFTP_test_data/test.fasta'
    # test_fasta_path = '/geniusland/home/huangjinpeng/MyModel/data/YPVEPFmutations/YPVEPF_mutations.fasta'
    # test_fasta_path = '/geniusland/home/huangjinpeng/MyModel/data/AVPDVAFNAYGmutations/AVPDVAFNAYG_mutations.fasta'
    test_fasta_path = '/geniusland/home/huangjinpeng/MyModel/data/mutations/GSSSGRGDSPA_mutations.fasta'
    # train_sequence_data, train_sequence_label = getSequenceData(first_dir, 'train')
    # test_sequence_data, test_sequence_label = getSequenceData(first_dir, 'test')
    tokenizer = EsmTokenizer.from_pretrained(esm_path)
    # train_dataset = ProteinDataset(train_sequence_data, train_sequence_label, tokenizer)
    # test_dataset = ProteinDataset(test_sequence_data, test_sequence_label, tokenizer)
    train_dataset,_=load_data4(train_fasta_path, 0, 1)
    print(len(train_dataset))
    test_dataset,_=load_data4(test_fasta_path, 0, 1)
    print(len(test_dataset))
    # 训练参数配置
    train_args = {
        'epochs': 100,
        'batch_size': 64,
        'lr': 1e-3,       # 顶层分类层学习率
        'max_grad_norm': 1.0
    }
    
    # 开始训练
    # train_esm(model, train_dataset, test_dataset, device, train_args,tokenizer)
    real_res, pred_res,y_prob=test_esm(model, train_dataset, test_dataset, device, train_args, tokenizer)
    # np.save(f'./YPVEPF_ESMfine0_noFGM.npy', y_prob)
    # np.save(f'./YPVEPF_real_res.npy', real_res)
    np.save(f'./mutationProb/GSSSGRGDSPA_pred_res.npy', y_prob)
    