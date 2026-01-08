import os
import numpy as np
import torch
from torch_geometric.data import DataLoader
import argparse
from models.GAT_plm import GAT_Model
from utils.data_processing import load_data,load_data2
from utils.evaluation import check_model2
from utils.log_helper import logger_init
from utils.loss import FocalLoss, multilabel_categorical_crossentropy
import logging
import datetime
import warnings
import pickle
from tqdm import tqdm
warnings.filterwarnings("ignore")

import os
current_path = os.getcwd()
esm_650M_path = os.path.join(current_path, 'ESM_650M')
logs_path = os.path.join(current_path, 'logs')
saved_model_path = os.path.join(current_path, 'saved_models')
saved_png_path = os.path.join(current_path, 'png')

def train(args):
    # 日志初始化
    model_num = args.model_num
    logger_init(log_file_name=f"{model_num}", log_level=logging.INFO, log_dir=logs_path)
    threshold = args.d

    # 加载训练集
    logging.info('Loading training data...')
    if os.path.exists(os.path.join(args.GATnodeTrain_path)):
        with open(os.path.join(args.GATnodeTrain_path), "rb") as f:
            train_data_list = pickle.load(f)
    else:
        train_fasta_path = args.train_fasta_path
        train_data_list, _ = load_data(train_fasta_path, threshold, 1)
        with open(os.path.join(args.GATnodeTrain_path), "wb") as f:
            pickle.dump(train_data_list, f)
    logging.info(f'train seqs number: {len(train_data_list)}')
    
    # 加载测试集
    logging.info('Loading testing data...')
    if os.path.exists(os.path.join(args.GATnodeTest_path)):
        with open(os.path.join(args.GATnodeTest_path), "rb") as f:
            test_data_list = pickle.load(f)
    else:
        test_fasta_path = args.test_fasta_path
        test_data_list, _ = load_data(test_fasta_path, threshold, 1) 
        with open(os.path.join(args.GATnodeTest_path), "wb") as f:
            pickle.dump(test_data_list, f)
    logging.info(f'test seqs number: {len(test_data_list)}')

    print(train_data_list[0].x.shape,test_data_list[0].x.shape)
    train_fasta_path = args.train_fasta_path
    Xs_train,_=load_data2(train_fasta_path, args.trainESM2_path, args.testESM2_path)
    test_fasta_path = args.test_fasta_path
    Xs_test,_=load_data2(test_fasta_path, args.trainESM2_path, args.testESM2_path)

    

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f'device: {str(device)}')

    node_feature_dim = args.node_feature_dim
    n_class = args.n_class # 我们是5分类
    graph_info = np.ones((n_class,n_class))
    x = np.diag([1 for i in range(n_class)])
    graph_info = graph_info - x
    adj_matrix = torch.from_numpy(graph_info).float().to(device)


    extra_hidden_dim=Xs_train[0].x.shape[-1]
    logging.info(f"extra_hidden_dim: {extra_hidden_dim}")
    att_hidden_dim=args.att_dim
    # ESM 1280,pos 20,GAT 64
    modal_dims=args.modal_dim
    logging.info(f"modal_dims: {modal_dims}")
    model = GAT_Model(node_feature_dim, args.hd, n_class, args.drop, args.heads,extra_hidden_dim=extra_hidden_dim,att_hidden_dim=att_hidden_dim,modal_dims=modal_dims)
    model.to(device)
    logging.info(args)
    logging.info(f"{model}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 调整学习率，每隔 5 个 epoch，学习率会乘以 0.9
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9) 
    # 损失函数
    if args.loss_function == 'BCE':
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.loss_function == 'Focal':
        criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
    else:
        criterion = multilabel_categorical_crossentropy
    # 加载训练数据集和验证数据集
    train_data_list=list(zip(train_data_list,Xs_train))
    test_data_list=list(zip(test_data_list,Xs_test))
    train_dataloader = DataLoader(train_data_list, batch_size=args.b)
    test_dataloader = DataLoader(test_data_list, batch_size=args.b)

    best_aim = 0
    best_coverage = 0
    best_accuracy = 0
    best_absolute_true = 0
    best_absolute_false = 1

    best_epoch = 0
    absolute_true_list = []

    for epoch in range(args.e):
        logging.info(f'Epoch: {epoch + 1}')
        model.train()
        arr_loss = []
        i=0
        for data,xs in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            if (i + 1) % 10 == 0:
                logging.info(f"Epoch: [{epoch + 1}/{args.e}], Batch: [{i + 1}/{len(train_dataloader)}]")
            optimizer.zero_grad()
            data = data.to(device)
            xs = xs.to(device)
            logits,features,attention_weights = model(data.x, data.edge_index,data.batch,xs.x,data.seq,adj_matrix)
            loss = criterion(logits, data.y)
            loss.backward()


            optimizer.step()
            arr_loss.append(loss.item())
            i+=1
        # 训练平均损失
        avgl = np.mean(arr_loss)
        logging.info(f"Training Average loss: {avgl}")

        aiming, coverage, accuracy, absolute_true, absolute_false,trainY,trainAttention,trainFeatures = check_model2(model, train_dataloader, device,args.n_class,adj_matrix)
        logging.info("train_data")
        logging.info(F"aiming: {aiming}")
        logging.info(F"coverage: {coverage}")
        logging.info(F"accuracy: {accuracy}")
        logging.info(F"absolute_true: {absolute_true}")
        logging.info(F"absolute_false: {absolute_false}")
        aiming, coverage, accuracy, absolute_true, absolute_false,testY,testAttention,testFeatures = check_model2(model, test_dataloader, device,args.n_class,adj_matrix)
        logging.info("test_data")
        logging.info(F"aiming: {aiming}")
        logging.info(F"coverage: {coverage}")
        logging.info(F"accuracy: {accuracy}")
        logging.info(F"absolute_true: {absolute_true}")
        logging.info(F"absolute_false: {absolute_false}")
        absolute_true_list.append(absolute_true)

        if absolute_true > best_absolute_true: 
            best_absolute_true = absolute_true
            best_aim = aiming
            best_coverage = coverage
            best_accuracy = accuracy
            best_absolute_false = absolute_false
            best_epoch = epoch
            
            torch.save(model.state_dict(), os.path.join(saved_model_path, f"model_{model_num}.pth"))
            
        logging.info('-' * 50)

        scheduler.step()

    logging.info("当absolute_true最高的时候的模型的结果：")
    logging.info(f'time: {datetime.datetime.now()}')
    logging.info(f'best_epoch: {best_epoch + 1}')
    logging.info(f'aim: {best_aim}')
    logging.info(f'coverage: {best_coverage}')
    logging.info(f'accuracy: {best_accuracy}')
    logging.info(f'absolute_true: {best_absolute_true}')
    logging.info(f'absolute_false: {best_absolute_false}')

    logging.info(f'model_num: {model_num}')

if __name__ == '__main__':
    for model_num in range(0,5):
        torch.manual_seed(model_num)  
        torch.cuda.manual_seed(model_num)  
        np.random.seed(model_num)
        parser = argparse.ArgumentParser()
        task = 'MFBP'

        
        train_rootPath='data/origin_train_data1'
        test_rootPath='data/origin_test_data1'

        parser.add_argument('-train_fasta_path', type=str, default=os.path.join(train_rootPath,'train.fasta'),
                        help='Path of the training dataset')
        parser.add_argument('-test_fasta_path', type=str, default=os.path.join(test_rootPath,'test.fasta'),
                            help='Path of the testing dataset')
        
        
        parser.add_argument('-trainESM2_path', type=str, default='/geniusland/home/huangjinpeng/MyModel/ESMsavemodel/feature/MFBP_ESM2_train0.npy',
                            help='Path of the training ESM2 dataset')
        parser.add_argument('-testESM2_path', type=str, default='/geniusland/home/huangjinpeng/MyModel/ESMsavemodel/feature/MFBP_ESM2_test0.npy',
                            help='Path of the testing ESM2 dataset')
        
        parser.add_argument('-GATnodeTrain_path', type=str, default=os.path.join(train_rootPath,'train_data_list3.pkl'),
                            help='Path of the training GATnode dataset')
        parser.add_argument('-GATnodeTest_path', type=str, default=os.path.join(test_rootPath,'test_data_list3.pkl'),
                            help='Path of the testing GATnode dataset')
        
        parser.add_argument('-lr', type=float, default=0.001, help='Learning rate') 
        parser.add_argument('-drop', type=float, default=0.5, help='Dropout rate')
        parser.add_argument('-e', type=int, default=100, help='Maximum number of epochs')
        parser.add_argument('-b', type=int, default=64, help='Batch size')
        parser.add_argument('-hd', type=int, default=64, help='Hidden layer dim')

        
        parser.add_argument('-heads', type=int, default=8, help='Number of heads')
        parser.add_argument('-loss_function', type=str, default='Mcc', help='Loss function')

        parser.add_argument('-model_num', type=str, default=f'MFBP_fineESM5+GAT60_Rescon_noFGM_{model_num}', help='Model number')
        parser.add_argument('-num', type=int, default=model_num, help='Model number')
        parser.add_argument('-att_dim', type=int, default=1024, help='Attention dimension')
        parser.add_argument('-modal_dim', type=list, default=[512,1280], help='Modal dimension')
        parser.add_argument('-node_feature_dim', type=int, default=60, help='node_feature_dim')
        parser.add_argument('-n_class', type=int, default=5, help='Number of classes')
        parser.add_argument('-device', type=str, default='cuda:4', help='Device to use for training')


        parser.add_argument('-d', type=int, default=37, help='Distance threshold to construct a graph, 0-37, 37 means 20A')
        args = parser.parse_args()

        start_time = datetime.datetime.now()
        train(args)

        end_time = datetime.datetime.now()
        logging.info(f'End time(min): {(end_time - start_time).seconds / 60}')
