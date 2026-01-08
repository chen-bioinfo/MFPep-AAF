import re
import numpy as np
import torch
from utils.encoding_methods import sa_encoding, onehot_encoding, pssm_encoding, position_encoding, hhm_encoding,  cat
from torch_geometric.data import Data
import os

# 返回肽序列name列表，seq列表，label列表
def load_seqs(fasta_path):
    """
    :param fasta_path: source file name in fasta format
    :return:
        ids: name list
        seqs: peptide sequence list
        labels: label list
    """
    ids = []
    seqs = []
    labels = []
    t = 0
    current_path = os.getcwd()
    # data_dir_path = os.path.join(current_path, 'data')
    data_dir_path = '/geniusland/home/huangjinpeng/MyModel/data'
   
    all_seq_label_path = os.path.join(data_dir_path, 'all_data.npy')
    all_seq_label = np.load(all_seq_label_path, allow_pickle=True).item() # 加载seq-label字典
    all_seq_label_upper = {seq.upper(): label for seq, label in all_seq_label.items()}
    with open(fasta_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line[0] == '>':
                t = line # 注意这里没有去除'>'
            else:
                seqs.append(line)
                ids.append(t)
                if line.upper() in all_seq_label_upper:
                    labels.append(np.array(all_seq_label_upper[line.upper()]))
                else:
                    print(line)
                    # exit(0)
                    labels.append(np.array([0, 0, 0, 0, 0]))

    return ids, seqs, labels

def load_embedding(fasta_path, trainESMembedding_path, testESMembedding_path):
    ids = []
    seqs = []
    labels = []
    AAindexembedding=[]
    ESMembedding=[]
    GATembedding=[]
    gearnetembedding=[]
    if 'train' in fasta_path:
        GATembeddingdict = np.load('/geniusland/home/huangjinpeng/MyModel/GATfeature/MFBP_GAT60_train0.npy', allow_pickle=True).item()
        ESMembeddingdict = np.load(trainESMembedding_path, allow_pickle=True).item()
    else:
        GATembeddingdict = np.load('/geniusland/home/huangjinpeng/MyModel/GATfeature/MFBP_GAT60_test0.npy', allow_pickle=True).item()
        ESMembeddingdict = np.load(testESMembedding_path, allow_pickle=True).item()
    with open(fasta_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        for i in range(0, len(lines), 2):
            name = lines[i].lstrip(">")
            seq = lines[i+1]
            ESMembedding.append(ESMembeddingdict[seq])
            # GATembedding.append(GATembeddingdict[seq])
            seqs.append(seq)
            ids.append(name)
    return ids, GATembedding, ESMembedding, AAindexembedding,gearnetembedding
    
def load_data(fasta_path, threshold=10, add_self_loop=True):
    """
    :param fasta_path: file path of fasta
    :param npz_dir: dir that saves npz files
    :param pdb_dri: dir that saves pdb files
    :param threshold: threshold for build adjacency matrix
    :param label: labels
    :return:
        data_list: list of Data
        labels: list of labels
    """
    ids, seqs, labels = load_seqs(fasta_path)
    # 获得结构信息
    npz_dir = '/'.join(fasta_path.split('/')[:-1]) + '/npz/'
    As = get_cmap(npz_dir, ids, threshold, add_self_loop)
    # As= new_get_cmap(pdb_dir, ids, threshold, add_self_loop)
 
    # 获得四个残基信息
    one_hot_encodings = onehot_encoding(seqs)
    print("one_hot",one_hot_encodings[0].shape)

    position_encodings = position_encoding(seqs)
    print("position",position_encodings[0].shape)

    hhm_dir = '/'.join(fasta_path.split('/')[:-1]) + '/hhm/'
    hhm_encodings = hhm_encoding(ids, hhm_dir)
    print("hmm",hhm_encodings[0].shape)

    Xs = cat(one_hot_encodings,position_encodings,hhm_encodings)
    

    n_samples = len(As)
    data_list = []
    for i in range(n_samples):
        data_list.append(to_parse_matrix(As[i], Xs[i], labels[i], ids[i], seqs[i]))
    return data_list, labels

def load_data2(fasta_path, trainESMembedding_path, testESMembedding_path):
    ids, seqs, labels = load_seqs(fasta_path)
    _, GATembedding, ESMembedding, AAembedding,gearnetembedding = load_embedding(fasta_path, trainESMembedding_path, testESMembedding_path)

    print("ESM",ESMembedding[0].shape,ESMembedding[1].shape,ESMembedding[2].shape)
    Xs = cat(ESMembedding)
    data_list=[]
    for i in range(len(Xs)):
        s = torch.tensor(Xs[i], dtype=torch.float32)
        data_list.append(Data(x=s))
    AAembedding = torch.tensor(AAembedding)
    return data_list, AAembedding

def load_data3(fasta_path, threshold=10, add_self_loop=True):
    ids, seqs, labels = load_seqs(fasta_path)
    _, MPNNembedding, ESMembedding, AAembedding,gearnetembedding = load_embedding(fasta_path)
    print(len(ESMembedding),len(gearnetembedding))
    print("ESM",ESMembedding[0].shape,ESMembedding[1].shape)
    print("gearnet",gearnetembedding[0].shape,gearnetembedding[1].shape)
    one_hot_encodings = onehot_encoding(seqs)
    position_encodings = position_encoding(seqs)
    print("pos",position_encodings[0].shape,position_encodings[1].shape)
    # print(position_encodings[0].mean(axis=1))

    # pssm_dir = '/'.join(fasta_path.split('/')[:-1]) + '/pssm/'
    # pssm_encodings = pssm_encoding(ids, pssm_dir)
    # print("pssm",pssm_encodings[0].shape,pssm_encodings[1].shape)
    # print(pssm_encodings[0].mean(axis=1))
    hhm_dir = '/'.join(fasta_path.split('/')[:-1]) + '/hhm/'
    hhm_encodings = hhm_encoding(ids, hhm_dir)
    print("hmm",hhm_encodings[0].shape,position_encodings[1].shape)
    # Xs = cat(position_encodings, pssm_encodings,ESMembedding,AAembedding,MPNNembedding)
    for i in range(len(ESMembedding)):
        temp=[position_encodings[i].shape[0],hhm_encodings[i].shape[0],ESMembedding[i].shape[0],gearnetembedding[i].shape[0]]
        if len(set(temp))>1:
            print(i)
            print(position_encodings[i].shape,hhm_encodings[i].shape,ESMembedding[i].shape,gearnetembedding[i].shape)
            exit(0)
    # Xs = cat(position_encodings,hhm_encodings,ESMembedding,gearnetembedding)
    # data_list=[]
    # for i in range(len(Xs)):
    #     s = torch.tensor(Xs[i], dtype=torch.float32)
    #     data_list.append(Data(x=s))
    # return [one_hot_encodings,position_encodings,hhm_encodings, ESMembedding,gearnetembedding],labels
    return ESMembedding

def load_data4(fasta_path, threshold=10, add_self_loop=True):
    ids, seqs, labels = load_seqs(fasta_path)
    data_list=[]
    for i in range(len(seqs)):
        y=torch.tensor(labels[i], dtype=torch.float32)
        data_list.append(Data(y=y,seq=seqs[i],id=ids[i]))

    return data_list, labels

# 结构特征 list_A是邻接矩阵，list_E是对应边的信息
def get_cmap(npz_folder, ids, threshold, add_self_loop=True):
    if npz_folder[-1] != '/':
        npz_folder += '/'

    list_A = []
    list_E = []

    for id in ids:
        npz = id[1:] + '.npz' # id以'>'开头，npz文件名不含'>'
        f = np.load(npz_folder + npz)

        # 获得四个信息，距离和角度
        mat_dist = f['dist']
        mat_omega = f['omega']
        mat_theta = f['theta']
        mat_phi = f['phi']

        # 第一条肽序列长度是25，前两维大小取决于该序列的长度，最后一维是固定的
        # print(type(mat_dist), type(mat_omega), type(mat_theta), type(mat_phi))
        # <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'>
        # print(mat_dist.shape, mat_omega.shape, mat_theta.shape, mat_phi.shape)
        # (25, 25, 37) (25, 25, 25) (25, 25, 25) (25, 25, 13)
        # break

        """ 
        The distance range (2 to 20 Å) is binned into 36 equally spaced segments, 0.5 Å each, 
        plus one bin indicating that residues are not in contact.
            - Improved protein structure prediction using predicted interresidue orientations: 
        """
        dist = np.argmax(mat_dist, axis=2)  # 37 equally spaced segments
        omega = np.argmax(mat_omega, axis=2)
        theta = np.argmax(mat_theta, axis=2)
        phi = np.argmax(mat_phi, axis=2)

        A = np.zeros(dist.shape, dtype=np.int32)

        threshold = 37 # 重新设置一个阈值
        A[dist < threshold] = 1
        A[dist == 0] = 0
        # A[omega < threshold] = 1
        # 对角线元素置为1
        if add_self_loop:
            A[np.eye(A.shape[0]) == 1] = 1
        else:
            A[np.eye(A.shape[0]) == 1] = 0

        dist[A == 0] = 0
        omega[A == 0] = 0
        theta[A == 0] = 0
        phi[A == 0] = 0

        dist = np.expand_dims(dist, -1) # 在最后一维增加一个维度
        omega = np.expand_dims(omega, -1)
        theta = np.expand_dims(theta, -1)
        phi = np.expand_dims(phi, -1)

        edges = dist
        edges = np.concatenate((edges, omega), axis=-1)
        edges = np.concatenate((edges, theta), axis=-1)
        edges = np.concatenate((edges, phi), axis=-1)
        # print(A.shape) # (25, 25)
        # print(edges.shape) # (25, 25, 4)
        # break

        list_A.append(A)
        list_E.append(edges)

    return list_A

def new_get_cmap(pdb_folder, ids, threshold, add_self_loop=True):
    if pdb_folder[-1] != '/':
        pdb_folder += '/'
    
    list_A = []
    D = []

    for i, id in enumerate(ids):
        # print(i)
        num = 0
        pdb = id[1:] + '.pdb' # id以'>'开头，pdb文件名不含'>'
        with open(pdb_folder + pdb, 'r') as f:
            lines = f.readlines()
            for line in lines:
                l = line.strip().split()
                if l[0] == 'ATOM' and l[2] == 'CA':
                    # print(id)
                    D.append((float(l[6]), float(l[7]), float(l[8])))
                    num += 1
    
        A = np.zeros((num, num), dtype=np.int32)
        if add_self_loop:
            A[np.eye(A.shape[0]) == 1] = 1
        else:
            A[np.eye(A.shape[0]) == 1] = 0
        
        for i in range(num):
            # print(i)
            for j in range(i + 1, num):
                dist = np.linalg.norm(np.array(D[i]) - np.array(D[j]))
                if dist < threshold:
                    A[i][j] = 1
                    A[j][i] = 1
        
        list_A.append(A)
    
    return list_A


# 获得四个信息，距离和角度
# mat_dist = f['dist']
# mat_omega = f['omega']
# mat_theta = f['theta']
# mat_phi = f['phi']

# print(mat_dist.shape, mat_omega.shape, mat_theta.shape, mat_phi.shape)
# get_cmap(npz_dir, ids, 37)  

# A:(25, 25), X: (25, 80), E:(25, 25, 4), Y:(5,)
def to_parse_matrix(A, X, Y, id, seq, eps=1e-6):
    """
    :param A: Adjacency matrix with shape (n_nodes, n_nodes)
    :param X: node embedding with shape (n_nodes, n_node_features)
    :return:
    """
    num_row, num_col = A.shape
    rows = []
    cols = []

    for i in range(num_row):
        for j in range(num_col):
            if A[i][j] >= eps:
                rows.append(i)
                cols.append(j)

    edge_index = torch.tensor([rows, cols], dtype=torch.int64)
    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor([Y], dtype=torch.float32)
    # print(x.shape, edge_index.shape, edge_attr.shape, y.shape)
    # 输出：torch.Size([25, 80]) torch.Size([2, 483]) torch.Size([483, 4]) torch.Size([1, 5])

    return Data(x=x, edge_index=edge_index, y=y, id=id, seq=seq)