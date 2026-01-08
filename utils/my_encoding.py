import os
import numpy as np
from utils.my_utils import load_hhm, load_pssm

current_path = os.path.dirname(os.path.abspath("__file__"))
# print(current_path) # /geniusland/home/huangjinpeng/sAMPpred-GAT
data_dir = os.path.join(current_path, 'data')

# 先判断是train_data还是test_data
seq_data_dict_path = os.path.join(data_dir, 'seq_data_dict.npy')
seq_data_dict = np.load(seq_data_dict_path, allow_pickle=True).item()
# print(len(seq_data_dict.keys())) # 5917

# 再找到具体的name
train_seq_name_dict_path = os.path.join(data_dir, 'train_data', 'train_seq_name_dict.npy')
train_seq_name_dict = np.load(train_seq_name_dict_path, allow_pickle=True).item()
test_seq_name_dict_path = os.path.join(data_dir, 'test_data', 'test_seq_name_dict.npy')
test_seq_name_dict = np.load(test_seq_name_dict_path, allow_pickle=True).item()
# print(len(train_seq_name_dict.keys())) # 4749
# print(len(test_seq_name_dict.keys())) # 1168

# 把特征文件夹路径保存下来
train_pssm_path = os.path.join(data_dir, 'train_data', 'pssm')
test_psssm_path = os.path.join(data_dir, 'test_data', 'pssm')
train_hhm_path = os.path.join(data_dir, 'train_data', 'hhm')
test_hhm_path = os.path.join(data_dir, 'test_data', 'hhm')


def my_hhm_encoding(seqs):
    res = []
    for seq in seqs:
        name = ''
        data_class = seq_data_dict[seq]
        hhm_dir = os.path.join(data_dir, data_class, 'hhm/')
        hhm_fs = os.listdir(hhm_dir + 'output/')
        if data_class == 'train_data':
            name = train_seq_name_dict[seq]
        else:
            name = test_seq_name_dict[seq]
        assert name + '.hhm' in hhm_fs
        tmp = load_hhm(name, hhm_dir + 'output/')
        res.append(np.array(tmp))
        
    return res

def my_pssm_encoding(seqs):
    res = []
    for seq in seqs:
        name = ''
        data_class = seq_data_dict[seq]
        pssm_dir = os.path.join(data_dir, data_class, 'pssm/')
        pssm_fs = os.listdir(pssm_dir + 'output/')
        if data_class == 'train_data':
            name = train_seq_name_dict[seq]
        else:
            name = test_seq_name_dict[seq]
        if name + '.pssm' in pssm_fs:
            tmp = load_pssm(name, pssm_dir + 'output/')
            res.append(np.array(tmp))
        else:
            tmp = load_pssm(name, pssm_dir + 'blosum/')
            res.append(np.array(tmp)) 
    return res

def my_getcmap(seqs, threshold, add_self_loop=True):
    list_A = []
    list_E = []

    for seq in seqs:
        data_class = seq_data_dict[seq]
        npz_folder = ''
        name = ''
        if data_class == 'train_data':
            npz_folder = os.path.join(data_dir, data_class, 'npz/')
            name = train_seq_name_dict[seq]
        else:
            npz_folder = os.path.join(data_dir, data_class, 'npz/')
            name = test_seq_name_dict[seq]

        f = np.load(npz_folder + name + '.npz')
        
        # --------------下面和get_cmap一样--------------
        
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

        A = np.zeros(dist.shape, dtype=np.int)

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

    return list_A, list_E