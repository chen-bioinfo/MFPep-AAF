import os
import numpy as np

# 对seq进行独热编码，三维矩阵
def onehot_encoding(seqs):
    residues = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K',
                'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

    encoding_map = np.eye(len(residues))

    residues_map = {}
    for i, r in enumerate(residues):
        residues_map[r] = encoding_map[i]

    res_seqs = []

    for seq in seqs:
        tmp_seq = [residues_map[r] for r in seq]
        res_seqs.append(np.array(tmp_seq))

    return res_seqs

# 位置编码，三维矩阵
def position_encoding(seqs):
    """
    Position encoding features introduced in "Attention is all your need",
    the b is changed to 1000 for the short length of peptides.
    """
    d = 20
    b = 1000
    res = []
    for seq in seqs:
        N = len(seq)
        value = []
        for pos in range(N):
            tmp = []
            for i in range(d // 2):
                tmp.append(pos / (b ** (2 * i / d)))
            value.append(tmp)
        value = np.array(value)
        pos_encoding = np.zeros((N, d))
        pos_encoding[:, 0::2] = np.sin(value[:, :])
        pos_encoding[:, 1::2] = np.cos(value[:, :])
        res.append(pos_encoding)
    return res

# pssm矩阵，返回二维矩阵，表示单个肽序列的pssm矩阵
def load_pssm(query, pssm_path):
    """
    :param query: query id
    :param pssm_path: dir saving pssm files
    """
    if pssm_path[-1] != '/': pssm_path += '/'
    with open(pssm_path + query + '.pssm', 'r') as f:
        lines = f.readlines()
        res = []
        for line in lines[3:]:
            line = line.strip()
            lst = line.split(' ')
            while '' in lst:
                lst.remove('')
            if len(lst) == 0:
                break
            r = lst[2:22]
            r = [int(x) for x in r]
            res.append(r)
    # 转换为 NumPy 数组以方便归一化
    # res_array = np.array(res)
    
    # # Z-score 归一化
    # mean_vals = np.mean(res_array, axis=0)
    # std_vals = np.std(res_array, axis=0)
    # normalized_res_array = (res_array - mean_vals) / std_vals
    
    # return normalized_res_array.tolist()
    return res

# hhm矩阵，返回二维矩阵，表示单个肽序列的hhm矩阵    
def load_hhm(query, hhm_path):
    """
    :param query: query id
    :param hhm_path: dir saving hhm files
    """
    if hhm_path[-1] != '/': hhm_path += '/'
    with open(hhm_path + query + '.hhm', 'r') as f:
        lines = f.readlines()
        res = []
        tag = 0
        for line in lines:
            line = line.strip()
            if line == '#': # 
                tag = 1
                continue
            if tag != 0 and tag < 5:
                tag += 1
                continue
            if tag >= 5:
                line = line.replace('*', '0')
                lst = line.split('\t')
                if len(lst) >= 20:
                    tmp0 = [int(lst[0].split(' ')[-1])]  # First number
                    tmp1 = list(map(int, lst[1:20]))
                    tmp0.extend(tmp1)
                    normed = [i if i == 0 else 2 ** (-0.001 * i) for i in tmp0]
                    res.append(normed)
    # 转换为 NumPy 数组以方便归一化
    # res_array = np.array(res)
    
    # # Z-score 归一化
    # mean_vals = np.mean(res_array, axis=0)
    # std_vals = np.std(res_array, axis=0)
    # normalized_res_array = (res_array - mean_vals) / std_vals
    
    # return normalized_res_array.tolist()
    return res


def pssm_encoding(ids, pssm_dir):
    """
    parser pssm features
    """
    if pssm_dir[-1] != '/': pssm_dir += '/'
    pssm_fs = os.listdir(pssm_dir + 'output/')

    res = []
    for id in ids:
        name = id
        if id[0] == '>': name = id[1:]
        if name + '.pssm' in pssm_fs:
            # psiblast
            tmp = load_pssm(name, pssm_dir + 'output/')
            res.append(np.array(tmp))
        else:
            # blosum 感觉没用
            tmp = load_pssm(name, pssm_dir + 'blosum/')
            res.append(np.array(tmp)) 
    return res

def hhm_encoding(ids, hhm_dir):
    """
    parser pssm features
    """
    if hhm_dir[-1] != '/': hhm_dir += '/'
    hhm_fs = os.listdir(hhm_dir + 'output/')
    res = []
    for id in ids:
        name = id
        if id[0] == '>': name = id[1:]
        assert name + '.hhm' in hhm_fs
        tmp = load_hhm(name, hhm_dir + 'output/')
        res.append(np.array(tmp))

    return res

def load_sa(id, sa_dir):
    """
    :param id: query id
    :param sa_dir: dir saving sa files
    """
    if sa_dir[-1] != '/': sa_dir += '/'
    with open(sa_dir + id + '.txt', 'r') as f:
        lines = f.readlines()
        return lines[0].strip()

# 加载完sa序列后，需要输入SaProt模型中
def sa_encoding(ids, sa_dir):
    seqs = []
    for id in ids:
        if id[0] == '>': id = id[1:]
        seq = load_sa(id, sa_dir)
        seqs.append(seq)

    return seqs

def add(e1, e2):
    res = []
    for i in range(len(e1)):
        res.append(e1[i] + e2[i])
    return res

# 拼接多个特征矩阵
def cat(*args):
    """
    :param args: feature matrices
    """
    res = args[0]
    for matrix in args[1:]:
        for i in range(len(matrix)):
            res[i] = np.hstack((res[i], matrix[i]))
    return res

def cat2(*matrices):
    """
    水平连接多个矩阵（二维数组）
    要求所有矩阵行数相同
    
    参数:
        matrices: 可变数量的二维数组
    
    返回:
        水平连接后的新数组
    """
    # 获取所有矩阵的行数（应该相同）
    num_rows = matrices[0].shape[0]
    
    # 创建一个空列表存储每一行的连接结果
    rows = []
    result = np.empty((7873, 50, 4412), dtype=np.float32)
    # 对每一行进行处理
    for i in range(num_rows):
        # 收集当前行在所有矩阵中的片段
        row_parts = [mat[i] for mat in matrices]
        # print(len(row_parts))
        # 水平连接当前行的所有片段
        full_row = np.hstack(row_parts)
        rows.append(full_row)
        # result[i] = np.array(full_row, dtype=np.float32)
    # print(len(result),len(result[0]),len(result[0][0]))
    # 将所有连接好的行组合成新数组
    return rows

if __name__ == '__main__':
    encoding = position_encoding(['ARFGD', 'AAAAAA'])