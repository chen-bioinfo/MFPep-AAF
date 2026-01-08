# import torch
# import esm

# # 定义本地模型文件的路径
# model_path = "./ESM_650M/esm2_t33_650M_UR50D.pt"

# # 从本地路径加载模型和字母表
# model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
# batch_converter = alphabet.get_batch_converter()

# # 定义蛋白质序列
# data = [
#     ("protein1", "MKTIIALSYIFCLVFADYKDDDDA"),
# ]

# # 将数据转换为模型输入格式
# batch_labels, batch_strs, batch_tokens = batch_converter(data)

# # 关闭模型的梯度计算（推理模式）
# with torch.no_grad():
#     results = model(batch_tokens, repr_layers=[33],  return_contacts=True)
    
# # 获取表示层
# token_representations = results["representations"][33]

# # 将表示转换为每个氨基酸的平均表示
# protein_representations = token_representations.mean(1)

# # 输出结果
# print("蛋白质表示形状:", protein_representations.shape)
# print("蛋白质表示:", protein_representations)
import torch
from transformers import EsmTokenizer, EsmModel
import numpy as np
# 定义本地模型文件的路径
# model_path = "./SaProt_35M_AF2"
model_path = "/geniusland/home/huangjinpeng/MyModel/ESM_650M"

# 加载标记器和模型
tokenizer = EsmTokenizer.from_pretrained(model_path)
model = EsmModel.from_pretrained(model_path)

# 将模型移动到GPU（如果可用）
device = "cuda:2" if torch.cuda.is_available() else "cpu"
model.to(device)

# 定义蛋白质序列
# seq = "MdEvVpQpLrVyQdYaKv"
path='/geniusland/home/huangjinpeng/MyModel/data/origin_test_data1/test.fasta'
datadict={}
with open(path, "r") as f:  # 替换为你的文件名
    lines = [line.strip() for line in f if line.strip()]
    for i in range(0, len(lines), 2):
        name = lines[i].lstrip(">")
        seq = lines[i+1]

        tokens = tokenizer.tokenize(seq)
        # print("Tokenized sequence:", tokens)
        # 将序列转换为模型输入格式
        inputs = tokenizer(seq, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 获取模型输出，确保返回隐藏状态
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # 获取最后一层的隐藏状态
        hidden_states = outputs.hidden_states[-1]

        # 获取蛋白质表示（可以使用平均或其他方式进行池化）
        # protein_representation = hidden_states.mean(dim=1)

        # 输出蛋白质表示的形状和内容
        print("Protein representation shape:", hidden_states.shape)
        datadict[seq] = hidden_states[0][1:-1].cpu().numpy()
for key, value in datadict.items():
    print(f"{key}: {value.shape}")
    break
np.save('feature/origin_preESM2_test.npy', datadict, allow_pickle=True)



# loaded_dict = np.load('ESM2_train.npy', allow_pickle=True).item()
# for seq_id, matrix in loaded_dict.items():
#     matrix = np.array(matrix)
#     # print(matrix.shape)
#     row_means = matrix.mean(axis=1)
#     row_stds = matrix.std(axis=1)
#     print(row_means)
#     print(row_stds)
#     if not (np.allclose(row_means, 0, atol=1e-3) and np.allclose(row_stds, 1, atol=1e-3)):
#         print(f"{seq_id} 的特征矩阵不是标准化的")
#         break