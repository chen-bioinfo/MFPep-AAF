from Bio import SeqIO

# 定义标准氨基酸集合（20种标准氨基酸）
standard_aa = set("ACDEFGHIKLMNPQRSTVWY")

# 输入和输出文件路径
input_fasta = "/geniusland/home/huangjinpeng/MyModel/data/MFTP_train_data/train.txt"
output_fasta = "/geniusland/home/huangjinpeng/MyModel/data/MFTP_train_data/filter_train.txt"

# 筛选并写入新文件
with open(output_fasta, "w") as out_file:
    for record in SeqIO.parse(input_fasta, "fasta"):
        sequence = str(record.seq).upper()  # 转换为大写
        # 检查是否所有字符都是标准氨基酸
        if all(aa in standard_aa for aa in sequence):
            out_file.write(f">{record.id}\n{sequence}\n")

print("筛选完成！有效序列已保存至:", output_fasta)