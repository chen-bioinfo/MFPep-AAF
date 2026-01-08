import os
import numpy as np

current_path = os.getcwd()
all_data_path = os.path.join(current_path, 'all_data.npy')

all_data = np.load(all_data_path, allow_pickle=True).item()

# print(all_data)
labels = ['ACP', 'ADP', 'AHP', 'AIP', 'AMP']
for i in range(len(labels)):
    num = 0
    l = 0
    for key in all_data:
        label = all_data[key]
        if label[i] == 1:
            num += 1
            l += len(key)

    print(f'Number of {labels[i]}: {num}')
    print(f'Length of {labels[i]}: {l / num}')