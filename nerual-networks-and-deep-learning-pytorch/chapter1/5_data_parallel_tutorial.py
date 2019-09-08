"""选读：数据并行处理（多GPU），如果是多个GPU将每个bacth平均分配到多个GPU
https://github.com/zergtant/pytorch-handbook/blob/master/chapter1/5_data_parallel_tutorial.ipynb
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RandomDataset(Dataset):
    """制作一个虚拟（随机）数据集， 你只需实现 __getitem__
    """
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output



if __name__ == "__main__":
    ### 1. Parameters and DataLoaders
    input_size = 5
    output_size = 2

    batch_size = 30
    data_size = 100

    rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)


    ### 2.use parallel GPU
    model = Model(input_size, output_size)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    # 如果是多个GPU将每个bacth平均分配到多个GPU
    for data in rand_loader:
        input = data.to(device)
        output = model(input)
        print("Outside: input size", input.size(),
            "output_size", output.size())