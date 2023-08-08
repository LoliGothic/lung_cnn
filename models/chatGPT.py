import torch
import torch.nn as nn

# モデルを定義する
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2)
        self.fc1 = nn.Linear(262144, 256)  # nn.Linear（入力サイズ,出力サイズ）割り切れる値に変えてみた(poolingで出てきた値)
        self.fc2 = nn.Linear(256, 2)  # 2値分類だから最後は2

    # モデルの順伝播を定義する
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(-1, 262144)  # 割り切れる値に変えてみた(poolingで出てきた値)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x