import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# データセットを定義する
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)

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

# トレーニング関数を定義する
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(outputs)
        # print(labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# テスト関数を定義する
def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 予測値と正解ラベルから、真陽性、真陰性、偽陽性、偽陰性を計算する
            true_positives += ((predicted == 1) & (labels == 1)).sum().item()
            true_negatives += ((predicted == 0) & (labels == 0)).sum().item()
            false_positives += ((predicted == 1) & (labels == 0)).sum().item()
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()

    accuracy = correct / total
    try:
        precision = true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        precision = 0
    
    try:
        recall = true_positives / (true_positives + false_negatives)
    except ZeroDivisionError:
        recall = 0

    try:
        specificity = true_negatives / (true_negatives + false_positives)
    except ZeroDivisionError:
        specificity = 0
    
    try:
        f1_score = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1_score = 0

    return running_loss / len(dataloader), accuracy, precision, recall, specificity, f1_score

# パラメータを設定する
learning_rate = 0.001
batch_size = 2 #適当に変えた所
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# データセットを作成する
data = []  # 入力データのdepthは合わせる必要がある
tes_yugen = np.load("./people1.npy")
tes_soryu = np.load("./people2.npy")
for i in range(10):
    data.append(tes_yugen)
    data.append(tes_soryu)
# PyTorchのクラス内ではtorch.float型が前提
data = torch.tensor(data).float()

targets = []
for i in range(10):
    targets.append(1)
    targets.append(0)
# 正解ラベルはtorch.long型
targets = torch.tensor(targets).long()

dataset = MyDataset(data, targets)

# トレーニングセットとテストセットに分割する
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset)-2, 2])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# モデル、オプティマイザ、損失関数を定義する
model = MyModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# グラフを書くためのx座標とy座標
x_coordinate = []
y_coordinate = []

# トレーニングを実行する
for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, optimizer, criterion, device)
    test_loss, test_accuracy, test_precision, test_recall, test_specificity, test_f1_score = test(model, test_dataloader, criterion, device)

    x_coordinate.append(epoch)
    y_coordinate.append(train_loss)

    # トレーニングのloss,テストのloss,acc,pre,rec,spe,f1を表示
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.4f}, Test Precision = {test_precision:.4f}, Test Recall = {test_recall:.4f}, Test Specificity = {test_specificity:.4f}, Test F1 Score = {test_f1_score:.4f}")

# グラフを書いてカレントディレクトリに保存
plt.plot(x_coordinate, y_coordinate)
plt.savefig("loss")

# モデルを保存する
torch.save(model.state_dict(), "my_model.pt")