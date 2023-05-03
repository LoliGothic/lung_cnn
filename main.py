import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
        self.conv1 = nn.Conv3d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2)
        self.fc1 = nn.Linear(64*4*4*4, 256)
        self.fc2 = nn.Linear(256, 2)

    # モデルの順伝播を定義する
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(-1, 64*4*4*4)
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
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return running_loss / len(dataloader), accuracy, precision, recall, specificity, f1_score

# パラメータを設定する
learning_rate = 0.001
batch_size = 32
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データセットを作成する
data = ... # Load voxel data
targets = ... # Load class labels
dataset = MyDataset(data, targets)

# トレーニングセットとテストセットに分割する
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset)-100, 100])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# モデル、オプティマイザ、損失関数を定義する
model = MyModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# TensorBoardを設定する
writer = SummaryWriter()

# トレーニングを実行する
for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, optimizer, criterion, device)
    test_loss, test_accuracy, test_precision, test_recall, test_specificity, test_f1_score = test(model, test_dataloader, criterion, device)

    # TensorBoardにlossとaccuracyのログを出す
    writer.add_scalar("Training Loss", train_loss, epoch)
    writer.add_scalar("Testing Loss", test_loss, epoch)
    writer.add_scalar("Testing Accuracy", test_accuracy, epoch)

    # トレーニングのloss,テストのloss,acc,pre,rec,spe,f1を表示
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.4f}, Test Precision = {test_precision:.4f}, Test Recall = {test_recall:.4f}, Test Specificity = {test_specificity:.4f}, Test F1 Score = {test_f1_score:.4f}")

# モデルを保存する
torch.save(model.state_dict(), "my_model.pt")