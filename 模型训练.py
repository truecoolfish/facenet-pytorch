from facenet_pytorch import InceptionResnetV1
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score, recall_score
import numpy as np

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"运行设备: {device}")

# 数据集路径定义
dataset_dir = "G:\\智能实验室项目\\完整数据集（含训练、测试和验证）"  # 数据集根目录
train_dir = f"{dataset_dir}/train"  # 训练集目录
val_dir = f"{dataset_dir}/val"      # 验证集目录

# 数据增强和预处理
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),      # 随机水平翻转
    transforms.RandomRotation(10),               # 随机旋转
    transforms.ToTensor(),                       # 转为张量
    transforms.Normalize([0.5], [0.5])           # 归一化到 [-1, 1]
])

# 加载训练集和验证集
train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms)

# 统计每个类别的样本数量
class_counts = np.bincount([label for _, label in train_dataset.samples])
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)  # 类别权重
print(f"类别样本数量: {class_counts}")
print(f"类别权重: {class_weights}")

# 使用 WeightedRandomSampler 平衡样本分布
sample_weights = [class_weights[label] for _, label in train_dataset.samples]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# 定义数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 加载预训练的 InceptionResnetV1 并修改分类器
num_classes = len(train_dataset.classes)
model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=num_classes)
model = model.to(device)

# 定义损失函数（加权交叉熵）和优化器
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))  # 加权损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义训练和验证函数
def train_model(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失
        running_loss += loss.item()

        # 记录预测结果和真实标签
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    # 计算训练集的F1值和召回率
    f1 = f1_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    print(f"[训练] Epoch {epoch+1} - Loss: {running_loss/len(train_loader):.4f} | F1: {f1:.4f} | Recall: {recall:.4f}")


def validate_model(model, val_loader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 记录损失
            running_loss += loss.item()

            # 记录预测结果和真实标签
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 计算验证集的F1值和召回率
    f1 = f1_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    print(f"[验证] Epoch {epoch+1} - Loss: {running_loss/len(val_loader):.4f} | F1: {f1:.4f} | Recall: {recall:.4f}")

    # 返回分类报告
    return classification_report(all_labels, all_preds, target_names=val_dataset.classes, digits=4)


# 开始训练
epochs = 10
for epoch in range(epochs):
    train_model(model, train_loader, criterion, optimizer, epoch)
    report = validate_model(model, val_loader, criterion, epoch)

# 打印最终分类报告
print("\n最终验证分类报告：")
print(report)

# 保存模型
torch.save(model.state_dict(), "inceptionresnetv1_balanced.pth")
print("模型已保存为 'inceptionresnetv1_balanced.pth'")
