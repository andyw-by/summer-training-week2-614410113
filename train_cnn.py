# =========================================================
# 🧠 訓練主程式
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
import pandas as pd
import numpy as np
import wandb
import os
from rfmid_dataset import RFMiDDataset
from focal_loss import FocalLoss

# 创建一个包装器类来处理数据类型问题
class FixedRFMiDDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.original_dataset = RFMiDDataset(csv_file, img_dir, transform)
        
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        # 获取原始数据
        image, label = self.original_dataset[idx]
        return image, label

# Step 0: 設定超參數（可手動調整）
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
IMG_SIZE = 224
TRAIN_SPLIT = 0.8
MODEL_NAME = "resnet18"  # 可選: "vgg16", "resnet18", "resnet50"
USE_FOCAL_LOSS = True  # 是否使用 Focal Loss
ALPHA = 1.0
GAMMA = 2.0

# Step 1: 初始化 wandb
wandb.init(
    project="RFMiD CNN Classification",
    config={
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
        "model": MODEL_NAME,
        "use_focal_loss": USE_FOCAL_LOSS,
        "alpha": ALPHA,
        "gamma": GAMMA
    }
)
config = wandb.config

# Step 2: 定義圖像轉換
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Step 3: 建立 RFMiD Dataset
# 使用 os.path.expanduser 扩展 ~ 符号
csv_path = os.path.expanduser('~/Summer-Training-Week2/Retinal-disease-classification/labels.csv')
img_dir = os.path.expanduser('~/Summer-Training-Week2/Retinal-disease-classification/images/images/')

# 检查文件是否存在
if not os.path.exists(csv_path):
    print(f"错误：找不到 CSV 文件 {csv_path}")
    exit(1)

if not os.path.exists(img_dir):
    print(f"错误：找不到图像目录 {img_dir}")
    exit(1)


# 使用修改后的 CSV 文件
dataset = RFMiDDataset(
    csv_file=csv_path,
    img_dir=img_dir,
    transform=transform
)

# Step 4: 切分訓練集與驗證集
train_size = int(TRAIN_SPLIT * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Step 5: 建立 DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Step 6: 初始化模型
def get_model(model_name, num_classes=28):
    if model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model

model = get_model(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Step 7: 定義損失函數與 optimizer
if USE_FOCAL_LOSS:
    criterion = FocalLoss(alpha=ALPHA, gamma=GAMMA)
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Step 8: 開始訓練迴圈
train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(NUM_EPOCHS):
    # 訓練階段
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 前向傳播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向傳播與優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 計算統計數據
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # 驗證階段
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    # 記錄到 wandb
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc
    })
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# Step 9: 儲存模型
torch.save(model.state_dict(), f"{MODEL_NAME}_retinal_model.pth")

# 清理临时文件
if os.path.exists(temp_csv_path):
    os.remove(temp_csv_path)

wandb.finish()