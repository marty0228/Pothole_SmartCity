#!pip install ultralytics

from google.colab import drive
import os

#Training 폴더 경로로 변경해주시면 됩니다.
drive.mount('/content/drive')
base_path = '/content/drive/MyDrive/DL_Project'

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import cv2

from ultralytics import YOLO

import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold
import glob

class KFoldPotholeDataset(Dataset):
    def __init__(self, img_paths, tdisp_paths, transform=None):
        self.img_paths = img_paths
        self.tdisp_paths = tdisp_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def _decode_tdisp_to_depth(self, tdisp_rgb):
        # OpenCV 색 공간 변환을 위해 HSV로 변경
        hsv = cv2.cvtColor(tdisp_rgb, cv2.COLOR_RGB2HSV)

        # 색상 채널 추출
        h = hsv[:, :, 0].astype(np.float32)

        # 파란색이 0.0, 빨간색이 1.0이 되도록 변환
        depth = (120.0 - h) / 120.0

        # 빨간색 주변부 노이즈 및 배경 예외 처리 가두기
        depth = np.clip(depth, 0.0, 1.0)

        return depth

    def __getitem__(self, idx):
        # RGB 원본 이미지 불러오기
        image = np.array(Image.open(self.img_paths[idx]).convert('RGB'))

        # 정답 원본 이미지 불러오기 
        tdisp_rgb = np.array(Image.open(self.tdisp_paths[idx]).convert('RGB'))

        # 3차원을 1차원으로 변경
        tdisp_depth = self._decode_tdisp_to_depth(tdisp_rgb)

        # 데이터 증강 적용 
        if self.transform:
            augmented = self.transform(image=image, mask=tdisp_depth)
            image = augmented['image']
            tdisp_depth = augmented['mask']

        # 파이토치 규격에 맞추기
        label_tensor = tdisp_depth.unsqueeze(0)

        return image, label_tensor

all_rgb_paths = sorted(glob.glob(f"{base_path}/training/rgb/*") + glob.glob(f"{base_path}/validation/rgb/*"))
all_tdisp_paths = sorted(glob.glob(f"{base_path}/training/tdisp/*") + glob.glob(f"{base_path}/validation/tdisp/*"))

train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5), 
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5), 
    A.RandomBrightnessContrast(p=0.2), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

  def forward(self, x):
      return self.conv(x)
  

class PotholeUNet(nn.Module):
    def __init__(self):
        super(PotholeUNet, self).__init__()

        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2) 

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2) 

        self.down3 = DoubleConv(128, 256)  
        self.pool3 = nn.MaxPool2d(2)      

        self.down4 = DoubleConv(256, 512)  
        self.pool4 = nn.MaxPool2d(2)      

        self.bottleneck = DoubleConv(512, 1024) 

        self.upConv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up4 = DoubleConv(1024, 512)

        self.upConv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = DoubleConv(512, 256)

        self.upConv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = DoubleConv(256, 128)

        self.upConv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        p1 = self.pool1(x1)

        x2 = self.down2(p1)
        p2 = self.pool2(x2)

        x3 = self.down3(p2)
        p3 = self.pool3(x3)

        x4 = self.down4(p3)
        p4 = self.pool4(x4)

        b = self.bottleneck(p4)

        u4 = self.upConv4(b)
        u4 = torch.cat([u4, x4], dim=1)
        u4 = self.up4(u4)

        u3 = self.upConv3(u4)
        u3 = torch.cat([u3, x3], dim=1)
        u3 = self.up3(u3)

        u2 = self.upConv2(u3)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.up2(u2)

        u1 = self.upConv1(u2)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.up1(u1)

        out = self.out(u1)
        return torch.sigmoid(out)
    
class EdgeAwareDepthLoss(nn.Module):
    def __init__(self, alpha=0.5):
        # alpha: 경계선을 얼마나 중요하게 채점할지 결정하는 가중치
        super(EdgeAwareDepthLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.alpha = alpha

    def forward(self, pred, target):
        # 깊이 차이
        l1 = self.l1_loss(pred, target)

        # X축과 Y축으로 인접한 픽셀간의 차이를 구함 
        # 예측한 깊이 지도의 가로/세로 절벽
        pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])

        # 정답지의 가로/세로 절벽
        target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])

        # 절벽 오차 구하기
        edge_loss = torch.mean(torch.abs(pred_dx - target_dx)) + torch.mean(torch.abs(pred_dy - target_dy))

        # 4. 최종 점수 = 깊이 차이 + (가중치 * 절벽 오차)
        final_loss = l1 + (self.alpha * edge_loss)

        return final_loss
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
num_epochs = 30 

fold_results = []

history = {'train_loss': [], 'val_loss': []}

print("학습 시작!")

for fold, (train_idx, val_idx) in enumerate(kfold.split(all_rgb_paths)):
    train_imgs = [all_rgb_paths[i] for i in train_idx]
    train_tdisps = [all_tdisp_paths[i] for i in train_idx]

    val_imgs = [all_rgb_paths[i] for i in val_idx]
    val_tdisps = [all_tdisp_paths[i] for i in val_idx]

    train_dataset = KFoldPotholeDataset(train_imgs, train_tdisps, transform=train_transform)
    val_dataset = KFoldPotholeDataset(val_imgs, val_tdisps, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = PotholeUNet().to(device)
    criterion = EdgeAwareDepthLoss(alpha=0.2) 
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')

    fold_train_losses = []
    fold_val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for images, depths in train_loader:
            images, depths = images.to(device), depths.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, depths)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, depths in val_loader:
                images, depths = images.to(device), depths.to(device)
                outputs = model(images)
                loss = criterion(outputs, depths)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        fold_train_losses.append(avg_train_loss)
        fold_val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            #가중치 파일 저장 원하는 경로 입력
            torch.save(model.state_dict(), f" ")

        if (epoch + 1) % 5 == 0:
            print(f"Fold {fold+1} | Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    fold_results.append(best_val_loss)

    history['train_loss'].append(fold_train_losses)
    history['val_loss'].append(fold_val_losses)

print("학습 종료!")

#Training / Validation Loss 그래프 시각화
plt.figure(figsize=(15, 10))
plt.suptitle('K-Fold Training & Validation Loss', fontsize=16, fontweight='bold')

for i in range(5):
    plt.subplot(2, 3, i + 1)
    
    plt.plot(history['train_loss'][i], label='Train Loss', color='blue', linewidth=2)
    plt.plot(history['val_loss'][i], label='Validation Loss', color='orange', linewidth=2)
    
    plt.title(f'Fold {i + 1}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()