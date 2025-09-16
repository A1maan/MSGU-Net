
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from model.MSGUNet import MSGUNet
import os
from PIL import Image


class MultiSegmentationDataset(Dataset):
    def __init__(self, base_dirs, split, transform=None, mask_transform=None):
        self.samples = []
        self.transform = transform
        self.mask_transform = mask_transform

        for base_dir in base_dirs:
            img_dir = os.path.join(base_dir, split, "images")
            mask_dir = os.path.join(base_dir, split, "masks")
            images = sorted(os.listdir(img_dir))
            masks  = sorted(os.listdir(mask_dir))
            for i, m in zip(images, masks):
                self.samples.append((os.path.join(img_dir, i), os.path.join(mask_dir, m)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            img = self.transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return img, mask


base_dir_2017 = "/home/aminu_yusuf/msgunet/datasets/ISIC2017"
base_dir_2018 = "/home/aminu_yusuf/msgunet/datasets/ISIC2018"
base_dirs = [base_dir_2017, base_dir_2018]

transform_img = T.Compose([
    T.Resize((256, 256)),
    T.RandomHorizontalFlip(p=0.5),   # 50% chance to flip horizontally
    T.RandomVerticalFlip(p=0.5),     # 50% chance to flip vertically
    T.RandomRotation(degrees=15),    # rotate randomly between -15° to +15°
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

transform_mask = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])  

train_ds = MultiSegmentationDataset(base_dirs, "train",
                                    transform=transform_img,
                                    mask_transform=transform_mask)
val_ds   = MultiSegmentationDataset(base_dirs, "val",
                                    transform=transform_img,
                                    mask_transform=transform_mask)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False)

print("Combined Train size:", len(train_ds))
print("Combined Val size:", len(val_ds))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MSGUNet(in_channels=3, out_channels=1, base_channels=32).to(device)

criterion = nn.BCEWithLogitsLoss()   # binary segmentation
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-5
)


def train_epoch(loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

def eval_epoch(loader, model, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


n_epochs = 100
train_losses = []

for epoch in range(n_epochs):
    train_loss = train_epoch(train_loader, model, criterion, optimizer, device)
    train_losses.append(train_loss)
    print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f}")


plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


model.eval()
imgs, masks = next(iter(train_loader))
imgs, masks = imgs.to(device), masks.to(device)
with torch.no_grad():
    preds = torch.sigmoid(model(imgs))

i = 0
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title("Image")
plt.imshow(np.transpose(imgs[i].cpu().numpy(), (1,2,0)))

plt.subplot(1,3,2)
plt.title("Mask")
plt.imshow(masks[i,0].cpu().numpy(), cmap="gray")

plt.subplot(1,3,3)
plt.title("Prediction")
plt.imshow(preds[i,0].cpu().numpy() > 0.5, cmap="gray")
plt.show()


