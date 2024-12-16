import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from dataset import InpaintingDataset
import random
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np


class AutoencoderWithSkip(nn.Module):
    def __init__(self):
        super(AutoencoderWithSkip, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU())

        self.bottleneck = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.ReLU())

        self.dec3 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU())
        self.final_layer = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        x1 = self.enc1(x)  # Output: [B, 64, 64, 64]
        x2 = self.enc2(x1)  # Output: [B, 128, 32, 32]
        x3 = self.enc3(x2)  # Output: [B, 256, 16, 16]

        bottleneck = self.bottleneck(x3)  # [B, 512, 16, 16]

        d3 = self.dec3(bottleneck)  # [B, 256, 32, 32]
        d3 = d3 + nn.functional.interpolate(x3, size=d3.shape[-2:], mode="bilinear", align_corners=False)

        d2 = self.dec2(d3)  # [B, 128, 64, 64]
        d2 = d2 + nn.functional.interpolate(x2, size=d2.shape[-2:], mode="bilinear", align_corners=False)

        d1 = self.dec1(d2)  # [B, 64, 128, 128]
        d1 = d1 + nn.functional.interpolate(x1, size=d1.shape[-2:], mode="bilinear", align_corners=False)

        output = torch.sigmoid(self.final_layer(d1)) 
        return output


def split_dataset(dataset, val_ratio=0.2):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    split_idx = int(val_ratio * dataset_size)
    return indices[split_idx:], indices[:split_idx]


def calculate_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = (pred + target).clamp(0, 1).sum()
    return (intersection / union).item() if union > 0 else 0.0


def train():
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    dataset = InpaintingDataset("train_image_paths.txt", transform=transform, mask_dir="masks")

    train_indices, val_indices = split_dataset(dataset, val_ratio=0.2)
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoencoderWithSkip().to(device)

    l1_loss = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    epochs =30
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for masked_img, mask, gt_img in train_loader:
            masked_img, gt_img = masked_img.to(device), gt_img.to(device)

            optimizer.zero_grad()
            output = model(masked_img)
            loss = l1_loss(output, gt_img)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        iou_scores = []
        with torch.no_grad():
            for masked_img, mask, gt_img in val_loader:
                masked_img, gt_img = masked_img.to(device), gt_img.to(device)
                output = model(masked_img)
                val_loss += l1_loss(output, gt_img).item()

                iou = calculate_iou(output, gt_img)
                iou_scores.append(iou)

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_iou = sum(iou_scores) / len(iou_scores)

        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | IoU: {avg_iou:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "autoencoder_skip_model.pth")
    print("Model saved to autoencoder_skip_model.pth")


if __name__ == "__main__":
    train()
