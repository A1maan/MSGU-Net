import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from metrics_isic2018 import *

# Load the model and test data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MSGUNet(in_channels=3, out_channels=1, base_channels=32).to(device)

# Load pretrained weights
pretrained_weights_path = "weights/best_model_isic2018_4.pth"
if os.path.exists(pretrained_weights_path):
    print(f"Loading weights from: {pretrained_weights_path}")
    model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
    print("✅ Weights loaded!")
else:
    print(f"❌ Weights not found at: {pretrained_weights_path}")
    exit()

model.eval()

# Get a batch for analysis
imgs, masks = next(iter(test_loader_2018))
imgs, masks = imgs.to(device), masks.to(device)

with torch.no_grad():
    outputs = model(imgs)
    predictions = torch.sigmoid(outputs)

print("=== DEBUGGING ANALYSIS ===")
print(f"Batch size: {imgs.shape[0]}")
print(f"Image shape: {imgs.shape}")
print(f"Mask shape: {masks.shape}")
print(f"Predictions shape: {predictions.shape}")

# Analyze first sample in detail
sample_idx = 0
pred_sample = predictions[sample_idx]
mask_sample = masks[sample_idx]

print(f"\n=== SAMPLE {sample_idx} ANALYSIS ===")
print(f"Prediction range: [{pred_sample.min():.4f}, {pred_sample.max():.4f}]")
print(f"Prediction mean: {pred_sample.mean():.4f}")
print(f"Mask range: [{mask_sample.min():.4f}, {mask_sample.max():.4f}]")
print(f"Mask mean: {mask_sample.mean():.4f}")

# Check different thresholds
thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
print(f"\n=== THRESHOLD ANALYSIS ===")
for thresh in thresholds:
    metrics = calculate_metrics(pred_sample.unsqueeze(0), mask_sample.unsqueeze(0), threshold=thresh)
    print(f"Threshold {thresh:.1f}: IoU={metrics['mIoU']:.2f}%, DSC={metrics['DSC']:.2f}%")

# Check mask statistics
mask_binary = (mask_sample > 0.5).float()
pred_binary_05 = (pred_sample > 0.5).float()

print(f"\n=== MASK STATISTICS ===")
print(f"Ground truth pixels (white): {mask_binary.sum().item()}")
print(f"Total pixels: {mask_binary.numel()}")
print(f"Lesion coverage: {(mask_binary.sum() / mask_binary.numel() * 100):.2f}%")

print(f"\n=== PREDICTION STATISTICS ===")
print(f"Predicted pixels (>0.5): {pred_binary_05.sum().item()}")
print(f"Prediction coverage: {(pred_binary_05.sum() / pred_binary_05.numel() * 100):.2f}%")

# Check if predictions are too conservative or aggressive
intersection = (mask_binary * pred_binary_05).sum().item()
print(f"Intersection pixels: {intersection}")
print(f"Recall (sensitivity): {intersection / max(mask_binary.sum().item(), 1) * 100:.2f}%")
print(f"Precision: {intersection / max(pred_binary_05.sum().item(), 1) * 100:.2f}%")

# Analyze entire test dataset statistics
print(f"\n=== FULL DATASET ANALYSIS ===")
all_pred_means = []
all_mask_means = []
all_pred_maxs = []

with torch.no_grad():
    for batch_imgs, batch_masks in test_loader_2018:
        batch_imgs, batch_masks = batch_imgs.to(device), batch_masks.to(device)
        batch_outputs = model(batch_imgs)
        batch_preds = torch.sigmoid(batch_outputs)
        
        all_pred_means.extend(batch_preds.mean(dim=[1,2,3]).cpu().numpy())
        all_mask_means.extend(batch_masks.mean(dim=[1,2,3]).cpu().numpy())
        all_pred_maxs.extend(batch_preds.max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0].cpu().numpy())

print(f"Average prediction confidence: {np.mean(all_pred_means):.4f}")
print(f"Average mask coverage: {np.mean(all_mask_means):.4f}")
print(f"Average max prediction: {np.mean(all_pred_maxs):.4f}")
print(f"Min max prediction: {np.min(all_pred_maxs):.4f}")

# Check if model is actually making predictions
if np.mean(all_pred_maxs) < 0.1:
    print("⚠️  WARNING: Model predictions are very low - model might not be trained properly!")
elif np.mean(all_pred_means) > 0.8:
    print("⚠️  WARNING: Model predictions are very high - might be predicting everything as lesion!")
else:
    print("✅ Model prediction ranges seem reasonable")

print(f"\n=== RECOMMENDATIONS ===")
if np.mean(all_pred_means) < 0.1:
    print("1. Model predictions are too low - check if model is properly trained")
    print("2. Try lower thresholds (0.1-0.3) for binarization")
elif np.mean(all_pred_means) > 0.7:
    print("1. Model predictions are too high - might be overfitting")
    print("2. Try higher thresholds (0.7-0.9) for binarization")
else:
    print("1. Model predictions seem in reasonable range")
    print("2. Consider checking data preprocessing consistency between training and testing")