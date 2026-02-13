import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from dataset import OccludedDataset, get_transforms
from models.pixelrnn import PixelRNN
from utils import save_output
import os
import matplotlib.pyplot as plt

try:
    import piq
    use_ssim = True
except ImportError:
    print("piq not installed, using only MSE loss")
    use_ssim = False

# ==========================
# Hyperparameters
# ==========================
img_size = 64
batch_size = 4
lr = 5e-4
num_epochs = 20
val_split = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================
# Dataset Split (Train/Val)
# ==========================
full_dataset = OccludedDataset(
    occluded_dir="data/train/occluded_images",
    original_dir="data/train/original_images",
    transform=get_transforms(img_size)
)

val_size = int(len(full_dataset) * val_split)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

# ==========================
# Model, Loss, Optimizer
# ==========================
model = PixelRNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

os.makedirs("outputs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

train_losses, val_losses = [], []

# ==========================
# Training Loop
# ==========================
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0

    for i, (occluded, original) in enumerate(train_loader):
        occluded, original = occluded.to(device), original.to(device)
        optimizer.zero_grad()

        outputs = model(occluded)
        outputs_clamped = torch.clamp(outputs, 0.0, 1.0)
        original_clamped = torch.clamp(original, 0.0, 1.0)

        loss = criterion(outputs_clamped, original_clamped)
        if use_ssim:
            ssim_loss = 1 - piq.ssim(outputs_clamped, original_clamped, data_range=1.0)
            loss = loss + 0.5 * ssim_loss

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ==========================
    # Validation Phase
    # ==========================
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for occluded, original in val_loader:
            occluded, original = occluded.to(device), original.to(device)
            outputs = model(occluded)
            outputs_clamped = torch.clamp(outputs, 0.0, 1.0)
            original_clamped = torch.clamp(original, 0.0, 1.0)

            v_loss = criterion(outputs_clamped, original_clamped)
            if use_ssim:
                v_ssim_loss = 1 - piq.ssim(outputs_clamped, original_clamped, data_range=1.0)
                v_loss = v_loss + 0.5 * v_ssim_loss
            val_loss += v_loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch [{epoch}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    save_output(occluded, outputs, folder="outputs", epoch=epoch)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
    }, f"checkpoints/pixelrnn_epoch{epoch}.pth")

print("Training completed!")

# ==========================
# Plot Loss Curves
# ==========================
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('PixelRNN Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("outputs/loss_curve.png")
plt.show()
