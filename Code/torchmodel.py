import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import cv2
import os
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UpConv, self).__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up_conv(x)


class TLUNet(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], patch_size: int):
        super(TLUNet, self).__init__()
        self.patch_size = patch_size

        # Load pretrained VGG16 with explicit weights specification
        base_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Freeze VGG16 parameters
        for param in base_vgg.parameters():
            param.requires_grad = False

        # Extract and store intermediate dimensions
        self.dims = {
            'input': patch_size,
            'block1': patch_size // 2,    # 128
            'block2': patch_size // 4,    # 64
            'block3': patch_size // 8,    # 32
            'block4': patch_size // 16,   # 16
            'block5': patch_size // 32    # 8
        }

        # Encoder blocks from VGG16
        self.block1 = nn.Sequential(
            *list(base_vgg.features.children())[:5])   # 256->128
        self.block2 = nn.Sequential(
            *list(base_vgg.features.children())[5:10])  # 128->64
        self.block3 = nn.Sequential(
            *list(base_vgg.features.children())[10:17])  # 64->32
        self.block4 = nn.Sequential(
            *list(base_vgg.features.children())[17:24])  # 32->16
        self.block5 = nn.Sequential(
            *list(base_vgg.features.children())[24:31])  # 16->8

        # Decoder path with reusable components
        self.upconv1 = UpConv(512, 512)  # 8->16
        self.conv6 = DoubleConv(1024, 512)  # After concatenation

        self.upconv2 = UpConv(512, 256)  # 16->32
        self.conv7 = DoubleConv(512, 256)

        self.upconv3 = UpConv(256, 128)  # 32->64
        self.conv8 = DoubleConv(256, 128)

        self.upconv4 = UpConv(128, 64)  # 64->128
        self.conv9 = DoubleConv(128, 64)

        # Final upsampling and convolution
        self.upconv_final = UpConv(64, 32)  # 128->256
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder path with skip connections
        e1 = self.block1(x)
        e2 = self.block2(e1)
        e3 = self.block3(e2)
        e4 = self.block4(e3)
        e5 = self.block5(e4)

        # Decoder path with skip connections
        d1 = self.upconv1(e5)
        d1 = torch.cat([d1, e4], dim=1)
        d1 = self.conv6(d1)

        d2 = self.upconv2(d1)
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.conv7(d2)

        d3 = self.upconv3(d2)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.conv8(d3)

        d4 = self.upconv4(d3)
        d4 = torch.cat([d4, e1], dim=1)
        d4 = self.conv9(d4)

        # Final upsampling and convolution
        out = self.upconv_final(d4)
        out = self.final_conv(out)

        return out


class ImageMaskDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, target_size: Tuple[int, int]):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.image_paths = sorted(
            [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, fname)
                                 for fname in os.listdir(mask_dir)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = image / 255.0
        image = torch.FloatTensor(image).permute(
            2, 0, 1)  # Convert to CHW format

        # Load mask
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0
        mask = torch.FloatTensor(mask).unsqueeze(0)  # Add channel dimension

        return image, mask


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, gamma=2):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        intersection = torch.sum(y_pred * y_true)
        denominator = torch.sum(y_pred.pow(self.gamma)) + \
            torch.sum(y_true.pow(self.gamma))

        dice_score = (2.0 * intersection + self.smooth) / \
            (denominator + self.smooth)
        return 1 - dice_score


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                num_epochs: int,
                device: torch.device) -> dict:

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    history = {
        'train_loss': [],
        'val_loss': [],
        'batch_losses': []  # Track individual batch losses
    }

    total_batches = len(train_loader)
    print(f"Training on {total_batches} batches per epoch")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Progress tracking variables
        batch_losses = []
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            # Record batch loss
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            train_loss += batch_loss

            # Print progress every 10% of batches
            if (batch_idx + 1) % max(1, total_batches // 10) == 0:
                current_loss = train_loss / (batch_idx + 1)
                progress = (batch_idx + 1) / total_batches * 100
                print(f"Batch {batch_idx + 1}/{total_batches} [{progress:.1f}%] - "
                      f"Current Loss: {current_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch_idx, (images, masks) in enumerate(val_loader):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        # Calculate and record average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['batch_losses'].extend(batch_losses)

        # Print epoch summary
        print("\nEpoch Summary:")
        print(f"Average Train Loss: {avg_train_loss:.4f}")
        print(f"Average Val Loss: {avg_val_loss:.4f}")
        print(f"Best batch loss: {min(batch_losses):.4f}")
        print(f"Worst batch loss: {max(batch_losses):.4f}")
        print("-" * 60)

    return history


def test_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    criterion = nn.BCELoss()

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss

def get_hw():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    return device

def plot_training_history(history, save_path=None, show_plot=True):
    """
    Plot training metrics from model history.
    
    Args:
        history (dict): Dictionary containing 'train_loss', 'val_loss', and 'batch_losses'
        save_path (str, optional): Path to save the plot. Defaults to None.
        show_plot (bool, optional): Whether to display the plot. Defaults to True.
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Training History', fontsize=16)
    
    # Plot epoch-wise losses
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Top subplot: Training and validation loss per epoch
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Epoch-wise Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Bottom subplot: Batch losses
    batches = range(1, len(history['batch_losses']) + 1)
    ax2.plot(batches, history['batch_losses'], 'g-', alpha=0.5, label='Batch Loss')
    
    # Add moving average line for batch losses
    window_size = min(100, len(history['batch_losses']) // 10)  # Adaptive window size
    if window_size > 1:
        moving_avg = np.convolve(history['batch_losses'], 
                               np.ones(window_size)/window_size, 
                               mode='valid')
        ax2.plot(range(window_size, len(batches) + 1), 
                moving_avg, 
                'r-', 
                label=f'Moving Average (window={window_size})')
    
    ax2.set_title('Batch-wise Training Loss')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    ax2.legend()
    
    # Add summary statistics as text
    stats_text = (
        f"Final Training Loss: {history['train_loss'][-1]:.4f}\n"
        f"Final Validation Loss: {history['val_loss'][-1]:.4f}\n"
        f"Best Training Loss: {min(history['train_loss']):.4f}\n"
        f"Best Validation Loss: {min(history['val_loss']):.4f}\n"
        f"Best Batch Loss: {min(history['batch_losses']):.4f}"
    )
    fig.text(0.95, 0.05, stats_text, fontsize=10, ha='right', 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


# Usage example:
if __name__ == "__main__":
    hw = get_hw()
    # Parameters
    patch_size = 256
    version = 0
    batch_size = 48
    num_epochs = 8
    device = torch.device(hw)

    # Paths
    path = f"Data/Data_{patch_size}_{version}"
    train_dir_images = os.path.join(path, "train", "images")
    train_dir_masks = os.path.join(path, "train", "masks")
    val_dir_images = os.path.join(path, "val", "images")
    val_dir_masks = os.path.join(path, "val", "masks")
    test_dir_images = os.path.join(path, "test", "images")
    test_dir_masks = os.path.join(path, "test", "masks")

    # Create datasets and dataloaders
    target_size = (patch_size, patch_size)
    input_shape = (patch_size, patch_size, 3)

    train_dataset = ImageMaskDataset(
        train_dir_images, train_dir_masks, target_size)
    val_dataset = ImageMaskDataset(val_dir_images, val_dir_masks, target_size)
    test_dataset = ImageMaskDataset(
        test_dir_images, test_dir_masks, target_size)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Create and train model
    model = TLUNet(input_shape, patch_size).to(device)
    history = train_model(model, train_loader, val_loader, num_epochs, device)

    # Test model
    test_loss = test_model(model, test_loader, device)

    # Save model
    model_name = f"model__{patch_size}_{batch_size}_{version}.pth"
    torch.save(model.state_dict(), os.path.join("Models", model_name))
    plot_training_history(
    history,
    save_path=f'training_history_{patch_size}_{batch_size}_{version}.png'
)
