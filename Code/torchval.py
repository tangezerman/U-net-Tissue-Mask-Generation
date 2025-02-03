import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
from typing import Tuple, Dict
import matplotlib.pyplot as plt
from torchmodel import DiceLoss, ImageMaskDataset, TLUNet, DiceLoss, get_hw


def calculate_iou(predictions: np.ndarray, ground_truth: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculate Intersection over Union
    """
    # Convert to binary masks
    pred_mask = (predictions > threshold).astype(np.float32)
    true_mask = (ground_truth > threshold).astype(np.float32)
    
    # Calculate intersection and union
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    return intersection / union


def evaluate_model(model: nn.Module,
                   test_loader: DataLoader,
                   device: torch.device) -> Dict[str, float]:
    """
    Evaluate model performance using multiple metrics
    """
    model.eval()
    total_dice = 0.0
    total_bce = 0.0
    total_iou = 0.0
    total_batches = len(test_loader)

    dice_criterion = DiceLoss()
    bce_criterion = nn.BCELoss()

    predictions = []
    ground_truth = []

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            # Calculate losses
            dice_loss = dice_criterion(outputs, masks)
            bce_loss = bce_criterion(outputs, masks)

            # Store predictions and ground truth for additional analysis
            batch_preds = outputs.cpu().numpy()
            batch_truth = masks.cpu().numpy()
            
            # Calculate IoU for current batch
            batch_iou = calculate_iou(batch_preds, batch_truth)
            total_iou += batch_iou

            total_dice += (1 - dice_loss.item())  # Convert loss to score
            total_bce += bce_loss.item()

            predictions.extend(batch_preds)
            ground_truth.extend(batch_truth)

    # Calculate average metrics
    avg_dice_score = total_dice / total_batches
    avg_bce_loss = total_bce / total_batches
    avg_iou = total_iou / total_batches

    # Calculate additional metrics
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Calculate pixel-wise accuracy
    threshold = 0.5
    binary_preds = (predictions > threshold).astype(np.float32)
    binary_truth = (ground_truth > threshold).astype(np.float32)
    pixel_accuracy = np.mean(binary_preds == binary_truth)

    metrics = {
        'dice_score': avg_dice_score,
        'bce_loss': avg_bce_loss,
        'pixel_accuracy': pixel_accuracy,
        'iou_score': avg_iou
    }

    return metrics


def plot_sample_predictions(model: nn.Module,
                            test_loader: DataLoader,
                            device: torch.device,
                            num_samples: int = 5,
                            save_path: str = None):
    """
    Plot sample predictions alongside ground truth
    """
    model.eval()

    with torch.no_grad():
        # Get a batch of images
        images, masks = next(iter(test_loader))
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)

        # Convert tensors to numpy arrays
        images = images.cpu().numpy()
        masks = masks.cpu().numpy()
        predictions = outputs.cpu().numpy()

        # Plot samples
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
        fig.suptitle('Sample Predictions', fontsize=16)

        for i in range(min(num_samples, len(images))):
            # Original image
            axes[i, 0].imshow(np.transpose(images[i], (1, 2, 0)))
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')

            # Ground truth mask
            axes[i, 1].imshow(masks[i, 0], cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            # Predicted mask
            axes[i, 2].imshow(predictions[i, 0], cmap='gray')
            axes[i, 2].set_title(f'Prediction\nIoU: {calculate_iou(predictions[i:i+1], masks[i:i+1]):.3f}')
            axes[i, 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sample predictions saved to {save_path}")

        plt.show()


if __name__ == "__main__":
    # Parameters (make sure these match your training parameters)
    patch_size = 256
    version = 0
    batch_size = 48
    device = get_hw()
    torch.device(device)

    # Paths
    path = f"Data/Data_{patch_size}_{version}"
    test_dir_images = os.path.join(path, "val", "images")
    test_dir_masks = os.path.join(path, "val", "masks")
    model_path = os.path.join(
        "Models", f"model__{patch_size}_{batch_size}_{version}.pth")

    # Create test dataset and dataloader
    target_size = (patch_size, patch_size)
    input_shape = (patch_size, patch_size, 3)

    test_dataset = ImageMaskDataset(
        test_dir_images, test_dir_masks, target_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Load model
    model = TLUNet(input_shape, patch_size).to(device)
    model.load_state_dict(torch.load(model_path))

    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, test_loader, device)

    # Print metrics
    print("\nModel Performance Metrics:")
    print("-" * 30)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # Plot sample predictions
    print("\nGenerating sample predictions...")
    plot_sample_predictions(
        model,
        test_loader,
        device,
        num_samples=5,
        save_path=f'sample_predictions_{patch_size}_{batch_size}_{version}.png'
    )