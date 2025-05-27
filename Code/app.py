import torch
import torch.nn as nn
import gradio as gr
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
from torchmodel import TLUNet, get_hw, ImageMaskDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class SegmentationApp:
    def __init__(self, model_path: str, test_dir_images: str, test_dir_masks: str, patch_size: int = 256, version: int = 0):
        self.patch_size = patch_size
        self.version = version
        self.device = torch.device(get_hw())
        self.test_dir_images = test_dir_images
        self.test_dir_masks = test_dir_masks
        
        # Load model
        input_shape = (patch_size, patch_size, 3)
        self.model = TLUNet(input_shape, patch_size).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Load test dataset
        target_size = (patch_size, patch_size)
        self.test_dataset = ImageMaskDataset(test_dir_images, test_dir_masks, target_size)
        
        # Get list of image files for dropdown
        self.image_files = sorted([f for f in os.listdir(test_dir_images) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    def calculate_iou(self, predictions: np.ndarray, ground_truth: np.ndarray, threshold: float = 0.5) -> float:
        """Calculate Intersection over Union"""
        pred_mask = (predictions > threshold).astype(np.float32)
        true_mask = (ground_truth > threshold).astype(np.float32)
        
        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def get_test_sample(self, image_index: int):
        """Get a test sample by index"""
        if image_index >= len(self.test_dataset):
            return None, None
        
        image_tensor, mask_tensor = self.test_dataset[image_index]
        return image_tensor, mask_tensor
    
    def predict_sample(self, image_index: int, threshold: float = 0.5):
        """Make prediction on a test sample"""
        try:
            # Get sample from dataset
            image_tensor, mask_tensor = self.get_test_sample(image_index)
            if image_tensor is None:
                return None, "Invalid image index"
            
            # Add batch dimension and move to device
            input_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                prediction = output.cpu().numpy()[0, 0]  # Remove batch and channel dims
            
            # Convert tensors to numpy for visualization
            image_np = image_tensor.permute(1, 2, 0).numpy()
            mask_np = mask_tensor.numpy()[0]  # Remove channel dim
            
            # Apply threshold to prediction
            binary_mask = (prediction > threshold).astype(np.uint8) * 255
            
            # Calculate metrics
            iou_score = self.calculate_iou(prediction.reshape(1, -1), mask_np.reshape(1, -1), threshold)
            pixel_accuracy = np.mean((binary_mask > 127) == (mask_np > 0.5))
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Top row
            axes[0, 0].imshow(image_np)
            axes[0, 0].set_title(f"Original Image\n{self.image_files[image_index]}")
            axes[0, 0].axis("off")
            
            axes[0, 1].imshow(mask_np, cmap="gray")
            axes[0, 1].set_title("Ground Truth")
            axes[0, 1].axis("off")
            
            axes[0, 2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
            axes[0, 2].set_title("Raw Prediction")
            axes[0, 2].axis("off")
            
            # Bottom row
            axes[1, 0].imshow(mask_np, cmap="gray")
            axes[1, 0].set_title("Ground Truth (Binary)")
            axes[1, 0].axis("off")
            
            axes[1, 1].imshow(binary_mask, cmap="gray")
            axes[1, 1].set_title(f"Prediction (Binary)\nThreshold: {threshold:.2f}")
            axes[1, 1].axis("off")
            
            # Overlay comparison
            overlay = np.zeros((*prediction.shape, 3))
            overlay[:, :, 0] = mask_np  # Ground truth in red
            overlay[:, :, 1] = binary_mask / 255.0  # Prediction in green
            axes[1, 2].imshow(overlay)
            axes[1, 2].set_title("Overlay (GT=Red, Pred=Green)")
            axes[1, 2].axis("off")
            
            plt.tight_layout()
            
            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            plt.close()
            
            result_image = Image.open(buf)
            
            # Statistics
            confidence_mean = float(np.mean(prediction))
            confidence_max = float(np.max(prediction))
            mask_coverage = float(np.sum(binary_mask > 0) / binary_mask.size * 100)
            
            stats_text = f"""
            **Sample: {self.image_files[image_index]}**
            
            **Evaluation Metrics:**
            - IoU Score: {iou_score:.4f}
            - Pixel Accuracy: {pixel_accuracy:.4f}
            
            **Prediction Statistics:**
            - Mean Confidence: {confidence_mean:.4f}
            - Max Confidence: {confidence_max:.4f}
            - Mask Coverage: {mask_coverage:.2f}%
            - Threshold: {threshold:.2f}
            """
            
            return result_image, stats_text
            
        except Exception as e:
            return None, f"Error during prediction: {str(e)}"
    
    def evaluate_batch(self, start_index: int, batch_size: int, threshold: float = 0.5):
        """Evaluate a batch of samples"""
        try:
            results = []
            total_samples = min(batch_size, len(self.test_dataset) - start_index)
            
            for i in range(total_samples):
                idx = start_index + i
                image_tensor, mask_tensor = self.get_test_sample(idx)
                
                if image_tensor is None:
                    continue
                
                # Make prediction
                input_tensor = image_tensor.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.model(input_tensor)
                    prediction = output.cpu().numpy()[0, 0]
                
                mask_np = mask_tensor.numpy()[0]
                
                # Calculate metrics
                iou_score = self.calculate_iou(prediction.reshape(1, -1), mask_np.reshape(1, -1), threshold)
                binary_pred = (prediction > threshold).astype(np.float32)
                pixel_accuracy = np.mean(binary_pred == (mask_np > 0.5))
                
                results.append({
                    "filename": self.image_files[idx],
                    "iou": iou_score,
                    "pixel_accuracy": pixel_accuracy,
                    "mean_confidence": np.mean(prediction)
                })
            
            # Calculate averages
            if results:
                avg_iou = np.mean([r["iou"] for r in results])
                avg_pixel_acc = np.mean([r["pixel_accuracy"] for r in results])
                avg_confidence = np.mean([r["mean_confidence"] for r in results])
                
                summary = f"""
                **Batch Evaluation Results**
                **Samples: {start_index} to {start_index + total_samples - 1}**
                
                **Average Metrics:**
                - Average IoU: {avg_iou:.4f}
                - Average Pixel Accuracy: {avg_pixel_acc:.4f}
                - Average Confidence: {avg_confidence:.4f}
                - Threshold: {threshold:.2f}
                
                **Individual Results:**
                """
                
                for r in results:
                    summary += f"\n- {r['filename']}: IoU={r['iou']:.3f}, Acc={r['pixel_accuracy']:.3f}"
                
                return summary
            else:
                return "No valid samples found in the specified range."
                
        except Exception as e:
            return f"Error during batch evaluation: {str(e)}"


def create_gradio_interface(model_path: str, test_dir_images: str, test_dir_masks: str, 
                          patch_size: int = 256, version: int = 0):
    """Create Gradio interface"""
    app = SegmentationApp(model_path, test_dir_images, test_dir_masks, patch_size, version)
    
    with gr.Blocks(title="Image Segmentation Model - Test Dataset", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# Image Segmentation Model - Test Dataset Evaluation")
        gr.Markdown(f"Evaluating model on test dataset with {len(app.test_dataset)} samples.")
        
        with gr.Tab("Single Sample Prediction"):
            with gr.Row():
                with gr.Column():
                    image_dropdown = gr.Dropdown(
                        choices=[(f"{i}: {filename}", i) for i, filename in enumerate(app.image_files)],
                        label="Select Test Image",
                        value=0
                    )
                    threshold_slider = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                        label="Prediction Threshold"
                    )
                    predict_btn = gr.Button("Predict", variant="primary")
                    
                    # Navigation buttons
                    with gr.Row():
                        prev_btn = gr.Button("← Previous")
                        next_btn = gr.Button("Next →")
                
                with gr.Column():
                    output_image = gr.Image(label="Prediction Results")
                    stats_output = gr.Markdown()
            
            def update_dropdown(current_value, direction):
                if direction == "prev":
                    new_value = max(0, current_value - 1)
                else:  # next
                    new_value = min(len(app.image_files) - 1, current_value + 1)
                return new_value
            
            predict_btn.click(
                fn=app.predict_sample,
                inputs=[image_dropdown, threshold_slider],
                outputs=[output_image, stats_output]
            )
            
            prev_btn.click(
                fn=lambda x: update_dropdown(x, "prev"),
                inputs=[image_dropdown],
                outputs=[image_dropdown]
            )
            
            next_btn.click(
                fn=lambda x: update_dropdown(x, "next"),
                inputs=[image_dropdown],
                outputs=[image_dropdown]
            )
        
        with gr.Tab("Batch Evaluation"):
            with gr.Row():
                with gr.Column():
                    start_index = gr.Number(
                        value=0, minimum=0, maximum=len(app.test_dataset)-1,
                        label="Start Index", precision=0
                    )
                    batch_size = gr.Number(
                        value=10, minimum=1, maximum=50,
                        label="Batch Size", precision=0
                    )
                    threshold_slider_batch = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                        label="Prediction Threshold"
                    )
                    evaluate_btn = gr.Button("Evaluate Batch", variant="primary")
                
                with gr.Column():
                    batch_results = gr.Markdown()
            
            evaluate_btn.click(
                fn=app.evaluate_batch,
                inputs=[start_index, batch_size, threshold_slider_batch],
                outputs=[batch_results]
            )
        
        with gr.Tab("Dataset Info"):
            gr.Markdown(f"""
            **Model Configuration:**
            - Patch Size: {patch_size}x{patch_size}
            - Version: {version}
            - Device: {app.device}
            - Model Path: {model_path}
            
            **Dataset Information:**
            - Test Images Directory: {test_dir_images}
            - Test Masks Directory: {test_dir_masks}
            - Total Test Samples: {len(app.test_dataset)}
            
            **Available Images:**
            {chr(10).join([f"{i}: {filename}" for i, filename in enumerate(app.image_files[:20])])}
            {"..." if len(app.image_files) > 20 else ""}
            
            **Usage Instructions:**
            1. **Single Sample Tab**: Select an image from dropdown to see individual predictions
            2. **Batch Evaluation Tab**: Evaluate multiple samples at once
            3. Use navigation buttons (Previous/Next) to browse through samples
            4. Adjust threshold to control binary mask generation
            """)
    
    return interface


if __name__ == "__main__":
    # Configuration - Update these paths according to your setup
    patch_size = 256
    version = 0
    batch_size = 48
    
    # Paths
    path = f"Data/Data_{patch_size}_{version}"
    test_dir_images = os.path.join(path, "val", "images")
    test_dir_masks = os.path.join(path, "val", "masks")
    model_path = f"Models/model__{patch_size}_{batch_size}_{version}.pth"
    
    # Create and launch interface
    interface = create_gradio_interface(model_path, test_dir_images, test_dir_masks, patch_size, version)
    interface.launch(
        share=True,  # Set to False if you don't want to create a public link
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860
    )