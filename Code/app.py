import torch
import gradio as gr
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
from torchmodel import TLUNet, get_hw, ImageMaskDataset
import torchvision.transforms as transforms


class SegmentationApp:
    def __init__(self, model_path: str, base_data_path: str, patch_size: int = 256, version: int = 0):
        self.patch_size = patch_size
        self.version = version
        self.device = torch.device(get_hw())
        self.base_data_path = base_data_path

        # Load model
        input_shape = (patch_size, patch_size, 3)
        self.model = TLUNet(input_shape, patch_size).to(self.device)
        self.model.load_state_dict(torch.load(
            model_path, map_location=self.device))
        self.model.eval()

        # Initialize split directories
        self.splits = ["train", "test", "val"]
        self.split_data = {}

        # Load datasets for each split
        target_size = (patch_size, patch_size)
        for split in self.splits:
            images_dir = os.path.join(base_data_path, split, "images")
            masks_dir = os.path.join(base_data_path, split, "masks")

            if os.path.exists(images_dir) and os.path.exists(masks_dir):
                dataset = ImageMaskDataset(images_dir, masks_dir, target_size)
                image_files = sorted([f for f in os.listdir(images_dir)
                                      if f.lower().endswith((".png", ".jpg", ".jpeg"))])
                self.split_data[split] = {
                    "dataset": dataset,
                    "image_files": image_files,
                    "images_dir": images_dir,
                    "masks_dir": masks_dir
                }
            else:
                self.split_data[split] = None

        # Set initial split
        self.current_split = "val"
        self.update_current_split("val")

        # Initialize transform for custom images
        self.transform = transforms.Compose([
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
        ])

    def update_current_split(self, split: str):
        """Update the current split and associated data"""
        self.current_split = split
        if self.split_data[split] is not None:
            self.current_dataset = self.split_data[split]["dataset"]
            self.image_files = self.split_data[split]["image_files"]
            self.current_images_dir = self.split_data[split]["images_dir"]
            self.current_masks_dir = self.split_data[split]["masks_dir"]
            return True
        return False

    def get_image_choices(self, split: str):
        """Get image choices for the specified split"""
        if split in self.split_data and self.split_data[split] is not None:
            image_files = self.split_data[split]["image_files"]
            return [(f"{i}: {filename}", i) for i, filename in enumerate(image_files)]
        return []

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
        if image_index >= len(self.current_dataset):
            return None, None

        image_tensor, mask_tensor = self.current_dataset[image_index]
        return image_tensor, mask_tensor

    def predict_custom_image(self, uploaded_image, threshold: float = 0.5):
        """Make prediction on a custom uploaded image"""
        try:
            if uploaded_image is None:
                return None, "Please upload an image first."

            # Convert PIL Image to tensor
            if isinstance(uploaded_image, str):
                image_pil = Image.open(uploaded_image).convert("RGB")
            else:
                image_pil = uploaded_image.convert("RGB")

            # Apply transforms
            image_tensor = self.transform(image_pil)

            # Add batch dimension and move to device
            input_tensor = image_tensor.unsqueeze(0).to(self.device)

            # Make prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                # Remove batch and channel dims
                prediction = output.cpu().numpy()[0, 0]

            # Convert tensor to numpy for visualization
            image_np = image_tensor.permute(1, 2, 0).numpy()

            # Apply threshold to prediction
            binary_mask = (prediction > threshold).astype(np.uint8) * 255

            # Create visualization
            fig = plt.figure(figsize=(12, 8))

            # Original image
            ax1 = plt.subplot(2, 2, 1)
            plt.imshow(image_np)
            plt.title("Original Image")
            ax1.set_xticks([])
            ax1.set_yticks([])
            for spine in ax1.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(2)

            # Raw prediction
            ax2 = plt.subplot(2, 2, 2)
            plt.imshow(prediction, cmap="gray", vmin=0, vmax=1)
            plt.title("Raw Prediction")
            ax2.set_xticks([])
            ax2.set_yticks([])
            for spine in ax2.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(2)

            # Binary prediction
            ax3 = plt.subplot(2, 2, 3)
            plt.imshow(binary_mask, cmap="gray")
            plt.title(f"Binary Prediction\nThreshold: {threshold:.2f}")
            ax3.set_xticks([])
            ax3.set_yticks([])
            for spine in ax3.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(2)

            # Overlay
            ax4 = plt.subplot(2, 2, 4)
            overlay = image_np.copy()
            # Create mask for predicted tissue areas
            tissue_mask = binary_mask > 127
            # Show prediction as semi-transparent white overlay
            overlay[tissue_mask] = overlay[tissue_mask] * 0.5 + \
                np.array([1, 1, 1]) * 0.3  # 50% original + 50% white
            plt.imshow(overlay)
            plt.title("Overlay")
            ax4.set_xticks([])
            ax4.set_yticks([])
            for spine in ax4.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(2)

            plt.tight_layout()

            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            plt.close()

            result_image = Image.open(buf)

            # Statistics
            tissue_percentage = (binary_mask > 127).sum(
            ) / (binary_mask.shape[0] * binary_mask.shape[1]) * 100
            mean_confidence = np.mean(prediction)

            stats_text = f"""
            **Custom Image Prediction Results**
            
            - Tissue Percentage: {tissue_percentage:.2f}%
            - Mean Prediction Confidence: {mean_confidence:.4f}
            - Threshold Used: {threshold:.2f}
            - Image Size: {image_pil.size[0]} x {image_pil.size[1]} (resized to {self.patch_size} x {self.patch_size})
            """

            return result_image, stats_text

        except Exception as e:
            return None, f"Error during prediction: {str(e)}"

    def predict_sample(self, split: str, image_index: int, threshold: float = 0.5):
        """Make prediction on a sample from the specified split"""
        try:
            # Update current split if needed
            if split != self.current_split:
                if not self.update_current_split(split):
                    return None, f"Split '{split}' not available"

            # Get sample from dataset
            image_tensor, mask_tensor = self.get_test_sample(image_index)
            if image_tensor is None:
                return None, "Invalid image index"

            # Add batch dimension and move to device
            input_tensor = image_tensor.unsqueeze(0).to(self.device)

            # Make prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                # Remove batch and channel dims
                prediction = output.cpu().numpy()[0, 0]

            # Convert tensors to numpy for visualization
            image_np = image_tensor.permute(1, 2, 0).numpy()
            mask_np = mask_tensor.numpy()[0]  # Remove channel dim

            # Apply threshold to prediction
            binary_mask = (prediction > threshold).astype(np.uint8) * 255

            # Calculate metrics
            iou_score = self.calculate_iou(prediction.reshape(
                1, -1), mask_np.reshape(1, -1), threshold)
            pixel_accuracy = np.mean((binary_mask > 127) == (mask_np > 0.5))

            # Create visualization
            fig = plt.figure(figsize=(12, 14))

            # Create 2x2 grid for the main images
            ax1 = plt.subplot(3, 2, 1)
            plt.imshow(image_np)
            plt.title(f"Original Image\n{self.image_files[image_index]}")
            ax1.set_xticks([])
            ax1.set_yticks([])
            for spine in ax1.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(2)

            ax2 = plt.subplot(3, 2, 2)
            plt.imshow(mask_np, cmap="gray")
            plt.title("Ground Truth")
            ax2.set_xticks([])
            ax2.set_yticks([])
            for spine in ax2.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(2)

            ax3 = plt.subplot(3, 2, 3)
            plt.imshow(prediction, cmap="gray", vmin=0, vmax=1)
            plt.title("Raw Prediction")
            ax3.set_xticks([])
            ax3.set_yticks([])
            for spine in ax3.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(2)

            ax4 = plt.subplot(3, 2, 4)
            plt.imshow(binary_mask, cmap="gray")
            plt.title(f"Prediction (Binary)\nThreshold: {threshold:.2f}")
            ax4.set_xticks([])
            ax4.set_yticks([])
            for spine in ax4.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(2)

            # Create overlay image in the center of the bottom row
            ax5 = plt.subplot(3, 1, 3)
            overlay = np.zeros((prediction.shape[0], prediction.shape[1], 3))
            overlay[:, :, 0] = mask_np  # Ground truth in red
            overlay[:, :, 1] = (binary_mask > 127).astype(
                np.float32)  # Prediction in green
            plt.imshow(overlay)
            plt.title("Overlay Comparison (GT=Red, Pred=Green)",
                      fontsize=12, weight="bold")
            ax5.set_xticks([])
            ax5.set_yticks([])
            for spine in ax5.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(2)

            plt.tight_layout()

            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            plt.close()

            result_image = Image.open(buf)

            # Statistics
            stats_text = f"""
            **Model Performance Metrics**
            
            - IoU(Intersection over Union) Score: {iou_score:.4f}
            - Pixel Accuracy: {pixel_accuracy:.4f} (binary pixel classification accuracy)
            """

            return result_image, stats_text

        except Exception as e:
            return None, f"Error during prediction: {str(e)}"

    def evaluate_batch(self, split: str, start_index: int, batch_size: int, threshold: float = 0.5):
        """Evaluate a batch of samples from the specified split"""
        try:
            # Update current split if needed
            if split != self.current_split:
                if not self.update_current_split(split):
                    return f"Split '{split}' not available"

            results = []
            total_samples = min(batch_size, len(
                self.current_dataset) - start_index)

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
                iou_score = self.calculate_iou(prediction.reshape(
                    1, -1), mask_np.reshape(1, -1), threshold)
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

                summary = f"""
                **Batch Evaluation Results - {split.upper()} Split**
                **Samples: {start_index} to {start_index + total_samples - 1}**
                
                **Average Metrics:**
                - Average IoU: {avg_iou:.4f}
                - Average Pixel Accuracy: {avg_pixel_acc:.4f}
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


def create_gradio_interface(model_path: str, base_data_path: str, patch_size: int = 256, version: int = 0):
    """Create Gradio interface"""
    app = SegmentationApp(model_path, base_data_path, patch_size, version)

    with gr.Blocks(title="VGG-16 based U-net Tissue Mask Generation demo", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# VGG-16 based U-net Tissue Mask Generation demo")

        with gr.Tab("Single Sample Prediction"):
            with gr.Row():
                with gr.Column():
                    split_selector = gr.Radio(
                        choices=["train", "test", "val"],
                        value="val",
                        label="Select Split",
                        interactive=True
                    )

                    image_dropdown = gr.Dropdown(
                        choices=app.get_image_choices("val"),
                        label="Select Image",
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
                    gr.Markdown("Samples are randomly selected from the dataset; hence, perfect accuracy/IoU scores typically occur when patches are taken from tissue centers (entire patch is tissue)."
                                )

                with gr.Column():
                    output_image = gr.Image(label="Prediction Results")
                    stats_output = gr.Markdown()

            def update_image_dropdown(split):
                choices = app.get_image_choices(split)
                if choices:
                    return gr.Dropdown(choices=choices, value=0)
                else:
                    return gr.Dropdown(choices=[], value=None)

            def update_dropdown(split, current_value, direction):
                choices = app.get_image_choices(split)
                if not choices:
                    return current_value

                max_value = len(choices) - 1
                if direction == "prev":
                    new_value = max(0, current_value - 1)
                else:  # next
                    new_value = min(max_value, current_value + 1)
                return new_value

            split_selector.change(
                fn=update_image_dropdown,
                inputs=[split_selector],
                outputs=[image_dropdown]
            )

            predict_btn.click(
                fn=app.predict_sample,
                inputs=[split_selector, image_dropdown, threshold_slider],
                outputs=[output_image, stats_output]
            )

            prev_btn.click(
                fn=lambda s, x: update_dropdown(s, x, "prev"),
                inputs=[split_selector, image_dropdown],
                outputs=[image_dropdown]
            )

            next_btn.click(
                fn=lambda s, x: update_dropdown(s, x, "next"),
                inputs=[split_selector, image_dropdown],
                outputs=[image_dropdown]
            )

        with gr.Tab("Custom Upload"):
            with gr.Row():
                with gr.Column():
                    upload_image = gr.Image(
                        label="Upload Custom Slide Image",
                        type="pil",
                        height=300
                    )

                    threshold_slider_custom = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                        label="Prediction Threshold"
                    )

                    predict_custom_btn = gr.Button(
                        "Predict Custom Image", variant="primary")

                    gr.Markdown("""
                    **Instructions:**
                    - Upload any slide image (PNG, JPG, JPEG)
                    - The image will be automatically resized to 256x256 pixels
                    - Adjust the threshold to fine-tune the tissue detection
                    - Green overlay shows predicted tissue regions
                    """)

                with gr.Column():
                    custom_output_image = gr.Image(
                        label="Custom Prediction Results")
                    custom_stats_output = gr.Markdown()

            predict_custom_btn.click(
                fn=app.predict_custom_image,
                inputs=[upload_image, threshold_slider_custom],
                outputs=[custom_output_image, custom_stats_output]
            )

        with gr.Tab("Batch Evaluation"):
            with gr.Row():
                with gr.Column():
                    split_selector_batch = gr.Radio(
                        choices=["train", "test", "val"],
                        value="val",
                        label="Select Split",
                        interactive=True
                    )

                    start_index = gr.Number(
                        value=0, minimum=0,
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
                    evaluate_btn = gr.Button(
                        "Evaluate Batch", variant="primary")

                with gr.Column():
                    batch_results = gr.Markdown()

            evaluate_btn.click(
                fn=app.evaluate_batch,
                inputs=[split_selector_batch, start_index,
                        batch_size, threshold_slider_batch],
                outputs=[batch_results]
            )

        with gr.Tab("Info"):
            dataset_info = gr.Markdown()

            def get_dataset_info():
                info = f"""
                This is a small demo for demonstration purposes.
                You can access the code and results [here](https://github.com/tangezerman/U-net-Tissue-Mask-Generation).
                The model was trained using a private dataset that was provided to students in the class. The dataset has
                over 290 GBs worth of high resolution WSI, coloured by various staining techniques.
                Our goal was to seperate the tissues from background and obtain tissue masks using a deep learning approach of our choice. 
                The model architecture uses weights of VGG 16 trained on IMAGENET.
                """
                return info

            interface.load(
                fn=get_dataset_info,
                outputs=[dataset_info]
            )

    return interface


if __name__ == "__main__":
    # Configuration - Update these paths according to your setup
    patch_size = 256
    version = 0
    batch_size = 48

    # Paths
    base_data_path = "Data/hf_Data_256_0"
    model_path = "Models/model__256_48_0.pth"

    # Create and launch interface
    interface = create_gradio_interface(
        model_path, base_data_path, patch_size, version)
    interface.launch(
        share=True,  # Set to False if you don't want to create a public link
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860
    )
