import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from train_small import AutoencoderWithSkip, calculate_iou
from sklearn.metrics import precision_score, recall_score


def tensor_to_np(tensor):
    tensor = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    return np.clip(tensor, 0, 1)


def visualize_results(original, masked, inpainted, save_path=None):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original)
    axs[0].set_title("Original Image")
    axs[1].imshow(masked)
    axs[1].set_title("Masked Image")
    axs[2].imshow(inpainted)
    axs[2].set_title("Inpainted Image")

    for ax in axs:
        ax.axis("off")

    if save_path:
        plt.savefig(save_path)
        print(f"Result saved to: {save_path}")
    plt.show()


def calculate_metrics(pred, target, threshold=0.5):
    """Calculate Precision, Recall, and IoU after binarization."""
    pred = (pred > threshold).float()
    target = (target > threshold).float()

    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()

    precision = precision_score(target_np, pred_np, zero_division=0)
    recall = recall_score(target_np, pred_np, zero_division=0)

    intersection = (pred * target).sum().item()
    union = ((pred + target) > 0).float().sum().item()
    iou = intersection / union if union > 0 else 0.0

    return precision, recall, iou


def main():
    model_path = "autoencoder_skip_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Trained model '{model_path}' not found. Please train the model first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoencoderWithSkip().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with open("train_image_paths.txt", 'r') as f:
        image_paths = [line.strip() for line in f]

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    mask_dir = "masks"

    iou_scores = []
    precision_scores = []
    recall_scores = []

    for img_path in image_paths[:10]:  
        print(f"Processing: {img_path}")

        try:
            img = Image.open(img_path).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device)

            # Load corresponding mask
            base_name = os.path.basename(img_path)
            mask_name = os.path.splitext(base_name)[0] + "_mask.png"
            mask_path = os.path.join(mask_dir, mask_name)

            if os.path.exists(mask_path):
                mask_img = Image.open(mask_path).convert('L').resize((128, 128))
                mask_np = np.array(mask_img) / 255.0
                mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(device)
            else:
                print(f"Warning: No mask found for {img_path}. Skipping...")
                continue

            # Apply mask to the image
            mask_binary = mask_t.squeeze(1).bool().unsqueeze(1).expand(-1, 3, -1, -1)
            masked_img = img_t.clone()
            masked_img[mask_binary] = 0.0

            # Perform inpainting
            with torch.no_grad():
                output = model(masked_img)

            # Calculate metrics
            precision, recall, iou = calculate_metrics(output, img_t)
            iou_scores.append(iou)
            precision_scores.append(precision)
            recall_scores.append(recall)

            # Visualize results
            original_np = tensor_to_np(img_t)
            masked_np = tensor_to_np(masked_img)
            inpainted_np = tensor_to_np(output)

            save_path = f"inpainting_result_{base_name}.png"
            visualize_results(original_np, masked_np, inpainted_np, save_path=save_path)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if iou_scores and precision_scores and recall_scores:
        print("\nFinal Evaluation Metrics:")
        print(f"Average Precision: {np.mean(precision_scores):.4f}")
        print(f"Average Recall: {np.mean(recall_scores):.4f}")
        print(f"Average IoU: {np.mean(iou_scores):.4f}")
    else:
        print("No valid images processed. Please check your input paths and masks.")


if __name__ == "__main__":
    main()




    #testing
# import os
# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# from train_small import AutoencoderWithSkip, calculate_iou
# from sklearn.metrics import precision_score, recall_score

# def tensor_to_np(tensor):
#     tensor = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
#     return np.clip(tensor, 0, 1)

# def visualize_results(original, masked, inpainted, save_path=None):
#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#     axs[0].imshow(original)
#     axs[0].set_title("Original Image")
#     axs[1].imshow(masked)
#     axs[1].set_title("Masked Image")
#     axs[2].imshow(inpainted)
#     axs[2].set_title("Inpainted Image")

#     for ax in axs:
#         ax.axis("off")

#     if save_path:
#         plt.savefig(save_path)
#         print(f"Result saved to: {save_path}")
#     plt.show()

# def calculate_metrics(pred, target, threshold=0.5):
#     pred = (pred > threshold).float()
#     target = (target > threshold).float()

#     # Flatten tensors
#     pred_np = pred.cpu().numpy().flatten()
#     target_np = target.cpu().numpy().flatten()

#     precision = precision_score(target_np, pred_np, zero_division=0)
#     recall = recall_score(target_np, pred_np, zero_division=0)

#     intersection = (pred * target).sum().item()
#     union = ((pred + target) > 0).float().sum().item()
#     iou = intersection / union if union > 0 else 0.0

#     return precision, recall, iou

# def main():
#     model_path = "autoencoder_skip_model.pth"
#     if not os.path.exists(model_path):
#         print(f"Error: Trained model '{model_path}' not found. Please train the model first.")
#         return

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = AutoencoderWithSkip().to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()

#     # Load test dataset
#     from dataset import InpaintingDataset
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor()
#     ])
#     test_dataset = InpaintingDataset(
#         "test_image_paths.txt",
#         transform=transform,
#         mask_dir="masks",
#         epoch=0,  # Default epoch for testing
#         total_epochs=1  # Default total_epochs for testing
#     )
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

#     iou_scores = []
#     precision_scores = []
#     recall_scores = []

#     for i, (masked_img, mask, gt_img) in enumerate(test_loader):
#         masked_img, gt_img = masked_img.to(device), gt_img.to(device)
#         with torch.no_grad():
#             output = model(masked_img)

#         # Calculate metrics
#         precision, recall, iou = calculate_metrics(output, gt_img)
#         iou_scores.append(iou)
#         precision_scores.append(precision)
#         recall_scores.append(recall)

#         # Visualize a few test results
#         if i < 5:  # Visualize first 5 batches
#             original_np = tensor_to_np(gt_img[0])
#             masked_np = tensor_to_np(masked_img[0])
#             inpainted_np = tensor_to_np(output[0])

#             visualize_results(original_np, masked_np, inpainted_np)

#     # Print final metrics
#     print("\nFinal Evaluation Metrics:")
#     print(f"Average Precision: {np.mean(precision_scores):.4f}")
#     print(f"Average Recall: {np.mean(recall_scores):.4f}")
#     print(f"Average IoU: {np.mean(iou_scores):.4f}")

# if __name__ == "__main__":
#     main()
