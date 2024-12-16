
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import random
import numpy as np




class InpaintingDataset(Dataset):
    def __init__(self, image_list_file, transform=None, mask_dir=None, epoch=0, total_epochs=20):
        """
        Args:
            image_list_file (str): Path to the file containing the list of image paths.
            transform (callable, optional): Transform to be applied on a sample.
            mask_dir (str, optional): Directory where YOLO masks are stored.
            epoch (int, optional): Current epoch number (for progressive mask size adjustment).
            total_epochs (int, optional): Total number of epochs.
        """
        with open(image_list_file, 'r') as f:
            self.image_paths = [line.strip() for line in f if line.strip()]
        self.transform = transform
        self.mask_dir = mask_dir
        self.epoch = epoch
        self.total_epochs = total_epochs 

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        base_name = os.path.basename(path)
        mask_name = os.path.splitext(base_name)[0] + "_mask.png"
        mask_path = os.path.join(self.mask_dir, mask_name) if self.mask_dir else None

        progress = self.epoch / self.total_epochs
        max_mask_ratio = 0.1 + progress * 0.3  

        if mask_path and os.path.exists(mask_path) and random.random() < 0.5:
            mask_img = Image.open(mask_path).convert('L')
            mask_img = mask_img.resize((image.size(2), image.size(1))) 
            mask = torch.from_numpy(np.array(mask_img)).float() / 255.0
        else:
            mask = self.random_mask(image.shape[-2:], max_mask_ratio)

        masked_image = image.clone()
        mask_expanded = mask.unsqueeze(0).expand_as(image)
        masked_image[mask_expanded.bool()] = 0.0

        return masked_image, mask.unsqueeze(0), image

    def random_mask(self, size, max_ratio):
        """
        Generate a random rectangular mask.

        Args:
            size (tuple): Size of the image (H, W).
            max_ratio (float): Maximum mask size as a fraction of the image size.

        Returns:
            torch.Tensor: Random mask of shape (H, W).
        """
        h, w = size
        mask = torch.zeros((h, w))

        max_rect_w = int(w * max_ratio)
        max_rect_h = int(h * max_ratio)

        rect_w_min = max(10, min(w // 8, max_rect_w))
        rect_h_min = max(10, min(h // 8, max_rect_h))
        rect_w_max = max(rect_w_min, min(max_rect_w, w // 2))
        rect_h_max = max(rect_h_min, min(max_rect_h, h // 2))

        if rect_w_min > rect_w_max:
            rect_w_min = rect_w_max
        if rect_h_min > rect_h_max:
            rect_h_min = rect_h_max

        rect_w = random.randint(rect_w_min, rect_w_max)
        rect_h = random.randint(rect_h_min, rect_h_max)
        x1 = random.randint(0, max(0, w - rect_w))
        y1 = random.randint(0, max(0, h - rect_h))
        mask[y1:y1 + rect_h, x1:x1 + rect_w] = 1.0
        return mask
    


#testing

    # class InpaintingDataset(Dataset):
#     def __init__(self, image_list_file, transform=None, mask_dir=None, epoch=0, total_epochs=20):
       
#         with open(image_list_file, 'r') as f:
#             self.image_paths = [line.strip() for line in f if line.strip()]
#         self.transform = transform
#         self.mask_dir = mask_dir
#         self.epoch = epoch
#         self.total_epochs = total_epochs

#     def __len__(self):
#         return len(self.image_paths)



#     def random_mask(self, size, max_ratio):
        
#         h, w = size
#         mask = torch.zeros((h, w))

#         max_rect_w = int(w * max_ratio)
#         max_rect_h = int(h * max_ratio)

#         rect_w_min = max(10, min(w // 8, max_rect_w))
#         rect_h_min = max(10, min(h // 8, max_rect_h))
#         rect_w_max = max(rect_w_min, min(max_rect_w, w // 2))
#         rect_h_max = max(rect_h_min, min(max_rect_h, h // 2))

#         # Ensure ranges are valid
#         if rect_w_min > rect_w_max:
#             rect_w_min = rect_w_max
#         if rect_h_min > rect_h_max:
#             rect_h_min = rect_h_max

#         rect_w = random.randint(rect_w_min, rect_w_max)
#         rect_h = random.randint(rect_h_min, rect_h_max)
#         x1 = random.randint(0, max(0, w - rect_w))
#         y1 = random.randint(0, max(0, h - rect_h))
#         mask[y1:y1 + rect_h, x1:x1 + rect_w] = 1.0
#         return mask

