import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class WhitePixelRegionDataset(Dataset):
    def __init__(self, image_binary_pores, image_thin_section, image_target,
                 region_size=101, num_samples=10000, seed: int|None=42, use_img_to_tensor=False):
        # Load the image in grayscale
        self.image_binary_pores = image_binary_pores
        self.image_thin_section = image_thin_section
        self.image_target = image_target
        self.region_size = region_size
        self.half_size = region_size // 2
        self.num_samples = num_samples

        # Set random seed for reproducibility
        self.local_random = random.Random(seed) if seed is not None else None

        # Get coordinates of all white pixels
        white = 255 if image_binary_pores.dtype == np.uint8 else 1.0
        self.white_coords = np.column_stack(np.where(self.image_binary_pores == white))

        # Filter out white pixels that are within 50 pixels of the image border
        self.white_coords = [
            (y, x) for y, x in self.white_coords 
            if self.half_size <= y < self.image_binary_pores.shape[0] - self.half_size 
            and self.half_size <= x < self.image_binary_pores.shape[1] - self.half_size
        ]

        # If there are fewer white pixels than requested samples, reduce sample size
        if self.num_samples == -1:
            self.num_samples = len(self.white_coords)
        self.num_samples = min(self.num_samples, len(self.white_coords))

        # Randomly sample white pixel coordinates
        if self.local_random is not None:
            self.sample_coords = self.local_random.sample(
                self.white_coords, self.num_samples)
        else:
            self.sample_coords = self.white_coords[0:self.num_samples]

        self.to_tensor = ((lambda x: TF.to_tensor(x).float())
                          if use_img_to_tensor
                          else (lambda x: torch.tensor(x, dtype=torch.float32)))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get the center pixel coordinates for the region
        center_pixel = self.sample_coords[idx]

        # Calculate top-left and bottom-right corners of the 101x101 region
        top_left_x = center_pixel[1] - self.half_size
        top_left_y = center_pixel[0] - self.half_size
        bottom_right_x = center_pixel[1] + self.half_size
        bottom_right_y = center_pixel[0] + self.half_size

        # Extract the region without needing padding (boundary-safe by design)
        region1 = self.image_thin_section[top_left_y:bottom_right_y + 1, top_left_x:bottom_right_x + 1]
        if self.image_target is not None:
            target = self.image_target[center_pixel[0]:center_pixel[0] + 1, center_pixel[1]:center_pixel[1] + 1]
        else:
            target = None

        # Convert to tensor and return
        return (
            self.to_tensor(region1),
            self.to_tensor(target) if target is not None else None,
            torch.tensor(center_pixel),
            )
        
    def __iter__(self):
        for idx in range(self.num_samples):
            yield self.__getitem__(idx)
    
    def get_whites_in_target(self):
        sum_all = 0
        for idx in range(self.num_samples):
            # Get the center pixel coordinates for the region
            center_pixel = self.sample_coords[idx]

            tg_top_left_x = center_pixel[0]# - self.target_half_size
            tg_top_left_y = center_pixel[1]# - self.target_half_size
            tg_bottom_right_x = center_pixel[0]# + self.target_half_size
            tg_bottom_right_y = center_pixel[1]# + self.target_half_size

            target = self.image_target[
                tg_top_left_y:tg_bottom_right_y + 1,
                tg_top_left_x:tg_bottom_right_x + 1]

            sum_all += np.sum(target == 1.)
        
        total_pixels = self.num_samples# * (self.target_region_size ** 2)
        
        return sum_all, total_pixels
