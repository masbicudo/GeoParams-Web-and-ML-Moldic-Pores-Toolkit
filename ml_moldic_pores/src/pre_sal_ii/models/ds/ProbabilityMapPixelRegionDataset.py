import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

def sample_coords_from_probmap(prob_map, N, seed):
    # Flatten the 2D array into 1D
    flat_probs = prob_map.flatten().astype(np.float64)
    
    # Normalize so they sum to 1
    flat_probs /= flat_probs.sum()
    
    # Choose N indices according to the probability distribution
    local_random = np.random.default_rng(seed)
    indices = local_random.choice(len(flat_probs), size=N, replace=False, p=flat_probs)

    # Convert 1D indices back to 2D (y, x)
    ys, xs = np.unravel_index(indices, prob_map.shape)
    
    # Stack and return coordinates as (x, y)
    return np.column_stack((xs, ys))

def zero_border_prob(prob_map, border=50):
    # Copy to avoid modifying original
    prob = prob_map.copy()

    h, w = prob.shape

    # Zero out top and bottom borders
    prob[:border, :] = 0
    prob[-border:, :] = 0

    # Zero out left and right borders
    prob[:, :border] = 0
    prob[:, -border:] = 0

    # Renormalize so probabilities sum to 1 again
    total = float(prob.sum())
    if total > 0:
        prob /= total

    return prob

class ProbabilityMapPixelRegionDataset(Dataset):
    def __init__(self, image_prob, image_thin_section, image_target,
                 region_size=101, num_samples=10000, seed=42,
                 target_region_size=1):
        # Load the image in grayscale
        self.image_thin_section = image_thin_section
        self.image_target = image_target
        self.region_size = region_size
        self.half_size = region_size // 2
        self.target_region_size = target_region_size
        self.target_half_size = target_region_size // 2
        self.num_samples = num_samples
        # Filter out white pixels that are within 50 pixels of the image border
        self.image_prob = zero_border_prob(
            image_prob.astype(np.float32), self.half_size)

        # Set random seed for reproducibility
        self.local_random = random.Random(seed) if seed is not None else None
        # self.np_rng = np.random.default_rng(seed)

        # Get coordinates of pixels based on probability map
        self.pixel_coords = sample_coords_from_probmap(
            self.image_prob, N=num_samples, seed=seed)

        # If there are fewer pixels than requested samples, reduce sample size
        if self.num_samples == -1:
            self.num_samples = len(self.pixel_coords)
        self.num_samples = min(self.num_samples, len(self.pixel_coords))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get the center pixel coordinates for the region
        center_pixel = self.pixel_coords[idx]

        # Calculate top-left and bottom-right corners of the 101x101 region
        top_left_x = center_pixel[0] - self.half_size
        top_left_y = center_pixel[1] - self.half_size
        bottom_right_x = center_pixel[0] + self.half_size
        bottom_right_y = center_pixel[1] + self.half_size

        tg_top_left_x = center_pixel[0] - self.target_half_size
        tg_top_left_y = center_pixel[1] - self.target_half_size
        tg_bottom_right_x = center_pixel[0] + self.target_half_size
        tg_bottom_right_y = center_pixel[1] + self.target_half_size

        # Extract the region without needing padding (boundary-safe by design)
        region1 = self.image_thin_section[
            top_left_y:bottom_right_y + 1,
            top_left_x:bottom_right_x + 1]
        if self.image_target is not None:
            target = self.image_target[
                tg_top_left_y:tg_bottom_right_y + 1,
                tg_top_left_x:tg_bottom_right_x + 1]
        else:
            target = None

        # Convert to tensor and return
        return (
            TF.to_tensor(region1).float(),
            TF.to_tensor(target).float() if target is not None else None,
            torch.tensor(center_pixel),
            )

    def get_whites_in_target(self):
        sum_all = 0
        for idx in range(self.num_samples):
            # Get the center pixel coordinates for the region
            center_pixel = self.pixel_coords[idx]

            tg_top_left_x = center_pixel[0] - self.target_half_size
            tg_top_left_y = center_pixel[1] - self.target_half_size
            tg_bottom_right_x = center_pixel[0] + self.target_half_size
            tg_bottom_right_y = center_pixel[1] + self.target_half_size

            target = self.image_target[
                tg_top_left_y:tg_bottom_right_y + 1,
                tg_top_left_x:tg_bottom_right_x + 1]

            sum_all += np.sum(target == 1.)
        
        total_pixels = self.num_samples * (self.target_region_size ** 2)
        
        return sum_all, total_pixels
