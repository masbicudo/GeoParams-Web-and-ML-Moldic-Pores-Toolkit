import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
import torchvision.transforms.functional as TF

class PoreDataset(Dataset):
    def __init__(self, bf, rf, shape_resize, augmentations=None):
        """
        Args:
            bf (Callable): Binarization function -> mask = bf(image)
            rf (Callable): Region filter predicate -> bool = rf(region, all_regions, image)
            shape_resize (tuple): Desired output shape (H, W).
            augmentations (list of Callable, optional): List of augmentations to apply.
        """
        self.bf = bf
        self.rf = rf
        self.shape_resize = shape_resize  # (H, W)
        self.samples = []

        # For each pore, we’ll create 8 augmented variants
        if augmentations is None:
            augmentations = self._default_augmentations()
        self.augmentations = list(self._multiply_augmentations(augmentations))

    def add_image(self, path):
        img = Image.open(path)
        mask = self.bf(img)
        labeled = label(mask)
        all_regions = regionprops(labeled)
        for region in all_regions:
            if not self.rf(region, all_regions, img):
                continue
            bbox = region.bbox
            minr, minc, maxr, maxc = bbox
            patch_img = np.zeros((maxr-minr, maxc-minc), dtype=np.uint8)
            ys = region.coords.T[0]
            xs = region.coords.T[1]
            patch_img[ys-minr, xs-minc] = 255
            
            # Resize to shape_resize keeping aspect ratio and centered
            pore = self._resize_keep_aspect(Image.fromarray(patch_img), self.shape_resize)

            self.samples.append(pore)

    def _default_augmentations(self):
        fs = [
            [
                lambda img: img,
                lambda img: TF.rotate(img, 90),
                lambda img: TF.rotate(img, 180),
                lambda img: TF.rotate(img, 270)
            ],
            [
                lambda img: img,
                TF.hflip
            ],
        ]
        return fs
    
    def _multiply_augmentations(self, augmentations):
        """Generate all combinations of augmentations."""
        if len(augmentations) == 0:
            return [lambda img: img]
        first, *rest = augmentations
        if len(rest) == 0:
            return first
        else:
            rest_combinations = self._multiply_augmentations(rest)
            return [lambda img, f=f, r=r: r(f(img)) for r in rest_combinations for f in first]

    def __len__(self):
        return len(self.samples) * len(self.augmentations)

    def __getitem__(self, idx):
        pore_idx = idx // len(self.augmentations)
        aug_idx = idx % len(self.augmentations)
        pore = self.samples[pore_idx]

        # Apply the specific augmentation
        aug = self.augmentations[aug_idx]
        pore_aug = aug(pore)

        pore_aug = TF.to_tensor(pore_aug)  # [1,H,W], float

        return pore_aug

    def _resize_keep_aspect(self, img, size):
        """Resize while keeping aspect ratio and padding with black."""
        w, h = img.size
        target_w, target_h = size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = img.resize((new_w, new_h), Image.BILINEAR)

        new_img = Image.new("L", (target_w, target_h))
        left = (target_w - new_w) // 2
        top = (target_h - new_h) // 2
        new_img.paste(resized, (left, top))
        return new_img
