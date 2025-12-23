from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

class ImageSubdivDataset(Dataset):
    def __init__(self, rf, shape_size, line_items_count,
                 column_items_count, augmentations=None):
        """
        Args:
            rf (Callable): Region filter predicate -> bool = rf(region, all_regions, image)
            shape_size (tuple): Desired output shape (H, W).
            line_items_count (int): Number of items per line to extract.
            column_items_count (int): Number of items per column to extract.
            augmentations (list of Callable, optional): List of augmentations to apply.
        """
        self.rf = rf
        self.shape_size = shape_size  # (H, W)
        self.line_items_count = line_items_count
        self.column_items_count = column_items_count
        self.samples = []

        # For each sample, we’ll create 8 augmented variants
        if augmentations is None:
            augmentations = self._default_augmentations()
        self.augmentations = list(self._multiply_augmentations(augmentations))

    def add_image(self, path, zoom: float|int = 1.0, bbox: tuple | list | dict | None = None):
        img = Image.open(path)
        if bbox is not None:
            if isinstance(bbox, dict):
                if "x" in bbox and "y" in bbox and "width" in bbox and "height" in bbox:
                    bbox = (bbox["x"], bbox["y"], bbox["x"] + bbox["width"], bbox["y"] + bbox["height"])
                else: raise ValueError("Invalid bbox dictionary format")
            if len(bbox) == 4:
                img = img.crop(bbox)
            else:
                raise ValueError("Invalid bbox format")
        if zoom != 1.0:
            zoom_filter = Image.LANCZOS if zoom < 1.0 else Image.BICUBIC
            img = img.resize((int(img.width * zoom), int(img.height * zoom)), zoom_filter)
        xs = np.linspace(0, img.width - self.shape_size[1], self.column_items_count, dtype=int)
        ys = np.linspace(0, img.height - self.shape_size[0], self.line_items_count, dtype=int)
        for x in xs:
            for y in ys:
                box = (x, y, x + self.shape_size[1], y + self.shape_size[0])
                if self.rf is None or self.rf(box, img):
                    self.samples.append({"box": box, "image": img})

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
        item_idx = idx // len(self.augmentations)
        aug_idx = idx % len(self.augmentations)
        subdiv_info = self.samples[item_idx]
        box = subdiv_info["box"]
        subimg = subdiv_info["image"].crop(box)  # PIL Image

        # Apply the specific augmentation
        aug = self.augmentations[aug_idx]
        subimg = aug(subimg)

        subimg = TF.to_tensor(subimg)  # [3,H,W], float

        return subimg
