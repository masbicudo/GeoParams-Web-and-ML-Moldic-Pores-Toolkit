from typing import Any
import numpy as np
import cv2
from pre_sal_ii.improc import colorspace
from typing import cast
from typing import Sequence, Collection

def compute_porosity(binary_image: np.ndarray[Any, np.dtype[np.uint8]]) -> int:
    porosity = np.sum(binary_image) / (binary_image.shape[0] * binary_image.shape[1] * 255)
    return int(porosity)

def compute_components(binary_image: np.ndarray[Any, np.dtype[np.uint8]]) -> int:
    num_labels, labels_im = cv2.connectedComponents(binary_image)
    return num_labels - 1  # Subtract 1 to ignore the background label

def compute_mean_image(
            image: np.ndarray[Any, np.dtype[np.uint8]],
            max_ks: Collection[int],
            min_cs: Collection[int],
            show_progress: bool = False,
        ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], list[int], list[int]]:
    
    if len(max_ks) != len(min_cs):
        raise ValueError("max_ks and min_cs must have the same length")
    
    total = len(max_ks)

    mean_image = np.zeros(image.shape[0:2], dtype=np.float32)

    porosities = []
    component_counts = []

    if show_progress:
        from pre_sal_ii import progress
        iterator = progress(zip(max_ks, min_cs), total=total)
    else:
        iterator = zip(max_ks, min_cs)
        
    for it, (k, c) in enumerate(iterator):
        # print(it, k, c)
        
        img_cmyk = colorspace.bgr2cmyk(image.astype(np.uint8))
        
        lower_range = np.array([  c,   0,   0,   0], dtype=np.uint8)
        upper_range = np.array([255, 255,  64,   k], dtype=np.uint8)
        binaryImage = cast(np.ndarray[Any, np.dtype[np.uint8]], cv2.inRange(
            img_cmyk,
            lower_range,
            upper_range
        ))

        porosity = compute_porosity(binaryImage)
        porosities.append(porosity)

        components = compute_components(binaryImage)
        component_counts.append(components)

        binary_f = np.asarray(binaryImage, dtype=np.float32)
        mean_image += binary_f / float(total)
    return mean_image, porosities, component_counts

def compute_std_image(
            image: np.ndarray[Any, np.dtype[np.uint8]],
            max_ks: Collection[int],
            min_cs: Collection[int],
            mean_image: np.ndarray[Any, np.dtype[np.float32]],
            show_progress: bool = False,
        ) -> np.ndarray[Any, np.dtype[np.float32]]:
    
    if len(max_ks) != len(min_cs):
        raise ValueError("max_ks and min_cs must have the same length")
    
    total = len(max_ks)

    variance_image = np.zeros(image.shape[0:2], dtype=np.float32)

    if show_progress:
        from pre_sal_ii import progress
        iterator = progress(zip(max_ks, min_cs), total=total)
    else:
        iterator = zip(max_ks, min_cs)

    for it, (k, c) in enumerate(iterator):
        # print(it, k, c)
        
        img_cmyk = colorspace.bgr2cmyk(image.astype(np.uint8))
        
        lower_range = np.array([  c,   0,   0,   0], dtype=np.uint8)
        upper_range = np.array([255, 255,  64,   k], dtype=np.uint8)
        binaryImage = cast(np.ndarray[Any, np.dtype[np.uint8]], cv2.inRange(
            img_cmyk,
            lower_range,
            upper_range
        ))

        variance_image += (binaryImage - mean_image)**2 / (total - 1)

    std_image = np.sqrt(variance_image)
    return std_image
