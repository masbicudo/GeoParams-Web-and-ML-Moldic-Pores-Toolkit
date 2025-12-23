import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def scale_image_and_save(input_file, output_path, scale_percent):
    image_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"{output_path}/{image_name}_{scale_percent}.jpg"

    if os.path.exists(output_file):
        return

    # Reading an image in default mode:
    inputImage = cv2.imread(input_file)

    # Set the scaling factors
    scale_percent = 25  # e.g., downscale by 50%

    # Calculate the new dimensions
    width = int(inputImage.shape[1] * scale_percent / 100)
    height = int(inputImage.shape[0] * scale_percent / 100)
    new_dimensions = (width, height)

    resized_image = cv2.resize(inputImage, new_dimensions, interpolation=cv2.INTER_AREA)
    os.makedirs(f"{output_path}/", exist_ok=True)
    cv2.imwrite(
        output_file,
        resized_image,
        [cv2.IMWRITE_JPEG_QUALITY, 99]
        )
    
def adjust_gamma(image, gamma=1.0):
   invGamma = 1.0 / gamma
   table = np.array([
      ((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)])
   return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))


def generate_region_map_from_centroids(mask, centroids):
    """
    Quickly assign each pixel in mask to its nearest centroid.
    mask: binary or any 2D array (H, W)
    centroids: np.ndarray of shape (n_clusters, 2), (row, col)
    """
    H, W = mask.shape
    coords = np.column_stack(np.nonzero(mask))
    centroids = np.asarray(centroids)

    # Compute squared distances (vectorized)
    d2 = np.sum((coords[:, None, :] - centroids[None, :, :])**2, axis=2)

    # Assign each pixel to closest centroid
    labels = np.argmin(d2, axis=1)

    # Reconstruct full region image
    regions = np.zeros((H, W), dtype=np.int32)
    regions[coords[:, 0], coords[:, 1]] = labels

    return regions

import cv2
import numpy as np

def ensure_binary(image, threshold=128):
    """
    Ensures that an input image is binary (0 or 255).
    - If image has 3 channels (RGB/BGR), convert to grayscale then binary.
    - If image is grayscale, convert to binary.
    - If image is already binary (only 0 and 255), do nothing.
    """

    # If image has 3 channels → convert to grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Check if already binary (only 0 and 255)
    unique_vals = np.unique(gray)
    if np.array_equal(unique_vals, [0, 255]) or np.array_equal(unique_vals, [0]) or np.array_equal(unique_vals, [255]):
        return gray  # already binary

    # If not binary, threshold it
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    return binary

from skimage.measure import label, regionprops
from typing import cast

def preprocess_segments(inputImage, gray_threshold=128, area_threshold=0.01, center_distance_threshold=0.8, morphological_processing:bool|str|dict=False):

    binaryImage = ensure_binary(inputImage, threshold=gray_threshold)

    if morphological_processing:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_ERODE, kernel, iterations=1)
        binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_DILATE, kernel, iterations=1)
        binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_DILATE, kernel, iterations=1)
        binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_ERODE, kernel, iterations=1)

    label_img = cast(np.ndarray, label(binaryImage))
    regions = regionprops(label_img)

    all_objs = []
    for it, region in enumerate(regions):
        ys = (region.coords.T[0] - label_img.shape[0]/2)/(label_img.shape[0]/2)
        xs = (region.coords.T[1] - label_img.shape[1]/2)/(label_img.shape[1]/2)
        obj = {
            "area": region.area,
            "max-dist": max((ys**2 + xs**2)**0.5),
        }
        all_objs.append(obj)

    df = pd.DataFrame(all_objs)

    total_area = binaryImage.shape[0] * binaryImage.shape[1]
    max_dist = max(df["max-dist"])
    pores_image3 = np.zeros(label_img.shape, dtype=np.uint8)
    for it, region in enumerate(regions):
        if df["max-dist"].iloc[it] > max_dist*center_distance_threshold:
            continue
        if df["area"].iloc[it] > total_area*area_threshold:
            continue
        color_value = 255
        pores_image3[region.coords.T[0], region.coords.T[1]] = color_value

    binaryImage = pores_image3
    if morphological_processing == "grow":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_DILATE, kernel, iterations=1)
    if morphological_processing == "shrink":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_ERODE, kernel, iterations=1)
    if isinstance(morphological_processing, dict) and "grow" in morphological_processing:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphological_processing["grow"], morphological_processing["grow"]))
        binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_DILATE, kernel, iterations=1)
    if isinstance(morphological_processing, dict) and "shrink" in morphological_processing:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphological_processing["shrink"], morphological_processing["shrink"]))
        binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_ERODE, kernel, iterations=1)

    return binaryImage/255.0

def rescale(value, min_src, max_src, min_dst, max_dst):
    return min_dst + (value - min_src) * (max_dst - min_dst) / (max_src - min_src)
