
from pathlib import Path
import pickle
from typing import cast
import cv2
from k_means_constrained import KMeansConstrained
import numpy as np
import pandas as pd
import logging
import torch

from skimage.measure import label, regionprops
import torch.nn.functional as F
from pre_sal_ii.improc import adjust_gamma, colorspace
from pre_sal_ii.models.features import data_models
from pre_sal_ii.training import Trainer
from pre_sal_ii.training.image_clustering import cluster_pixels_kmeans_constrained_model

logger = logging.getLogger(__name__)

class MyTrainer101x101to32x32(Trainer):
    def __init__(
                self, model, optimizer, criterion,
                device: str | torch.device = "cuda",
                channels=3, criterion_kwargs={},
            ):
        super().__init__(model, optimizer, criterion, device, criterion_kwargs)
        self.channels = channels

    def train_epoch_step(self, inputs):
        imgs = inputs[0].to(self.device)
        if logger.isEnabledFor(logging.DEBUG): assert (*imgs.shape[1:],) == (self.channels, 101, 101)
        imgs = F.interpolate(
            imgs, size=(32, 32), mode='bilinear',
            align_corners=False)
        if logger.isEnabledFor(logging.DEBUG): assert (*imgs.shape[1:],) == (self.channels, 32, 32)
        imgs = imgs.reshape(-1, self.channels*32*32)
        if logger.isEnabledFor(logging.DEBUG): assert (*imgs.shape[1:],) == (self.channels*32*32,)
        outputs = self.model(imgs)
        return imgs.shape[0], outputs

    def train_epoch_loss(self, inputs, outputs):
        expected = inputs[1].to(self.device)
        expected = torch.squeeze(expected, 1)
        expected = torch.squeeze(expected, 2)
        if logger.isEnabledFor(logging.DEBUG): assert (*expected.shape[1:],) == (1,)
        loss = self.criterion(outputs, expected, **self.criterion_kwargs)
        return loss


def get_input_image():
    
    image_name = "ML-tste_original"
    path = f"../out/classificada_01/{image_name}_25.jpg"
    inputImage_no_gamma: np.ndarray = cv2.imread(path)
    inputImage = adjust_gamma(inputImage_no_gamma, 0.5)
    
    return inputImage, inputImage_no_gamma

def get_probability_maps_simple(inputImage):

    # BGR to CMKY:
    inputImageCMYK = colorspace.bgr2cmyk(inputImage)

    binaryImage = cv2.inRange(
        inputImageCMYK,
        (92,   0,   0,   0),
        (255, 255,  64, 196))

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

    max_dist = max(df["max-dist"])
    pores_image3 = np.zeros(label_img.shape, dtype=np.uint8)
    for it, region in enumerate(regions):
        if df["max-dist"].iloc[it] <= max_dist*0.8:
            color_value = 255
            pores_image3[region.coords.T[0], region.coords.T[1]] = color_value

    return pores_image3/255.0

def load_manually_categorized_image():
    image_name = "ML-tste_classidicada"
    path = f"../out/classificada_01/{image_name}_25.jpg"
    inputImage_cl = cv2.imread(path)
    binaryImage_clRed: np.ndarray = cv2.inRange(
        inputImage_cl,
        #  B,   G,   R
        (  0,   0, 240),
        (  5,   5, 255))
    return binaryImage_clRed


def filter_central_objects(image: np.ndarray) -> np.ndarray:
    

    label_img = label(image)
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

    max_dist = max(df["max-dist"])
    binaryImage_clRed_mx = np.zeros(label_img.shape, dtype=np.uint8)
    for it, region in enumerate(regions):
        if df["max-dist"].iloc[it] <= max_dist*0.8:
            color_value = 255
            binaryImage_clRed_mx[region.coords.T[0], region.coords.T[1]] = color_value
    return binaryImage_clRed_mx


def get_kmc_model(binaryImage_clRed, debug=False) -> KMeansConstrained:
    cache_path = Path("../models/kmc_model_1.pkl")
    cache_path.parent.mkdir(exist_ok=True)

    if cache_path.exists():
        # Load cached model
        with open(cache_path, "rb") as f:
            kmc_model = pickle.load(f)
        if debug: print("Loaded cached model from disk.")
    else:
        # Train the model
        binaryImage_clRed_mx = filter_central_objects(binaryImage_clRed)
        kmc_model = cluster_pixels_kmeans_constrained_model(binaryImage_clRed_mx, fraction=10)
        # Save to cache
        with open(cache_path, "wb") as f:
            pickle.dump(kmc_model, f)
        if debug: print("Saved trained model to cache.")

    return kmc_model


def get_mean_stdev(inputImage_no_gamma):
    df = pd.read_csv("../data/c_min_k_max_params.csv")
    xs = df["clicked_x"].astype(int)
    ys = df["clicked_y"].astype(int)
    mean_img, _, _ = data_models.compute_mean_image(inputImage_no_gamma, xs, ys, show_progress=True)
    stdev_image = data_models.compute_std_image(inputImage_no_gamma, xs, ys, mean_img, show_progress=True)
    return mean_img, stdev_image
