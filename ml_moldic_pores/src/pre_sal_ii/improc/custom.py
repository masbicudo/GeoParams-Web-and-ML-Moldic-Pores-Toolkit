import cv2
from pre_sal_ii.improc import adjust_gamma, colorspace
from numpy.typing import NDArray
import pandas as pd
import numpy as np
import cv2.typing as cvt
import PIL.Image as PILImage

def proc_pores_basic(
            path_or_image: str|cvt.MatLike|PILImage.Image,
            dist_from_center: float = 0.8, kernel_size: int = 10,
            gamma: float|None = None
        ) -> NDArray[np.uint8]:
    """Process image to detect pores using basic image processing techniques.

    Args:
        path_or_image: Path to the image file or an image array (numpy ndarray or cv2 Mat or PIL Image).
        dist_from_center: Fraction of the max region distance used to filter regions.
        kernel_size: Size of the morphological kernel.

    Returns:
    Binary image (numpy ndarray, dtype=uint8) with detected pores marked as 255.
    """

    if isinstance(path_or_image, str):
        inputImage = cv2.imread(path_or_image)
    elif isinstance(path_or_image, PILImage.Image):
        print("PIL Image detected, converting to BGR format")
        inputImage = cv2.cvtColor(np.array(path_or_image), cv2.COLOR_RGB2BGR)
    else:
        inputImage = path_or_image

    if gamma is not None:
        inputImage = adjust_gamma(inputImage, gamma)

    # BGR to CMYK:
    inputImageCMYK = colorspace.bgr2cmyk(inputImage)
    # Use numpy arrays (with same dtype as the image) for lower/upper bounds
    # so the type-checker matches one of cv2.inRange overloads.
    lower = np.array([92, 0, 0, 0], dtype=inputImageCMYK.dtype)
    upper = np.array([255, 255, 64, 196], dtype=inputImageCMYK.dtype)
    binaryImage = cv2.inRange(inputImageCMYK, lower, upper)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_ERODE, kernel, iterations=1)
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_DILATE, kernel, iterations=1)
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_DILATE, kernel, iterations=1)
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_ERODE, kernel, iterations=1)

    # Label connected components
    from skimage.measure import label, regionprops
    # Ensure label image is an ndarray with a concrete integer dtype
    label_img = np.asarray(label(binaryImage), dtype=np.int32)
    regions = regionprops(label_img)

    # Analyze properties of labeled regions
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

    # Filter regions based on max distance
    max_dist = max(df["max-dist"])
    pores_image3 = np.zeros(label_img.shape, dtype=np.uint8)
    for it, region in enumerate(regions):
        if df["max-dist"].iloc[it] <= max_dist*dist_from_center:
            color_value = 255
            pores_image3[region.coords.T[0], region.coords.T[1]] = color_value
    
    return pores_image3


def proc_moldic_pores(path: str):
    """Process image to detect moldic pores using color thresholding."""
    inputImage_cl = cv2.imread(path)
    binaryImage_clRed = cv2.inRange(
        inputImage_cl,
        #  B,   G,   R
        (  0,   0, 240),
        (  5,   5, 255))
    return binaryImage_clRed
