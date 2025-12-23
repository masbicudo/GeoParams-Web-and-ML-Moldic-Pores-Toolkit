import cv2
import pandas as pd
import numpy as np
from skimage.measure import label, regionprops
import libs.colorspace as colorspace
from typing import cast, TypeVar, Any
from numpy.typing import NDArray
import os

def regions_df(regions, label_img):
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
    return df

T = TypeVar('T', int, float)
def rmm(value: T, minvalue: T, maxvalue: T) -> T:
    value = min(value, maxvalue)
    value = max(value, minvalue)
    return value

def binarize_c_k(
    base_image: NDArray[np.uint8],
    c: int,
    k: int
) -> tuple[NDArray[np.uint8], list[Any]]:
    lower_range = np.array([  c,   0,   0,   0], dtype=np.uint8)
    upper_range = np.array([255, 255,  64,   k], dtype=np.uint8)
    binaryImage = cv2.inRange(
        base_image,
        lower_range,
        upper_range
    )
    label_img = cast(np.ndarray, label(binaryImage))
    regions = regionprops(label_img)
    return label_img, regions

def get_images(base_image, step=4, thresh=480,
               set_total_steps=None, do_step=None
               ) -> tuple[NDArray[np.uint8] | None, list[list[NDArray[np.uint8]]] | None]:
    
    inputImageCMYK = colorspace.bgr2cmyk(base_image)

    import math
    if step != int(step) or math.log2(step) != int(math.log2(step)):
        raise Exception("step must be an integer power of 2")

    step = int(step)
    if set_total_steps is not None:
        set_total_steps((256//step)*(256//step))
    map_w = 256//step
    map_h = 256//step
    map_img: NDArray[np.uint8] = np.zeros(
        (map_h, map_w, 3),
        dtype=np.uint8)

    colors = [
        np.array([0, 0, 0]),         # black
        np.array([60, 60, 220]),     # red (softer)
        np.array([220, 100, 60]),    # blue (softer)
        np.array([75, 180, 60]),     # green (softer)
        np.array([53, 225, 255]),    # yellow (softer)
        np.array([200, 200, 70]),    # cyan (softer)
        np.array([210, 100, 210]),   # magenta (softer)
        np.array([255, 255, 255])    # white
    ]

    images: list[list[NDArray[np.uint8] | None]] = [
            [None for _ in range(256//step)] for _ in range(256//step)
        ]

    for c in [*range(0, 256, step)]:
        for k in range(0, 256, step):
            label_img, regions = binarize_c_k(inputImageCMYK, c, k)

            df = regions_df(regions, label_img)
            
            bin_image_4: NDArray[np.uint8] = np.zeros((*label_img.shape, 3), dtype=np.uint8)
            id_region = 0
            for it, region in enumerate(regions):
                #display(df)
                if df["area"].iloc[it] > thresh:
                    id_region += 1
                    #color_value = np.array([255, 255, 255])
                    color_value = colors[int(rmm(id_region, 0, len(colors)-1))]
                    ys = region.coords.T[0]
                    xs = region.coords.T[1]
                    bin_image_4[ys, xs] = color_value
            images[c//step][k//step] = bin_image_4
            
            # All None items have been replaced with np.ndarray
            
            # plt.imshow(binaryImage, cmap='gray')
            # plt.show()
            
            if len(regions) == 0:
                count_regions = 0
            else:
                count_regions = sum(df["area"] > thresh)
                
            #print(f"count_regions = {count_regions}")
            #map_img[c//step, y//step] = np.array([c, y, 0])
            map_img[c//step, k//step] = colors[int(rmm(count_regions, 0, len(colors)-1))]

            # `do_step` is a cancelation mechanism. It indicates whether to
            # continue processing or not. If the user cancels the operation,
            # for example by starting a new processing operation,
            # then it should return False for the previous call of get_images
            # and let the new call take over.
            if do_step is not None:
                if not do_step():
                    return None, None

    # At this point, all None values in images have been replaced with np.ndarray
    filled_images = cast(list[list[NDArray[np.uint8]]], images)
    return map_img, filled_images

def resize_image_file(in_file, out_file, percentage):
    image = cv2.imread(in_file)
    width = int(image.shape[1] * percentage/100)
    height = int(image.shape[0] * percentage/100)
    new_size = (width, height)
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    cv2.imwrite(out_file, resized_image)

def crop_image_file(in_file, out_file, x, y, width, height):
    image = cv2.imread(in_file)
    cropped_image = image[y:y+height, x:x+width]
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    cv2.imwrite(out_file, cropped_image)
