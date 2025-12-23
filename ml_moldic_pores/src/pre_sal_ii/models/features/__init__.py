import cv2
import numpy as np

from pre_sal_ii.improc.colorspace import bgr2cmyk
from pre_sal_ii.improc.filters import sobel, hessian, local_variance, gabor_features

# --- Compose all channels ---
# --- Build feature stack ---
def build_feature_stack(bgr):
    B, G, R = cv2.split(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    CMYK = bgr2cmyk(bgr)
    C, M, Y, K = cv2.split(CMYK)
    
    L, a, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
    H, S, V = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    
    sobel_channel = sobel(gray)
    hessian_channel = hessian(gray)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    laplacian = cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    median = cv2.medianBlur(gray, 5)
    
    scharrx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    scharry = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    scharr_mag = cv2.magnitude(scharrx, scharry)
    scharr_mag = cv2.normalize(scharr_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    variance = local_variance(gray)
    gabor_feats = gabor_features(gray)
    
    # Stack everything together
    stacked = np.stack(
        [R, G, B, gray, C, M, Y, K, L, a, b, H, S, V,
         sobel_channel, hessian_channel, laplacian, blur, median,
         scharr_mag, variance] + gabor_feats,
        axis=-1
    ).astype(np.float32) / 255.0
    
    return stacked
