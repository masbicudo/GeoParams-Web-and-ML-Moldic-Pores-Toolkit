import cv2
import numpy as np
import numba

@numba.njit(parallel=True, error_model="numpy", fastmath=True)
def bgr2cmyk(bgr_img):
    bgr_img = np.ascontiguousarray(bgr_img)
    (height, width) = bgr_img.shape[:2]
    CMYK = np.empty((height, width, 4), dtype=np.uint8)
    for i in numba.prange(height):
        for j in range(width):
            B,G,R = bgr_img[i,j]
            J = max(R, G, B)
            K = np.uint8(255 - J)
            C = np.uint8(255 * (J - R) / J)
            M = np.uint8(255 * (J - G) / J)
            Y = np.uint8(255 * (J - B) / J)
            CMYK[i,j] = (C,M,Y,K)
    return CMYK

# --- LAB channels ---
def bgr2lab(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    return L, a, b

