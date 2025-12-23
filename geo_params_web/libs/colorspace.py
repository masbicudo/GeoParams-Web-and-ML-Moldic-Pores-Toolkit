import numpy as np
import numba
from numpy.typing import NDArray

def bgr2cmyk(bgr_img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if bgr_img is None:
        raise ValueError("bgr_img is None. Did you forget to read/load it?")
    return bgr2cmyk_inner(np.ascontiguousarray(bgr_img))

@numba.njit(parallel=True, error_model="numpy", fastmath=True)
def bgr2cmyk_inner(bgr_img) -> NDArray[np.uint8]:
    (height, width) = bgr_img.shape[:2]
    CMYK = np.empty((height, width, 4), dtype=np.uint8)
    for i in numba.prange(height):
        for j in range(width):
            B,G,R = bgr_img[i,j]
            J = max(R, G, B)
            
            K = np.uint8(255 - J)
            
            if J == 0:
                C = M = Y = np.uint8(0)
            else:
                C = np.uint8(255 * (J - R) / J)
                M = np.uint8(255 * (J - G) / J)
                Y = np.uint8(255 * (J - B) / J)

            CMYK[i,j] = (C,M,Y,K)
    return CMYK
