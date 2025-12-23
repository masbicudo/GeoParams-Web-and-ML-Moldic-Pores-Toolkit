import cv2
import numpy as np

# --- Sobel gradient magnitude ---
def sobel(gray):
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)


# --- Hessian magnitude (2nd derivatives) ---
def hessian(gray):
    dxx = cv2.Sobel(gray, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(gray, cv2.CV_32F, 0, 2, ksize=3)
    hessian = np.sqrt(dxx**2 + dyy**2)
    hessian = cv2.normalize(hessian, None, 0, 255, cv2.NORM_MINMAX)
    return hessian.astype(np.uint8)

def local_variance(gray, ksize=9):
    mean = cv2.blur(gray.astype(np.float32), (ksize, ksize))
    sq_mean = cv2.blur(gray.astype(np.float32)**2, (ksize, ksize))
    var = np.sqrt(np.maximum(sq_mean - mean**2, 0))
    return cv2.normalize(var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def gabor_features(gray, frequencies=(0.05, 0.1, 0.2), thetas=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    feats = []
    for theta in thetas:
        for freq in frequencies:
            kernel = cv2.getGaborKernel((9, 9), 4.0, theta, 1.0/freq, 0.5, 0, ktype=cv2.CV_32F)
            fimg = cv2.filter2D(gray, cv2.CV_32F, kernel)
            feats.append(cv2.normalize(np.abs(fimg), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    return feats

