import cv2
import numpy as np
from scipy import stats
from PIL import Image

# === Utility Functions ===
def cv2_to_pil(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

def to_gray(img):
    if img is None:
        return None
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# === Point Operations ===
def point_add(img): return cv2.add(img, 50)
def point_subtract(img): return cv2.subtract(img, 50)
def point_divide(img): return np.clip(img.astype(np.float32)/2, 0, 255).astype(np.uint8)
def point_complement(img): return cv2.bitwise_not(img)

# === Color Image Operations ===
def change_light(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = cv2.add(hsv[..., 2], 50)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def change_red(img):
    img2 = img.copy()
    img2[..., 2] = cv2.add(img2[..., 2], 50)
    return img2

def swap_rg(img):
    img2 = img.copy()
    img2[..., [2, 1, 0]] = img2[..., [0, 1, 2]]
    return img2

def eliminate_red(img):
    img2 = img.copy()
    img2[..., 2] = 0
    return img2

# === Histogram Operations ===
def hist_stretch(img):
    gray = to_gray(img)
    min_val, max_val = np.min(gray), np.max(gray)
    stretched = ((gray - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)
    return cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)

def hist_equalize(img):
    gray = to_gray(img)
    eq = cv2.equalizeHist(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

# === Spatial Filters ===
def avg_filter(img): return cv2.blur(img, (3, 3))

def lap_filter(img):
    gray = to_gray(img)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)
    return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)

def max_filter(img): return cv2.dilate(img, np.ones((3, 3), np.uint8))
def min_filter(img): return cv2.erode(img, np.ones((3, 3), np.uint8))
def median_filter(img): return cv2.medianBlur(img, 3)

def mode_filter(img):
    def modefilt2d(a, window_size):
        pad_size = window_size // 2
        padded = np.pad(a, pad_size, mode='edge')
        out = np.zeros_like(a)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                window = padded[i:i+window_size, j:j+window_size]
                out[i, j] = stats.mode(window, axis=None)[0][0]
        return out
    if len(img.shape) == 2:
        return modefilt2d(img, 3)
    else:
        channels = cv2.split(img)
        filtered = [modefilt2d(ch, 3) for ch in channels]
        return cv2.merge(filtered)

# === Noise Addition & Removal ===
def add_salt_pepper(img):
    s_vs_p = 0.5
    amount = 0.05
    out = img.copy()
    num_salt = np.ceil(amount * img.size * s_vs_p)
    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape[:2]]
    out[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape[:2]]
    out[coords[0], coords[1]] = 0
    return out

def remove_sp_avg(img): return cv2.blur(img, (3, 3))
def remove_sp_median(img): return cv2.medianBlur(img, 3)

def remove_sp_outlier(img):
    result = img.copy()
    gray = to_gray(img)
    med = cv2.medianBlur(gray, 3)
    diff = cv2.absdiff(gray, med)
    mask = diff > 50
    if len(img.shape) == 3:
        for c in range(3):
            result[..., c][mask] = med[mask]
    else:
        result[mask] = med[mask]
    return result

def add_gauss(img):
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def remove_gauss_avg(img): return cv2.blur(img, (3, 3))

# === Thresholding & Segmentation ===
def thresh_basic(img):
    gray = to_gray(img)
    _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

def thresh_auto(img):
    gray = to_gray(img)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

def thresh_adapt(img):
    gray = to_gray(img)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

# === Edge Detection ===
def sobel(img):
    gray = to_gray(img)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sx, sy)
    sobel = np.clip(sobel, 0, 255).astype(np.uint8)
    return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

def apply_edge_detection(img):
    return cv2.Canny(to_gray(img), 100, 200)

# === Morphological Operations ===
def dilate(img): return cv2.dilate(img, np.ones((3, 3), np.uint8))
def erode(img): return cv2.erode(img, np.ones((3, 3), np.uint8))
def opening(img): return cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

def boundary_internal(img):
    gray = to_gray(img)
    eroded = cv2.erode(gray, np.ones((3, 3), np.uint8))
    boundary = cv2.subtract(gray, eroded)
    return cv2.cvtColor(boundary, cv2.COLOR_GRAY2BGR)

def boundary_external(img):
    gray = to_gray(img)
    dilated = cv2.dilate(gray, np.ones((3, 3), np.uint8))
    boundary = cv2.subtract(dilated, gray)
    return cv2.cvtColor(boundary, cv2.COLOR_GRAY2BGR)

def boundary_gradient(img):
    gray = to_gray(img)
    dilated = cv2.dilate(gray, np.ones((3, 3), np.uint8))
    eroded = cv2.erode(gray, np.ones((3, 3), np.uint8))
    gradient = cv2.subtract(dilated, eroded)
    return cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)

# === Extra Operations ===
def apply_blur(img): return cv2.GaussianBlur(img, (5, 5), 0)

def apply_sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def apply_hist_eq(img): return hist_equalize(img)

def apply_rotation(img):
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 45, 1.0)
    return cv2.warpAffine(img, M, (w, h))
