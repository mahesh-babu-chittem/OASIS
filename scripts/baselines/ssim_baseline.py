import cv2
import numpy as np
from PIL import Image
import os
from skimage.metrics import structural_similarity as ssim

def compute_ssim(path1, path2):
    img1 = np.array(Image.open(os.path.join("data/processed", path1)).convert("RGB"))
    img2 = np.array(Image.open(os.path.join("data/processed", path2)).convert("RGB"))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    return ssim(gray1, gray2)