import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def semantic_sim(e1, e2):
    return np.dot(e1, e2)

def perceptual_sim(img1, img2):
    img1 = cv2.resize(img1, (64, 64))
    img2 = cv2.resize(img2, (64, 64))
    return np.corrcoef(img1.flatten(), img2.flatten())[0, 1]

def structural_sim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    return ssim(gray1, gray2)