"""
DICTIONARY LEARNING

Author: Vighnesh Nayak
Date: 22 Apr 2024
Github: https://github.com/Kugelblitz25
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
from src.processor import Processor
from src.learner import Dictionary

def bayerCFA(image: np.ndarray) -> np.ndarray:
    """
    Function to simulate Bayer CFA on an image

    Args:
    image: np.ndarray: The image to simulate Bayer CFA on

    Returns:
    np.ndarray: The image with Bayer CFA applied
    """
    # Create a copy of the image
    image_bayer = np.zeros_like(image)

    # Create the Bayer CFA pattern
    image_bayer[0::2, 0::2, 0] = image[0::2, 0::2, 0]   # Red
    image_bayer[1::2, 0::2, 1] = image[1::2, 0::2, 1]   # Green 
    image_bayer[0::2, 1::2, 1] = image[0::2, 1::2, 1]   # Green
    image_bayer[1::2, 1::2, 2] = image[1::2, 1::2, 2]   # Blue  

    return image_bayer

processor = Processor()
k = 100
b = 5
lambda_ = 0.001

img = cv.cvtColor(cv.imread('images/lighthouse.jpg'), cv.COLOR_BGR2RGB)
img_bayer = bayerCFA(img)
img_interpolated = processor.interpolate(img_bayer) 
Y0 = processor.downsample(img_interpolated, 2)
Y_CFA = bayerCFA(Y0)
X0 = processor.interpolate(Y_CFA)
Z = processor.getZ(Y0, X0, b)
l, c = Z.shape

dict_ = Dictionary(l, k, c)
dict_.learn(Z, 100, lambda_)
Y = processor.dictionaryCorrection(dict_.D, img_interpolated, b, lambda_)

plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(img_bayer)
plt.title('Bayer CFA Image')
plt.axis('off')
plt.subplot(2, 2, 3)
plt.imshow(img_interpolated)
plt.title('Interpolated Image')
plt.axis('off')
plt.subplot(2, 2, 4)
plt.imshow(Y)
plt.title('Corrected Image')
plt.axis('off')
plt.tight_layout()
plt.savefig('images/corrected_image.png')
plt.show()