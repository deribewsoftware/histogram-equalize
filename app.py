import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def histogramEqual():
    root = os.getcwd()
    imgPath = os.path.join(root, 'demoImages', 'badQuality.webp')  # Better path handling
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Failed to load image at {imgPath}")
        return

    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    cdf = hist.cumsum()
    cdfNorm = cdf * float(hist.max()) / cdf.max()

    plt.figure()
    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(234)
    plt.plot(hist)
    plt.plot(cdfNorm,color='b')

    # equalize img
    plt.subplot(235)
    equImg= cv.equalizeHist(img)
    eqhist = cv.calcHist([equImg], [0], None, [256], [0, 256])
    eqcdf = hist.cumsum()
    eqcdfNorm = cdf * float(eqhist.max()) / eqcdf.max()
    plt.imshow(equImg, cmap='gray')

    plt.subplot(236)
    plt.plot(eqhist)
    plt.plot(eqcdfNorm,color='b') 
    plt.show()

if __name__ == "__main__":
    histogramEqual()
