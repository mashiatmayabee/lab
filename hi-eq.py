import cv2
import numpy as np
import matplotlib.pyplot as plt
def main():
  path = "/home/mayabee/Documents/lab-2/aurora.jpg"
  img=plt.imread(path)

  gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  r,c = gray.shape
#   res = hist_equ(gray)
  y = range(0,256)

  plt.subplot(2,2,2)
  
  histogram = cv2.calcHist([gray],[0],None, [256],[0,256])
  plt.plot(y, histogram)

  hi3 = hi(gray)
  plt.subplot(2,2,1)
  plt.imshow(gray, cmap = 'gray')
  hi21 = cv2.calcHist([hi3],[0],None, [256],[0,256])
  plt.subplot(2,2,3)
  plt.imshow(hi3,cmap = 'gray')
  plt.subplot(2,2,4)
  plt.plot(y, hi21)
  plt.show()

def hi(im):
    r,c = im.shape
    hist = cv2.calcHist([im], [0], None, [256], [0,256])
    cdf = hist.cumsum()
    cdf_min = cdf.min()
    size = r * c
    res = np.zeros((r,c), dtype = np.uint8)
    for i in range(r):
        for j in range(c):
            res[i,j] =((cdf[im[i,j]] - cdf_min) / (size - cdf_min))* 255
    return res

if __name__ == "__main__":
  main()