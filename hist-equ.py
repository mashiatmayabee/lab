import cv2
import numpy as np
import matplotlib.pyplot as plt
def main():
  path = "/home/mayabee/Documents/lab-2/aurora.jpg"
  img=plt.imread(path)

  gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  r,c = gray.shape
  res = hist_equ(gray)
  y = range(0,256)

  plt.subplot(2,2,2)
  
  histogram = cv2.calcHist([gray],[0],None, [256],[0,256])
  plt.plot(y, histogram)

  plt.subplot(2,2,1)
  plt.imshow(gray, cmap = 'gray')
  hi = cv2.calcHist([res],[0],None, [256],[0,256])
  plt.subplot(2,2,3)
  plt.imshow(res,cmap = 'gray')
  plt.subplot(2,2,4)
  plt.plot(y, hi)
  plt.show()

def hist_equ(img):
  l = 255
  histogram = cv2.calcHist([img],[0],None, [256],[0,256])
  print(histogram)
  CDF = histogram.cumsum()
  CDF_min = CDF.min()
  r,c = img.shape
  size = r * c
  result = np.zeros((r,c) ,dtype = np.uint8)
  for i in range(r):
    for j in range(c):
      result[i,j] = ((CDF[img[i,j]] - CDF_min) / (size - CDF_min)) * l
  return result;

if __name__ == "__main__":
  main()