from cv2 import bitwise_and
import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
    img_path1 = '/home/mayabee/Documents/lab-2/pea.jpg'
    img1 = plt.imread(img_path1)    
    gray1 =cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
    _, binary1 = cv2.threshold(gray1, 50, 255, cv2.THRESH_BINARY)

    img_path2 = 'G:/4y1s/DIP_Lab/Assignment-5/img2.jpg'
    img2 = plt.imread(img_path2)
    gray2 =cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
    _, binary2 = cv2.threshold(gray2, 50, 255, cv2.THRESH_BINARY)
    print(gray1.shape)
    print(gray2.shape)
    

    processed_img = cv2.bitwise_and(binary2,gray1)
    img_set = [ img1, img2,binary1, gray2 ,processed_img]
    title_set = [ 'Image1','Image2','Binary of image1', 'Grayscale of image 2', 'Binary mask on grayscale']
    plot_img(img_set,title_set)

def plot_img(img_set, title_set):
    n=len(img_set)
    plt.figure(figsize= (20,20))

    for i in range(n):
        img = img_set[i]
        ch = len(img.shape)
        plt.subplot(2,3,i+1)
        if(ch == 3):
            plt.imshow(img_set[i])
        else:
            plt.imshow(img_set[i], cmap = 'gray')
        plt.title(title_set[i])    
    plt.savefig('binary_masking.jpg')
    plt.show()
  
    
    
if __name__== '__main__':
    main()