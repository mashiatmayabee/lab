from turtle import circle, shape
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    rgbImg = plt.imread('/home/mayabee/Documents/lab-2/pea.jpg')
    print(rgbImg.shape)
    gray = cv2.cvtColor(rgbImg, cv2.COLOR_RGB2GRAY)
    fftImg = np.fft.fft2(gray)
    centeredFti = np.fft.fftshift(fftImg)
    mag_spec = 100 * np.log(np.abs(fftImg)
    centered_mag_spec = 100 * np.log(np.abs(centeredFti))
    r,c = gray.shape
    while = np.ones((r,c), dtype = np.uint8)
    filter =cv2.circle()
    
    img_set = [rgbImg, gray, magnitude_spectrum, centered_magnitude_spectrum, filter4, filtered_img]
    title_set = ['RGB', 'Gray', 'FFT2', 'Centered FFT2', 'Filter', 'Filtered Img']

    plot_img(img_set,title_set)

def plot_img(img_set, title_set):		
    plt.figure(figsize = (20, 20))
    n = len(img_set)
    for i in range(n):
        plt.subplot(2, 3, i + 1)
        plt.title(title_set[i])
        img = img_set[i]
        ch = len(img.shape)
        if (ch == 2):
            plt.imshow(img, cmap = 'gray')
        else:
            plt.imshow(img)			
    plt.show()





if __name__ == '__main__':
    main()