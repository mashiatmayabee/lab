import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

def main():
    img_path = '/home/mayabee/Documents/lab-2/pea.jpg'
    img_original = plt.imread(img_path)
    gray =cv2.cvtColor(img_original,cv2.COLOR_RGB2GRAY)
    gray1 = graysc(img_original)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    r,c = gray.shape
    noisy_img = gray1
    for i in range(c*r//10):
        x = random.randint(0, r-1)
        y = random.randint(0, c-1)
        flag = random.randint(0,2)
        if(flag == 0):
            noisy_img[x,y] = 0
        else:
            noisy_img[x,y] = 255
    
    fig_size = 5
    res = cv2.medianBlur(noisy_img, fig_size)
   
    img_set = [img_original, gray, gray1, noisy_img, res]
    title_set = [ 'Original', 'gray', 'made grayscale', 'Noisy', 'result']
    plot_img(img_set,title_set)

def graysc(img):
    r,c ,l= img.shape
    gray = np.zeros((r,c), dtype = np.uint8)
    
    
    for i in range(r):
        for j in range(c):
            gray[i,j] = 0.144*img[i,j,0]+0.587*img[i,j,1]+0.299*img[i,j,2]
            
    return gray
    
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
    plt.savefig('morph.jpg')
    plt.show()
    
if __name__ == '__main__':
    main()