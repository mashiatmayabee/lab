import matplotlib.pyplot as plt
import numpy as np
import cv2

def main():
    img_path = '/home/mayabee/Documents/lab-2/pea.jpg'
    img_original = plt.imread(img_path)
    gray =cv2.cvtColor(img_original,cv2.COLOR_RGB2GRAY)
    gray1 = graysc(img_original)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    histogram_gen(gray)
    img_set = [img_original, binary, gray, gray1]
    title_set = [ 'Original','binary', 'gray', 'made grayscale']
    plot_img(img_set,title_set)

def histogram_gen(img):
    r,c = img.shape
    
    h = np.zeros(256)
    n=range(0, 256)

    for i in range(r):
        for j in range(c):
            h[img[i,j]] = h[img[i,j]]+1
            
    plt.stem(n,h)
    plt.show()
            
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
    
def binary(img):
    r,c = img.shape
    for i in range r:
        for j in range c:
if __name__ == '__main__':
    main()