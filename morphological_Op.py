
import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    path = '/home/mayabee/Documents/lab-2/img3.jpg'
    imgo = plt.imread(path)
    
    r,c,l = imgo.shape
    structuring_element = np.ones((5,5), dtype = np.uint8)
    gray = cv2.cvtColor(imgo, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    
    er_img = mor_erode(binary, structuring_element);
    dil_img = mor_dilate(binary, structuring_element)
    op = opening(binary, structuring_element)
    cl = closing(binary, structuring_element)
    img_set = [ imgo, gray,binary, er_img, dil_img, op, cl]
    title_set = [ 'Image1','gray','Binary ', 'eroded', 'dilated', 'Opening', 'closing']
    plot_img(img_set,title_set)

def opening(img, element):
    r,c = img.shape
    img1 = mor_erode(img, element)
    img2 = mor_dilate(img1, element)
    return img2
def closing(img, element):
    r,c = img.shape
    img1 = mor_dilate(img, element)
    img2 = mor_erode(img1, element)
    return img2
    
def mor_dilate(img, element):
    r,c = img.shape
    m,n = element.shape
    p = (m//2)
    proc = np.zeros((r,c), dtype = np.int8)
    img = np.pad(img, p, constant_values = 0)
    
    for i in range(r):
        for j in range(c):
            res = np.sum(img[i:i+m, j:j+n] * element)
            if res > 0:
                proc[i,j] = 255
    return proc

def mor_erode(img, element):
    r,c = img.shape
    m,n = element.shape
    p = (m//2)
    proc = np.zeros((r,c), dtype = np.int8)
    img = np.pad(img, p, constant_values = 0)
    
    for i in range(r):
        for j in range(c):
            res = np.sum(img[i:i+m, j:j+n] * element)
            if res  == 255*m*n:
                proc[i,j] = 255
    return proc


def plot_img(img_set, title_set):
    n=len(img_set)
    plt.figure(figsize= (20,20))

    for i in range(n):
        img = img_set[i]
        ch = len(img.shape)
        plt.subplot(2,4,i+1)
        if(ch == 3):
            plt.imshow(img_set[i])
        else:
            plt.imshow(img_set[i], cmap = 'gray')
        plt.title(title_set[i])    
    plt.savefig('binary_masking.jpg')
    plt.show()
  

    
if __name__ == '__main__':
    main()