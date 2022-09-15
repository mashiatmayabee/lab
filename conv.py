import matplotlib.pyplot as plt
import numpy as np
import cv2

def main():
    img_path = '/home/mayabee/Documents/lab-2/pea.jpg'
    img_original = plt.imread(img_path)
    gray =cv2.cvtColor(img_original,cv2.COLOR_RGB2GRAY)
    gray1 = graysc(img_original)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    # histogram_gen(gray)
    kernel = np.array([[0,-1,0], [-1,4,-1], [0,-1,0]])
    pr1 = cv2.filter2D(gray, -1, kernel)
    pr2 = conv(gray, kernel)
    img_set = [img_original, binary, gray, gray1, pr1, pr2]
    title_set = [ 'Original','binary', 'gray', 'made grayscale', 'kenel', 'made']
    plot_img(img_set,title_set)

def conv(img, kernel):
    r,c = img.shape
    img = np.pad(img, 3//2, constant_values = 0)
    for i in range(r):
        for j in range(c):
            res = np.sum(img[i:i+3, j:j+3]* kernel)
            if res > 255:
                res = 255
            if res <=0:
                res = 0
            res = np.rint(res)
            img[i,j] = res
            
    return img

            
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