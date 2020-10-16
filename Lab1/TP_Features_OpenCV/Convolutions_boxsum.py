import numpy as np
import cv2

from matplotlib import pyplot as plt

#Read grayscale image and conversion to float64
img=np.float64(cv2.imread('../Image_Pairs/FlowerGarden2.png',0))
(h,w) = img.shape
print("Image dimension:",h,"rows x",w,"columns")

plt.subplot(121)
plt.imshow(img,cmap = 'gray')
plt.title('Original')

#Set the size of the box
kernel_size = 5

def boxsum_kernel(kernel_size):

    kernel = np.ones((kernel_size,kernel_size))
    
    return kernel

kernel = boxsum_kernel(kernel_size)

img3 = cv2.filter2D(img,-1,kernel)

plt.subplot(122)

#Expand the value range to size^2 times the original to get a correct display
plt.imshow(img3,cmap = 'gray',vmin = 0.0,vmax = 255.0*(kernel_size**2))
plt.title('Convolution - filter2D')

plt.show()
