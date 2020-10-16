import numpy as np
import cv2

from matplotlib import pyplot as plt

#Reading grayscale image and conversion to float64
img=np.float32(cv2.imread('../Image_Pairs/Graffiti0.png',cv2.IMREAD_GRAYSCALE))
(h,w) = img.shape
print("Dimension of image:",h,"rows x",w,"columns")
print("Type of image:",img.dtype)

#Beginning of calculus
t1 = cv2.getTickCount()
Theta = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
# Put here Harris interest function calculation
#
#

def cornerHarris(img,k,threshold):
    h,w = img.shape
    
    # compute Ix,Iy
    grad = np.zeros([h, w, 2], dtype=np.float32)
    grad[:, 1:-1, 0] = img[:, 2:] - img[:, :-2]  # Ix
    grad[1:-1, :, 1] = img[2:, :] - img[:-2, :]  # Iy
    
    # compute Ixx,Iyy,Ixy
    m = np.empty([h, w, 3], dtype=np.float)
    
    #compute the auto-correlation matrix with window weights = 1
    weight1_kernel = np.ones((5,5),np.float)
    m[:, :, 0] = cv2.filter2D(grad[:, :, 0]**2,-1,weight1_kernel)
    m[:, :, 1] = cv2.filter2D(grad[:, :, 1]**2,-1,weight1_kernel)
    m[:, :, 2] = cv2.filter2D(grad[:, :, 0]*grad[:, :, 1],-1,weight1_kernel)

    #compute the auto-correlation matrix with window gaussian.
    #m[:, :, 0] = cv2.GaussianBlur(grad[:, :, 0]**2, (5, 5), sigmaX=2)  # Ixx
    #m[:, :, 1] = cv2.GaussianBlur(grad[:, :, 1]**2, (5, 5), sigmaX=2)  # Iyy
    #m[:, :, 2] = cv2.GaussianBlur(grad[:, :, 0]*grad[:, :, 1], (5, 5), sigmaX=2)  # Ixy
    m = [np.array([[m[i, j, 0], m[i, j, 2]], [m[i, j, 2], m[i, j, 1]]]) for i in range(h) for j in range(w)]

    D = np.array(list(map(np.linalg.det, m)))
    T = np.array(list(map(np.trace, m)))
    R = D-k*np.power(T,2)  
    R_max = R.max()
   
    Theta = np.zeros_like(R, dtype=np.float32)
    Theta[R > R_max*threshold] = R[R > R_max*threshold] / R_max * 255.
     
    return Theta.reshape(h,w)

Theta = cornerHarris(img,0.06,0.01)
# Computing local maxima and thresholding
Theta_maxloc = cv2.copyMakeBorder(Theta,0,0,0,0,cv2.BORDER_REPLICATE)
d_maxloc = 3
seuil_relatif = 0.01
se = np.ones((d_maxloc,d_maxloc),np.uint8)
Theta_dil = cv2.dilate(Theta,se)
#Suppression of non-local-maxima
Theta_maxloc[Theta < Theta_dil] = 0.0
#Values to small are also removed
Theta_maxloc[Theta < seuil_relatif*Theta.max()] = 0.0
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("My computation of Harris points:",time,"s")
print("Number of cycles per pixel:",(t2 - t1)/(h*w),"cpp")


plt.figure(figsize=(16,6))
plt.subplot(131)
plt.imshow(img,cmap = 'gray')
plt.title('Original image')

plt.subplot(132)
plt.imshow(Theta,cmap = 'gray')
plt.title('Harris function')

se_croix = np.uint8([[1, 0, 0, 0, 1],
[0, 1, 0, 1, 0],[0, 0, 1, 0, 0],
[0, 1, 0, 1, 0],[1, 0, 0, 0, 1]])
Theta_ml_dil = cv2.dilate(Theta_maxloc,se_croix)
#Re-read image for colour display
Img_pts=cv2.imread('../Image_Pairs/Graffiti0.png',cv2.IMREAD_COLOR)
(h,w,c) = Img_pts.shape
print("Dimension of image:",h,"rows x",w,"columns x",c,"channels")
print("Type of image:",Img_pts.dtype)
#Points are displayed as red crosses
Img_pts[Theta_ml_dil > 0] = [255,0,0]
plt.subplot(133)
plt.imshow(Img_pts)
plt.title('Harris points')

plt.show()
