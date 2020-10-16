import numpy as np
import cv2

from matplotlib import pyplot as plt

import sys
if len(sys.argv) != 2:
  print ("Usage:",sys.argv[0],"detector(= orb or kaze)")
  sys.exit(2)
if sys.argv[1].lower() == "orb":
  detector = 1
elif sys.argv[1].lower() == "kaze":
  detector = 2
else:
  print ("Usage:",sys.argv[0],"detector(= orb or kaze)")
  sys.exit(2)

#Read the image pair
img1 = cv2.imread('../Image_Pairs/torb_small1.png')
print("Dimension of image 1:",img1.shape[0],"rows x",img1.shape[1],"columns")
print("Type of image 1:",img1.dtype)
img2 = cv2.imread('../Image_Pairs/torb_small2.png')
print("Dimension of image 2:",img2.shape[0],"rows x",img2.shape[1],"columns")
print("Type of image 2:",img2.dtype)

#Beggining the calculus...
t1 = cv2.getTickCount()
#Creation of objects "keypoints"
if detector == 1:
  kp1 = cv2.ORB_create(nfeatures = 500,#By default : 500
                       scaleFactor = 1.2,#By default : 1.2
                       nlevels = 8)#By default : 8
  kp2 = cv2.ORB_create(nfeatures=500,
                       scaleFactor = 1.2,
                       nlevels = 8)
  print("Detector: ORB")
else:
  kp1 = cv2.KAZE_create(upright = False,#By default : false
    		        threshold = 0.001,#By default : 0.001
  		            nOctaves = 4,#By default : 4
		            nOctaveLayers = 4,#By default : 4
		            diffusivity = 2)#By default : 2
  kp2 = cv2.KAZE_create(upright = False,#By default : false
	  	        threshold = 0.001,#By default : 0.001
		        nOctaves = 4,#By default : 4
		        nOctaveLayers = 4,#By default : 4
		        diffusivity = 2)#By default : 2
  print("Detector: KAZE")
#Conversion to grayscale
gray1 =  cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 =  cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#Keypoints detection

pts1,des1 = kp1.detectAndCompute(gray1,None)
pts2,des2 = kp2.detectAndCompute(gray2,None)

print(len(des1))

t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Detection of key points:",time,"s")

#Displaying the keypoints
img1 = cv2.drawKeypoints(gray1, pts1, None, flags=4)
# flags defines the information level on key points
# 0: position only; 4: position + scale + direction
img2 = cv2.drawKeypoints(gray2, pts2, None, flags=4)

plt.figure()
plt.subplot(121)
plt.imshow(img1)
plt.title('Image n°1')

plt.subplot(122)
plt.imshow(img2)
plt.title('Image n°2')

if detector == 1:
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
else:
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)

repeatability = np.float(len(matches))/min(np.float(des1.shape[0]),np.float(des2.shape[0]))
print('repeatability =', repeatability)

matchImg = cv2.drawMatches(gray1, pts1, gray2, pts2, matches, gray2, flags=2)

plt.figure()

plt.imshow(matchImg)
plt.title('Matching result')
plt.show()