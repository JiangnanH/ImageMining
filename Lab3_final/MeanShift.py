import numpy as np
import cv2

roi_defined = False

def define_ROI(event, x, y, flags, param):
    global r,c,w,h,roi_defined
    # if the left mouse button was clicked,
    # record the starting ROI coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        r, c = x, y
        roi_defined = False
    # if the left mouse button was released,
    # record the ROI coordinates and dimensions
    elif event == cv2.EVENT_LBUTTONUP:
        r2, c2 = x, y
        h = abs(r2-r)
        w = abs(c2-c)
        r = min(r,r2)
        c = min(c,c2)
        roi_defined = True

cap = cv2.VideoCapture('./Antoine_Mug.mp4')

# take first frame of the video
ret,frame = cap.read()
# load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("First image", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the ROI is defined, draw it!
    if (roi_defined):
        # draw a green rectangle around the region of interest
        cv2.rectangle(frame, (r,c), (r+h,c+w), (0, 255, 0), 2)
    # else reset the image...
    else:
        frame = clone.copy()
    # if the 'q' key is pressed, break from the loop
    if key == ord("q"):
        break

track_window = (r,c,h,w)
# set up the ROI for tracking
roi = frame[c:c+w, r:r+h]
# conversion to Hue-Saturation-Value space
# 0 < H < 180 ; 0 < S < 255 ; 0 < V < 255
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# computation mask of the histogram:
# Pixels with S<30, V<20 or V>235 are ignored
mask = cv2.inRange(hsv_roi, np.array((0.,30.,20.)), np.array((180.,255.,235.)))
# Marginal histogram of the Hue component
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
# Histogram values are normalised to [0,255]
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria: either 10 iterations,
# or move by less than 1 pixel
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

########### Start Q3 part-1 ##########
## function for computing gradient magnitude.
#def gradient(img):
#    
#    threshold = 30
#    
#    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    dx = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
#    dy = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
#    absX = cv2.convertScaleAbs(dx)
#    absY = cv2.convertScaleAbs(dy)
#    
#    dx[:,:][absX < threshold] = 0
#    dy[:,:][absY < threshold] = 0
#    absX[:,:][absX < threshold] = 0
#    absY[:,:][absY < threshold] = 0
#    
#    grad = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
#    orientation = [dx,dy]
#    return grad, orientation
###########  End Q3 part-1  ##########
   
    
########### Start Q4 part-1 ##########
## compute R-table
#roi_grad,roi_orientation = gradient(roi)
#cv2.imshow("R-table", roi_grad)
###########  End Q4 part-1  ##########
    
cpt = 1
while(1):
    ret ,frame = cap.read()

    if ret == True:
        
#        ########## Start Q3 part-2 ##########
#        # compute gradient
#        frame_grad,frame_orientation = gradient(frame)
#        frame_mask = np.copy(frame)
#        cv2.imshow('gradient magnitude',frame_grad)
#         
#        # mask the pixels which have gradients lower than threshold with red color 
#        frame_mask[:,:][frame_grad == 0] = [0,0,255]
#        frame_mask[:,:][frame_grad > 0] = [255,255,255]
#        cv2.imshow('gradient magnitude frame',frame_mask)
#        ##########  End Q3 part-2  ##########
    
    
#        ########## Start Q4 part-2 ##########
#        # implement Hough Transform (which works very slowly..)
#        H = np.zeros_like(frame_grad)
#        for x in range(1, frame_grad.shape[0]-w):
#            for y in range(1, frame_grad.shape[1]-h):
#                
#                pixel_roi_grad = frame_grad[x:x+w, y:y+h]
#                pixel_roi_dx = frame_orientation[0][x:x+w, y:y+h]
#                pixel_roi_dy = frame_orientation[1][x:x+w, y:y+h]
#                
#                H_roi = np.copy(pixel_roi_grad)
#                # set a threshold of 80 to determine if a gradient is similar to the correspond gradient orientation in the R-table.
#                H_roi[:,:][(np.abs(pixel_roi_dx - roi_orientation[0]) + \
#                     np.abs(pixel_roi_dy - roi_orientation[1])) >= 80] = 0
#                # set the average gradient of H_roi as H
#                H[x+int(w/2),y+int(h/2)] = np.mean(H_roi)
#                
#        # directly using the argmax(x,y) of H as the position of the new window
#        position = np.argmax(H)
#        x_p = int(position%H.shape[1]) - int(w/2)
#        y_p = int(position/H.shape[1]) - int(h/2)
#        
#        # augmente the H to display it
#        cv2.imshow('Hough transform',H * 30)
#        ##########  End Q4 part-2  ##########
    
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # display hsv
        # cv2.imshow('hsv',hsv)

        # Backproject the model histogram roi_hist onto the
        # current image hsv, i.e. dst(x,y) = roi_hist(hsv(0,x,y))
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # display density
        cv2.imshow('dst',dst)
        # apply meanshift to dst to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        
        ########## Start Q5 ##########
        # perform the MeanShift algo on the output of Hough Transform
        # ret, track_window = cv2.meanShift(H, track_window, term_crit)
        ##########  End Q5  ##########
        
        # Draw a blue rectangle on the current image
        
        ########## Start Q4 part-3 ##########
        # window for mean shift algo
        r,c,h,w = track_window
        # window for directly argmax method
        # r,c = x_p,y_p
        ##########  End Q4 part-3  ##########
        
        frame_tracked = cv2.rectangle(frame, (r,c), (r+h,c+w), (255,0,0) ,2)
        cv2.imshow('Sequence',frame_tracked)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('Frame_%04d.png'%cpt,frame_tracked)
        cpt += 1
    else:
        break

cv2.destroyAllWindows()
cap.release()
