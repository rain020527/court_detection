import cv2
import numpy as np
 
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    return cv2.LUT(image, table)

def CannyThreshold(lowThreshold):
    # detected_edges = cv2.GaussianBlur(gray,(3,3),0)

    blur = cv2.GaussianBlur(gray,(5, 5),0).astype(int)
    sub = gray.astype(int) - blur
    detected_edges = np.clip(gray.astype(int) + sub*2, a_min = 0, a_max = 255).astype('uint8')

    detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
    dst = cv2.bitwise_and(img,img,mask = detected_edges)  # just add some colours to edges from original image.


    lines = cv2.HoughLinesP(detected_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(dst,(x1,y1),(x2,y2),(0,0,254),1)
    cv2.imshow('canny demo',dst)
 
lowThreshold = 85
max_lowThreshold = 100
ratio = 3
kernel_size = 3

rho = 1  # distance resolution in pixels of the Hough grid (fixed)
theta = np.pi / 180  # angular resolution in radians of the Hough grid (fixed)

threshold = 30  # minimum number of votes (intersections in Hough grid cell) (doesnt matter)
min_line_length = 75  # minimum number of pixels making up a line
max_line_gap = 5  # maximum gap in pixels between connectable line segments
max_threshold = 150

img = cv2.imread('corner3.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = adjust_gamma(gray, 0.4)
 
cv2.namedWindow('canny demo')
 
cv2.createTrackbar('Min threshold','canny demo',lowThreshold, max_lowThreshold, CannyThreshold)
# cv2.createTrackbar('Min threshold','canny demo',threshold, max_threshold, CannyThreshold)
 
CannyThreshold(0)  # initialization
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()