import cv2 as cv 
import numpy as np
import sys

cap = cv.VideoCapture(0)
frame_num = 0
lower_hue = int(sys.argv[1])
lower_sat = int(sys.argv[2])
lower_val = int(sys.argv[3])
upper_hue = int(sys.argv[4])
upper_sat = int(sys.argv[5])
upper_val = int(sys.argv[6])
while cap.isOpened():

    if frame_num == 0:
        ret, original_frame = cap.read()
        original_image = cv.flip(original_frame, 1)
        cv.waitKey(5000)

    ret, frame = cap.read()
    image = cv.flip(frame, 1)

    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    img_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    ret, thresh = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)

    lower_colour_value = np.array([lower_hue, lower_sat, lower_val])
    upper_colour_value = np.array([upper_hue, upper_sat, upper_val])

    mask = cv.inRange(img_hsv, lower_colour_value, upper_colour_value)

    res = cv.bitwise_and(image, image, mask = mask)

    erode = cv.erode(mask, None, iterations = 2)
    dilate = cv.dilate(erode, None, iterations = 2)

    guassian_blur = cv.GaussianBlur(dilate,(3,3),0)

    median = cv.medianBlur(guassian_blur, 5)

    img_copy = image.copy()
    img_copy_2 = image.copy()

    _, cnts, _= cv.findContours(image = median, mode = cv.RETR_TREE, method = cv.CHAIN_APPROX_SIMPLE)

    if cnts:
        c = max(cnts, key = cv.contourArea)
        (x,y,w,h) = cv.boundingRect(c)  
        
        img_copy_2 = original_image[y:y+h, x:x+w]
        img_copy[y:y+h, x:x+w] = img_copy_2


    cv.imshow("invisible", img_copy)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    frame_num += 1

cap.release()
cv.destroyAllWindows()