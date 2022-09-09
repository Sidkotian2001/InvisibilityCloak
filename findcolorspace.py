#finding hsv range of target object(pen)
import cv2 as cv
import numpy as np
import time
# A required callback method that goes into the trackbar function.
def nothing(x):
    pass

def colour_tracker(cap):
    # Initializing the webcam feed.
    cap.set(3,1280)
    cap.set(4,720)

    # Create a window named trackbars.
    cv.namedWindow("Trackbars")

    # Now create 6 trackbars that will control the lower and upper range of 
    # H,S and V channels. The Arguments are like this: Name of trackbar, 
    # window name, range,callback function. For Hue the range is 0-179 and
    # for S,V its 0-255.
    cv.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    cv.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    cv.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
    
    while True:
        
        # Start reading the webcam feed frame by frame.
        ret, frame = cap.read()
        if not ret:
            break
        # Flip the frame horizontally (Not required)
        frame = cv.flip( frame, 1 ) 
        
        # Convert the BGR image to HSV image.
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        
        # Get the new values of the trackbar in real time as the user changes 
        # them
        l_h = cv.getTrackbarPos("L - H", "Trackbars")
        l_s = cv.getTrackbarPos("L - S", "Trackbars")
        l_v = cv.getTrackbarPos("L - V", "Trackbars")
        u_h = cv.getTrackbarPos("U - H", "Trackbars")
        u_s = cv.getTrackbarPos("U - S", "Trackbars")
        u_v = cv.getTrackbarPos("U - V", "Trackbars")
    
        # Set the lower and upper HSV range according to the value selected
        # by the trackbar
        lower_range = np.array([l_h, l_s, l_v])
        upper_range = np.array([u_h, u_s, u_v])
        
        # Filter the image and get the binary mask, where white represents 
        # your target color
        mask = cv.inRange(hsv, lower_range, upper_range)
    
        # You can also visualize the real part of the target color (Optional)
        res = cv.bitwise_and(frame, frame, mask=mask)
        
        # Converting the binary mask to 3 channel image, this is just so 
        # we can stack it with the others
        mask_3 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        
        # stack the mask, orginal frame and the filtered result
        stacked = np.hstack((mask_3,frame,res))
        
        # Show this stacked frame at 40% of the size.
        cv.imshow('Trackbars',cv.resize(stacked,None,fx=0.4,fy=0.4))
        
        # If the user presses ESC then exit the program
        key = cv.waitKey(1)
        
        if key & 0xFF == ord('q'):
            break
        
        # If the user presses `s` then print this array.
        if key == ord('s'):
            
            thearray = [[l_h,l_s,l_v],[u_h, u_s, u_v]]
            print(thearray)
            
            # Also save this array as penval.npy
            np.save('hsv_value',thearray)
            break
 


def main():
    cap = cv.VideoCapture(0)
    colour_tracker(cap)
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()