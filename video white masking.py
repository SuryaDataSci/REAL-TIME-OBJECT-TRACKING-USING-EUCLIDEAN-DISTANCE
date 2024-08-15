import cv2

# Initialize video capture
cap = cv2.VideoCapture(r"C:\Users\VENKATA SURYA\OneDrive\Documents\highway.mp4")

# Object detection from a stable camera using background subtraction
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# Loop through video frames
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Apply the background subtractor to get the mask
    mask = object_detector.apply(frame)

    # Display the original frame and the mask
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)

    # Check for key press, exit on 'Esc' key (key code 27)
    key = cv2.waitKey(30)
    if key == 27:
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
