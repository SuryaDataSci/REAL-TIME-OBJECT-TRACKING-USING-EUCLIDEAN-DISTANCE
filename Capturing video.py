import cv2
# Initialize video capture
cap = cv2.VideoCapture(r"C:\Users\VENKATA SURYA\OneDrive\Documents\highway.mp4")

# Loop through video frames
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Display the current frame
    cv2.imshow('Frame', frame)

    # Check for key press, exit on 'Esc' key (key code 27)
    key = cv2.waitKey(30)
    if key == 27:
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
