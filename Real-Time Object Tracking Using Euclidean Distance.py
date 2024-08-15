import cv2
import numpy as np
from scipy.spatial import distance as dist


# Custom tracker class using Euclidean distance
class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        self.id_count = 0

    def update(self, objects_rect):
        # Objects' bounding boxes and IDs to be returned
        objects_bbs_ids = []

        # Get center point of the new objects
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Check if this object is already being tracked
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist_between_points = dist.euclidean((cx, cy), pt)

                if dist_between_points < 25:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # If this is a new object, assign a new ID
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean up center points that are no longer tracked
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with new points
        self.center_points = new_center_points.copy()
        return objects_bbs_ids


# Initialize the tracker object
tracker = EuclideanDistTracker()

# Video capture
cap = cv2.VideoCapture(r"C:\Users\VENKATA SURYA\OneDrive\Documents\highway.mp4")

# Object detection using background subtraction
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Define the Region of Interest (ROI)
    roi = frame[340:720, 500:800]

    # 1. Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Display the ROI, original frame, and mask
    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # Break loop on 'Esc' key press
    key = cv2.waitKey(30)
    if key == 27:
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

'''
Explanation of code
This project implements a real-time object tracking system using OpenCV and a custom tracker based on Euclidean distance.
The video is processed frame by frame, with objects detected using background subtraction (createBackgroundSubtractorMOG2). 
The Region of Interest (ROI) is defined within each frame, where object detection is performed.
Contours are extracted from the binary mask to identify moving objects, which are then tracked across frames using their center points. 
Each object is assigned a unique ID, which remains consistent as long as the object is tracked. 
The system visualizes the tracking process by drawing bounding boxes around objects and displaying their IDs on the video frames.
'''