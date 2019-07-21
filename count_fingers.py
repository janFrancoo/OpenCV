import cv2
import numpy as np
from sklearn.metrics import pairwise

bg = None
cam = cv2.VideoCapture(0)

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)

    roi = frame[20:400, 50:350]
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    roi = cv2.bilateralFilter(roi, 15, 75, 75)
    if bg is None:
        bg = roi.copy()
    roi = cv2.absdiff(bg, roi)
    _, roi = cv2.threshold(roi, 15, 255, cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        handSegment = max(contours, key=cv2.contourArea)
        convHull = cv2.convexHull(handSegment)
        top = tuple(convHull[convHull[:, :, 1].argmin()][0])
        bottom = tuple(convHull[convHull[:, :, 1].argmax()][0])
        left = tuple(convHull[convHull[:, :, 0].argmin()][0])
        right = tuple(convHull[convHull[:, :, 0].argmax()][0])
        cX = (left[0] + right[0]) // 2
        cY = (top[1] + bottom[1]) // 2
        distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]
        max_distance = distance.max()
        radius = int(0.8 * max_distance)
        circumference = (2 * np.pi * radius)
        circular_roi = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.circle(circular_roi, (cX, cY), radius, 255, 10)
        circular_roi = cv2.bitwise_and(roi, roi, mask=circular_roi)
        image, contours, hierarchy = cv2.findContours(circular_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        count = 0
        for cnt in contours:
            (x,y,w,h) = cv2.boundingRect(cnt)
            out_of_wrist = ((cY + (cY * 0.25)) > (y + h))
            limit_points = ((circumference * 0.25) > cnt.shape[0])
            if out_of_wrist and limit_points:
                count += 1
        cv2.putText(frame, str(count), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.rectangle(frame, (50, 20), (350, 400), (0, 0, 255), 5)
    cv2.imshow('Camera', frame)
    cv2.imshow('Segment', roi)
    cv2.imwrite('frame.jpg', frame)
    cv2.imwrite('roi.jpg', roi)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
