import cv2

from alerts.snapshot_capture import save_snapshot

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

if ret:
    path = save_snapshot(frame)
    print("Saved:", path)

cap.release()