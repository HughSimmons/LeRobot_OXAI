import cv2

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if ret0:
        cv2.imshow("camera0", frame0)
    if ret1:
        cv2.imshow("camera1)", frame1)

    if ret2:
        cv2.imshow("camera2 )", frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release()
cap1.release()
cap2.release()
cv2.destroyAllWindows()