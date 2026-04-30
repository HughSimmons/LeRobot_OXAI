import cv2
import numpy as np

def detect_and_annotate(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    output = frame.copy()

    # 🔴 RED MASK (cube)
    lower_red1 = np.array([0, 150, 120])
    upper_red1 = np.array([5, 255, 255])
    lower_red2 = np.array([175, 150, 120])
    upper_red2 = np.array([180, 255, 255])

    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + \
               cv2.inRange(hsv, lower_red2, upper_red2)

    # 🔵 BLUE MASK (gripper)
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Clean masks
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel)

    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_DILATE, kernel)

    # 🔴 Find cube
    cube_center = None
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cube_center = (x + w // 2, y + h // 2)

            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(output, cube_center, 5, (0, 0, 255), -1)
            break

    # 🔵 Find gripper (blue)
    gripper_center = None
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 300:
            x, y, w, h = cv2.boundingRect(cnt)
            gripper_center = (x + w // 2, y + h // 2)

            cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.circle(output, gripper_center, 5, (255, 0, 0), -1)

    # 📏 Distance + line
    distance = None
    if cube_center is not None and gripper_center is not None:
        dx = cube_center[0] - gripper_center[0]
        dy = cube_center[1] - gripper_center[1]
        distance = np.sqrt(dx**2 + dy**2)

        cv2.line(output, gripper_center, cube_center, (0, 255, 255), 2)

        cv2.putText(output, f"{int(distance)} px",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

    return output, cube_center, gripper_center, distance


if __name__=="__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output, cube, gripper, dist = detect_and_annotate(frame)

        cv2.imshow("Detection", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()