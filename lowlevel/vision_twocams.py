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

    # 🔵 Find gripper
    gripper_center = None
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 300:
            x, y, w, h = cv2.boundingRect(cnt)
            gripper_center = (x + w // 2, y + h // 2)

            cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.circle(output, gripper_center, 5, (255, 0, 0), -1)

    # 📏 Distance
    distance = None
    if cube_center is not None and gripper_center is not None:
        dx = cube_center[0] - gripper_center[0]
        dy = cube_center[1] - gripper_center[1]
        distance = np.sqrt(dx**2 + dy**2)

        cv2.line(output, gripper_center, cube_center, (0, 255, 255), 2)

    return output, cube_center, gripper_center, distance


def detect_and_annotate_cam2(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    output = frame.copy()

    # 🔴 RED (same as before)
    lower_red1 = np.array([0, 150, 120])
    upper_red1 = np.array([5, 255, 255])
    lower_red2 = np.array([175, 150, 120])
    upper_red2 = np.array([180, 255, 255])

    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + \
               cv2.inRange(hsv, lower_red2, upper_red2)

    # 🔵 MORE FORGIVING BLUE
    # lower_blue = np.array([90, 50, 50])    # ↓ lower S and V
    lower_blue = np.array([90, 60, 80])   # ↑ V threshold
    upper_blue = np.array([140, 255, 255]) # ↑ wider hue
    # lower_blue = np.array([80, 40, 40])
    # upper_blue = np.array([150, 255, 255])

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Clean masks
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel)

    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_DILATE, kernel)

    # --- same contour logic as before ---
    cube_center = None
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cube_center = (x + w // 2, y + h // 2)
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(output, cube_center, 5, (0, 0, 255), -1)
            break

    gripper_center = None
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 200:  # slightly lower threshold
            x, y, w, h = cv2.boundingRect(cnt)
            gripper_center = (x + w // 2, y + h // 2)
            cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.circle(output, gripper_center, 5, (255, 0, 0), -1)

    distance = None
    if cube_center is not None and gripper_center is not None:
        dx = cube_center[0] - gripper_center[0]
        dy = cube_center[1] - gripper_center[1]
        distance = np.sqrt(dx**2 + dy**2)
        cv2.line(output, gripper_center, cube_center, (0, 255, 255), 2)

    return output, cube_center, gripper_center, distance


if __name__ == "__main__":
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            break

        out0, cube0, grip0, dist0 = detect_and_annotate(frame0)
        # out1, cube1, grip1, dist1 = detect_and_annotate(frame1)
        out1, cube1, grip1, dist1 = detect_and_annotate_cam2(frame1)

        # Overlay distances separately
        if dist0 is not None:
            cv2.putText(out0, f"Cam0: {int(dist0)} px",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2)

        if dist1 is not None:
            cv2.putText(out1, f"Cam1: {int(dist1)} px",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2)

        # Combine views side-by-side
        combined = np.hstack((out0, out1))

        cv2.imshow("Dual Camera Detection", combined)

        # Also print for logging
        print(f"Cam0 distance: {dist0}, Cam1 distance: {dist1}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()