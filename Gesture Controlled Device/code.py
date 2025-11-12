import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from collections import deque
import time
import math

# --- SETTINGS ---
SMOOTHING = 5
ALPHA = 0.25
CALIB_POINTS = 4
CONF_THRESH = 0.7
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480
FRAME_DELAY = 0.001
PINCH_THRESHOLD = 0.045  # smaller = more sensitive

pyautogui.FAILSAFE = True
SAFE_MARGIN = 30

screen_w, screen_h = pyautogui.size()

# Initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=CONF_THRESH,
    min_tracking_confidence=CONF_THRESH
)
mp_draw = mp.solutions.drawing_utils

# --- CALIBRATION ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESIZE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESIZE_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 60)

print(">>> Calibration mode <<<")
print("Point to 4 screen corners (TL, TR, BR, BL) in sequence.")
print("Press 'c' to capture each corner, 'q' to quit.")

calib_points_cam = []
corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]

while len(calib_points_cam) < CALIB_POINTS:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    cv2.putText(frame, f"Show {corner_names[len(calib_points_cam)]} corner and press 'c'",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark[8]
        cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

    cv2.imshow("Calibration", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c') and results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark[8]
        calib_points_cam.append([lm.x, lm.y])
        print(f"Captured {corner_names[len(calib_points_cam) - 1]} corner.")
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

cam_pts = np.array(calib_points_cam, dtype=np.float32)
screen_pts = np.array([[0, 0],
                       [screen_w, 0],
                       [screen_w, screen_h],
                       [0, screen_h]], dtype=np.float32)
transform, _ = cv2.findHomography(cam_pts, screen_pts)
print("Calibration done! Transform matrix:\n", transform)

# --- CONTROL MODE ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESIZE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESIZE_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 60)
print(">>> Control mode started. Press 'q' to exit <<<")

buf_x, buf_y = deque(maxlen=SMOOTHING), deque(maxlen=SMOOTHING)
prev_x, prev_y = 0, 0
drag_mode = False
click_down = False

def distance(lm1, lm2):
    return math.hypot(lm1.x - lm2.x, lm1.y - lm2.y)

while True:
    start_time = time.time()
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        lm = hand.landmark
        idx_tip, thumb_tip, mid_tip = lm[8], lm[4], lm[12]

        # Distances
        pinch_dist = distance(idx_tip, thumb_tip)
        two_finger_dist = distance(mid_tip, thumb_tip)

        # Map index finger to screen
        cam_coord = np.array([[idx_tip.x, idx_tip.y, 1]], dtype=np.float32).T
        mapped = transform @ cam_coord
        mapped /= mapped[2][0]
        x_s, y_s = mapped[0][0], mapped[1][0]

        buf_x.append(x_s)
        buf_y.append(y_s)

        avg_x, avg_y = np.mean(buf_x), np.mean(buf_y)
        smooth_x = prev_x + ALPHA * (avg_x - prev_x)
        smooth_y = prev_y + ALPHA * (avg_y - prev_y)

        sx = max(SAFE_MARGIN, min(screen_w - SAFE_MARGIN, int(smooth_x)))
        sy = max(SAFE_MARGIN, min(screen_h - SAFE_MARGIN, int(smooth_y)))

        pyautogui.moveTo(sx, sy, duration=0)
        prev_x, prev_y = smooth_x, smooth_y

        # --- Gestures ---
        if pinch_dist < PINCH_THRESHOLD and not click_down:
            pyautogui.mouseDown()
            click_down = True
            drag_mode = True
            cv2.putText(frame, "Left Click / Drag", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        elif pinch_dist > PINCH_THRESHOLD and click_down:
            pyautogui.mouseUp()
            click_down = False
            drag_mode = False

        elif two_finger_dist < PINCH_THRESHOLD:
            pyautogui.click(button='right')
            cv2.putText(frame, "Right Click", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            time.sleep(0.2)

        cv2.putText(frame, f"Cursor: ({sx},{sy})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Gesture Control Debug", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elapsed = time.time() - start_time
    if elapsed < FRAME_DELAY:
        time.sleep(FRAME_DELAY - elapsed)

cap.release()
cv2.destroyAllWindows()
