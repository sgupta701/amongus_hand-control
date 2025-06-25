import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import joblib
import time

# loading models
model = joblib.load('models/asl_mlp_model.pkl')
label_encoder = joblib.load('models/asl_label_encoder.pkl')
scaler = joblib.load('models/asl_scaler.pkl')

# setting up mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Finger Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Finger Control", 640, 480)

# states
last_predicted_label = None
debounce_counter = 0
DEBOUNCE_THRESHOLD = 7
prev_keys = set()
predicted_label = None
gesture_active = False
joystick_enabled = True  

# cooldown tracking
gesture_cooldowns = {}  
COOLDOWN_DURATION = 1.2  

def is_cooldown_over(label):
    now = time.time()
    last_time = gesture_cooldowns.get(label, 0)
    return now - last_time >= COOLDOWN_DURATION

def update_cooldown(label):
    gesture_cooldowns[label] = time.time()

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    h, w, _ = img.shape
    current_keys = set()
    key = cv2.waitKey(1) & 0xFF

    gesture_active = False

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        features = np.array(landmarks).reshape(1, -1)
        features_scaled = scaler.transform(features)

        predicted_index = model.predict(features_scaled)[0]
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]

        if predicted_label == last_predicted_label:
            debounce_counter += 1
        else:
            debounce_counter = 0
        last_predicted_label = predicted_label

        if debounce_counter == DEBOUNCE_THRESHOLD:
            debounce_counter = 0
            last_predicted_label = None
            gesture_active = True

            # Prevent repeat spam
            if is_cooldown_over(predicted_label):
                update_cooldown(predicted_label)

                if predicted_label == "ENTER":
                    pyautogui.press('enter')
                    print("ENTER")

                elif predicted_label == "P":
                    # P gesture to trigger vent and killing actions so that it saves time
                    pyautogui.press('space')      # vent
                    pyautogui.press('e')      # kill
                    
                    print("P => VENT + KILL")

                # to report, i am taking Y because it is easy and faster to detect
                elif predicted_label =='Y':
                    pyautogui.press('r')  # report

                    print("Y => REPORT")

                elif predicted_label == "F":
                    joystick_enabled = not joystick_enabled
                    print(f"Joystick {'ENABLED' if joystick_enabled else 'LOCKED'}")

                elif predicted_label.isalpha():
                    pyautogui.press(predicted_label.lower())
                    print(f"Typed: {predicted_label.lower()}")


    else:
        predicted_label = None
        debounce_counter = 0
        last_predicted_label = None

    if not gesture_active and joystick_enabled and results.multi_hand_landmarks:
        box_x_start = w // 2
        box_y_start = h // 2
        box_width = w // 2
        box_height = h // 2
        cv2.rectangle(img, (box_x_start, box_y_start), (w, h), (0, 255, 0), 2)

        x = int(hand_landmarks.landmark[8].x * w)
        y = int(hand_landmarks.landmark[8].y * h)

        # center 'idle' is 20% of joystick box size for smoother control
        center_zone_fraction = 0.2  
        center_w = int(box_width * center_zone_fraction)
        center_h = int(box_height * center_zone_fraction)

        center_x_start = box_x_start + (box_width - center_w) // 2
        center_y_start = box_y_start + (box_height - center_h) // 2
        center_x_end = center_x_start + center_w
        center_y_end = center_y_start + center_h

        cv2.rectangle(img, (center_x_start, center_y_start), (center_x_end, center_y_end), (255, 0, 0), 2)

        if (box_x_start <= x <= w) and (box_y_start <= y <= h):
            keys = []

            # check if inside the smaller idle zone
            if center_x_start <= x <= center_x_end and center_y_start <= y <= center_y_end:
                # Idle zone - no keys pressed
                keys = []
            else:
                # directions based on position relative to the smaller idle zone
                if y < center_y_start:
                    keys.append('w')
                elif y > center_y_end:
                    keys.append('s')
                if x < center_x_start:
                    keys.append('a')
                elif x > center_x_end:
                    keys.append('d')

            current_keys.update(keys)

            label = " + ".join([k.upper() for k in keys]) if keys else "Idle"
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.circle(img, (x, y), 10, (0, 255, 255), -1)

    pressed_keys = current_keys - prev_keys
    released_keys = prev_keys - current_keys

    for k in released_keys:
        pyautogui.keyUp(k)

    for k in pressed_keys:
        pyautogui.keyDown(k)

    prev_keys = current_keys.copy()

    # UI Feedback
    if predicted_label:
        cv2.putText(img, f"Gesture: {predicted_label}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,1.2 , (0, 255, 255), 2)
    joystick_text = "Joystick: ON" if joystick_enabled else "Joystick: LOCKED"
    cv2.putText(img, joystick_text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Finger Control", img)

    if key == 27:  
        break

for k in prev_keys:
    pyautogui.keyUp(k)
cap.release()
cv2.destroyAllWindows()
