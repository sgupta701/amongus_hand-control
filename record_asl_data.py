#record_asl_data.py

import cv2
import mediapipe as mp
import csv
import os
import numpy as np
import string

# mediapipe initilization
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# set up webcam 
cap = cv2.VideoCapture(0)
cv2.namedWindow("ASL Data Recorder")

# set up data storage
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)
csv_file = os.path.join(data_folder, "asl_data.csv")

# write header if file doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ('x','y','z')]
        writer.writerow(header)

print("Instructions:")
print("- Press A–Z to record those letters")
print("- Press SPACEBAR to record gesture for space ('SPACE')")
print("- Press ENTER to record gesture for new line ('ENTER')")
print("- Press ESC to quit")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture frame.")
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    img_height, img_width = img.shape[:2]

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks = None

    # this shows text on screen
    cv2.putText(img, "Press A-Z / SPACE / ENTER", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("ASL Data Recorder", img)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # esc
        break

    elif landmarks:
        # A-Z keys
        if chr(key).upper() in string.ascii_uppercase:
            label = chr(key).upper()
        # spacebar
        elif key == 32:
            label = "SPACE"
        # enter
        elif key == 13:
            label = "ENTER"
        else:
            continue  # skip other key

        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([label] + landmarks)
        print(f"Saved sample for label: {label}")

cap.release()
cv2.destroyAllWindows()
