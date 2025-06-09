import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

# loading the saved model, scaler, and label encoder
model = joblib.load('models/asl_mlp_model.pkl')
scaler = joblib.load('models/asl_scaler.pkl')
encoder = joblib.load('models/asl_label_encoder.pkl')

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def extract_landmarks(hand_landmarks):
    # flatten the 21 points with x,y,z into a single vector
    data = []
    for lm in hand_landmarks.landmark:
        data.extend([lm.x, lm.y, lm.z])
    return np.array(data)

cap = cv2.VideoCapture(0)
prev_pred = None
pred_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        landmarks = extract_landmarks(hand_landmarks).reshape(1, -1)
        landmarks_scaled = scaler.transform(landmarks)
        
        pred = model.predict(landmarks_scaled)[0]
        letter = encoder.inverse_transform([pred])[0]
        
        now = time.time()
        if letter != prev_pred or now - pred_time > 1.0:
            prev_pred = letter
            pred_time = now
        
        cv2.putText(frame, f'Predicted: {prev_pred}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        prev_pred = None

    cv2.imshow("ASL Letter Prediction", frame)
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
