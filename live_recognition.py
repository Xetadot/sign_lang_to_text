import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import pyttsx3
import json
import time
import sys
import os


class GestureClassifier(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=5):
        super(GestureClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class_names = ["Hello", "Ok", "Thank you", "Bye", "Help"]

model = GestureClassifier(input_size=63, hidden_size=128,
                          num_classes=len(class_names))
model.load_state_dict(torch.load('gesture_classifier.pth',
                      map_location=torch.device('cpu')))
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
engine = pyttsx3.init()
spoken_gesture = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            input_tensor = torch.tensor(
                landmarks, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                predicted_idx = torch.argmax(output, dim=1).item()
                predicted_gesture = class_names[predicted_idx]
                confidence = torch.softmax(output, dim=1)[
                    0][predicted_idx].item()

                result_json = {
                    "timestamp": time.time(),
                    "gesture": predicted_gesture,
                    "confidence": round(confidence, 4)
                }

                print(json.dumps(result_json))
                sys.stdout.flush()

            with open("output.json", "a") as f:
                f.write(json.dumps(result_json) + "\n")

            cv2.putText(frame, f'Gesture: {predicted_gesture}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if spoken_gesture != predicted_gesture:
                spoken_gesture = predicted_gesture
                engine.say(predicted_gesture)
                engine.runAndWait()
    else:
        spoken_gesture = None

    cv2.imshow('Sign Language Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if os.path.exists("output.json"):
    open("output.json", "w").close()

cap.release()
cv2.destroyAllWindows()
