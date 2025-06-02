# app/detect.py

import cv2
import mediapipe as mp
import torch
import joblib
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# === Model Definition ===
class EnhancedHandGestureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(63, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(64, 27)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)
        return x

# === Initialization (only once at import) ===
model = EnhancedHandGestureModel()
model.load_state_dict(torch.load("model/best_model.pth", map_location=torch.device('cpu')))
model.eval()

scaler = joblib.load("model/scaler.pkl")
class_map = ['Blank'] + [chr(i) for i in range(ord('A'), ord('Z') + 1)]

mp_hands = mp.solutions.hands

# === Core prediction function ===
def predict_asl(frame):
    """Predict ASL gesture from a single OpenCV frame (BGR image)"""
    with mp_hands.Hands(static_image_mode=True,
                        model_complexity=0,
                        min_detection_confidence=0.5) as hands:

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_world_landmarks:
            landmarks = results.multi_hand_world_landmarks[0].landmark
            coords = []
            for lm in landmarks:
                coords.extend([lm.x, lm.y, lm.z])
            
            coords = scaler.transform([coords])
            input_tensor = torch.tensor(coords, dtype=torch.float32)
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                gesture = class_map[predicted.item()]
                return gesture

        return "No hand detected"
