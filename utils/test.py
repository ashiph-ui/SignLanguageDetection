import cv2 
import mediapipe as mp
import torch
import joblib  
import torch.nn as nn
import torch.nn.functional as F  

class EnhancedHandGestureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(63, 128)  # Adjust input size (63) for hand coordinates
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.1)
        
        self.fc3 = nn.Linear(64, 27)  # Output layer for 27 classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))  # Apply ReLU after BatchNorm and Linear layer
        x = self.drop1(x)  # Dropout for regularization
        x = self.relu(self.bn2(self.fc2(x)))  # Apply ReLU after BatchNorm and Linear layer
        x = self.drop2(x)  # Dropout for regularization
        x = self.fc3(x)  # Output logits for 27 classes
        return x  # No Softmax required for CrossEntropyLoss

# Initialize the model
model = EnhancedHandGestureModel()

# Load the state dictionary (the saved model weights)
state_dict = torch.load("best_model.pth")

# Load the state dictionary into the model
model.load_state_dict(state_dict)
model.eval()

# Load the scaler
scaler = joblib.load("scaler.pkl")  # Adjust path as necessary

# Define the mapping for 27 classes
class_map = ['Blank'] + [chr(i) for i in range(ord('A'), ord('Z')+1)]  # Blank + A-Z

def map_class_to_letter(predicted_class):
    """ Map the predicted class index to a letter or blank """
    return class_map[predicted_class]

def predict_gesture(landmarks):
    """ Normalize landmarks and run prediction with the model """
    # Convert landmarks into a 63-dimensional vector (flattened)
    hand_coords = []
    for lm in landmarks:
        hand_coords.append(lm.x)
        hand_coords.append(lm.y)
        hand_coords.append(lm.z)

    # Convert to a 2D array and apply scaling
    hand_coords = [hand_coords]  # Convert to 2D array with shape (1, 63)
    hand_coords = scaler.transform(hand_coords)  # Normalize the coordinates
    
    # Convert to a torch tensor
    input_tensor = torch.tensor(hand_coords, dtype=torch.float32)  # Add batch dimension
    
    # Get model output
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)  # Get predicted class
    return predicted.item()  # Return the predicted class

# Mediapipe intialisation
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            break
        # Convert the image to RGB (MediaPipe works with RGB images)
        image.flags.writeable =True
        result = hands.process(image)
        if result.multi_hand_world_landmarks:
            for landmarks in result.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

            for landmarks in result.multi_hand_world_landmarks:
                # # Extract landmarks and make predictions
                predicted_class = predict_gesture(landmarks.landmark)
                # Map predicted class index to gesture label
                predicted_gesture = map_class_to_letter(predicted_class)

                # Display the predicted gesture on the frame
                cv2.putText(image, f"Gesture: {predicted_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('mediapipe', image)
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
