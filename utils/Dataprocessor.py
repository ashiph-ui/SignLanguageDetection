import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, 
                      max_num_hands=1,
                      min_detection_confidence=0.3,  # Increased from 0 for better detection
                      min_tracking_confidence=0.3)


def augment_data(image, label):
    # Augmentation techniques
    augmented_images = []
    augmented_labels = []
    
    # Horizontal flip
    flip_image = cv2.flip(image, 1)
    augmented_images.append(flip_image)
    augmented_labels.append(label)  # Keep the same label
    
    # Rotation (e.g., by 10 degrees)
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 10, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    augmented_images.append(rotated_image)
    augmented_labels.append(label)  # Keep the same label
    
    # Scaling (zoom-in effect)
    scale_factor = 1.1
    scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    augmented_images.append(scaled_image)
    augmented_labels.append(label)  # Keep the same label
    
    # Shifting (translation)
    M = np.float32([[1, 0, 10], [0, 1, 10]])  # Shift by 10 pixels in both directions
    shifted_image = cv2.warpAffine(image, M, (cols, rows))
    augmented_images.append(shifted_image)
    augmented_labels.append(label)  # Keep the same label
    
    # You can add more augmentation techniques like random cropping, brightness adjustment, etc.
    
    return augmented_images, augmented_labels


def get_coords(image):
    # Preprocess the image

    # image is float32 [0.0 - 1.0]; convert it back
    image_uint8 = (image * 255).astype(np.uint8)
    image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (224, 224))
    
    # Process the image
    results = hands.process(image_rgb)
    coords = []
    
    if results.multi_hand_world_landmarks:
        for hand_landmarks in results.multi_hand_world_landmarks:
            # Extract world landmarks coordinates
            for landmark in hand_landmarks.landmark:
                coords.extend([landmark.x, landmark.y, landmark.z])
                
    return coords if coords else None

def augment_dataset(x_data, y_data):
    augmented_images = []
    augmented_labels = []
    
    for image, label in zip(x_data, y_data):
        augmented_imgs, augmented_lbls = augment_data(image, label)
        
        # Append the augmented images and labels to the lists
        augmented_images.extend(augmented_imgs)
        augmented_labels.extend(augmented_lbls)
    
    return augmented_images, augmented_labels


def filter_data(x_data, y_data):
    coords = [get_coords(x) for x in x_data]
    filtered_coords, filtered_labels = zip(*[(coord, label) 
    for coord, label in zip(coords, y_data) if coord is not None])
    return filtered_coords, filtered_labels
