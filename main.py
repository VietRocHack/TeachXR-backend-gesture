import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define custom gestures
gestures = {
    'Open Palm': [[0, 5, 17], [0, 9, 17], [0, 13, 17], [0, 17, 17]],
    # 'Fist': [[0, 3, 6], [0, 3, 10], [0, 3, 14], [0, 3, 18]],
    'Pointing': [[0, 8, 12], [0, 8, 16], [0, 8, 20]],
    # 'Victory': [[0, 8, 12], [0, 12, 16], [0, 12, 20]],
    'Thumbs Up': [[4, 8, 12], [4, 8, 16], [4, 8, 20]],
    # Add more custom gestures here
}

def distance(p1, p2):
    return np.sqrt(((p1.x - p2.x) ** 2) + ((p1.y - p2.y) ** 2))

def check_gesture(landmarks, gesture_config):
    for a, b, c in gesture_config:
        dist_ab = distance(landmarks[a], landmarks[b])
        dist_bc = distance(landmarks[b], landmarks[c])
        if dist_ab >= dist_bc:
            return False
    return True

def recognize_gesture(landmarks):
    for gesture_name, gesture_config in gestures.items():
        if check_gesture(landmarks, gesture_config):
            return gesture_name
    return "Unknown"

# Initialize the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Recognize gesture
            gesture = recognize_gesture(hand_landmarks.landmark)

            # Display recognized gesture
            cv2.putText(frame, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()