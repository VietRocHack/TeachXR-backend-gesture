import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import win32gui
import win32ui
from ctypes import windll
from PIL import Image
from collections import deque

def get_window_rect(window_title):
    hwnd = win32gui.FindWindow(None, window_title)
    if not hwnd:
        raise Exception(f'Window not found: {window_title}')
    
    rect = win32gui.GetWindowRect(hwnd)
    x = rect[0]
    y = rect[1]
    w = rect[2] - x
    h = rect[3] - y
    
    return {"top": y, "left": x, "width": w, "height": h}

def capture_window(window_title, capture_width, capture_height):
    hwnd = win32gui.FindWindow(None, window_title)
    if not hwnd:
        raise Exception(f'Window not found: {window_title}')

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, capture_width, capture_height)

    saveDC.SelectObject(saveBitMap)

    result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)

    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)

    im = Image.frombuffer(
        'RGB',
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr, 'raw', 'BGRX', 0, 1)

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    if result == 1:
        return np.array(im)
    else:
        return None

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the GestureRecognizer
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    num_hands=1,  # Detect only one hand
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Specify the window title to capture
window_title = "Casting"
capture_width = 1500
capture_height = 1500

try:
    window_rect = get_window_rect(window_title)
except Exception as e:
    print(f"Error: {e}")
    exit(1)

cv2.namedWindow("Right Hand Gesture Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Right Hand Gesture Recognition", capture_width, capture_height)

# Initialize deque to store index finger tip coordinates
index_finger_coords = deque(maxlen=20)

while True:
    # Capture window content
    image = capture_window(window_title, capture_width, capture_height)
    
    if image is None:
        print("Failed to capture window content")
        continue

    # Create MediaPipe image directly from the captured image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # Recognize gestures in the input image
    recognition_result = recognizer.recognize(mp_image)

    # Draw hand landmarks and display gesture only for the right hand
    if recognition_result.gestures and recognition_result.hand_landmarks:
        # Check if the detected hand is the right hand
        if recognition_result.handedness[0][0].category_name == "Right":
            top_gesture = recognition_result.gestures[0][0]
            hand_landmarks = recognition_result.hand_landmarks[0]

            # Draw hand landmarks
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                for landmark in hand_landmarks
            ])
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Display gesture
            gesture_text = f"Right Hand: {top_gesture.category_name} ({top_gesture.score:.2f})"
            cv2.putText(image, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # If pointing up gesture is detected, save index finger tip coordinates
            if top_gesture.category_name == "Pointing_Up":
                index_finger_tip = hand_landmarks[8]  # Index 8 corresponds to the tip of the index finger
                x = int(index_finger_tip.x * capture_width)
                y = int(index_finger_tip.y * capture_height)
                index_finger_coords.append((x, y))

                # Draw a circle at the index finger tip
                cv2.circle(image, (x, y), 10, (0, 0, 255), -1)

            # Draw the trajectory of the index finger tip
            for i in range(1, len(index_finger_coords)):
                cv2.line(image, index_finger_coords[i-1], index_finger_coords[i], (255, 0, 0), 2)

        else:
            # If a hand is detected but it's not the right hand
            cv2.putText(image, "No right hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # If no hand is detected
        cv2.putText(image, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the image
    cv2.imshow('Right Hand Gesture Recognition', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()