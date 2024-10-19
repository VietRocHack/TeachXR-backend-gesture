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
import datetime
import os
import time

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

# Initialize the GestureRecognizer with lower thresholds
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    num_hands=1,  # Detect only one hand
    min_hand_detection_confidence=0.1,  # Lowered from 0.5
    min_hand_presence_confidence=0.1,  # Lowered from 0.5
    min_tracking_confidence=0.8  # Lowered from 0.5
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

# Initialize last crop time
last_crop_time = 0

# Initialize variables for displaying cropped image
cropped_image = None
crop_display_start_time = 0
crop_display_duration = 3  # Display cropped image for 3 seconds

while True:
    # Capture window content
    original_image = capture_window(window_title, capture_width, capture_height)
    
    if original_image is None:
        print("Failed to capture window content")
        continue

    # Create a copy of the original image for drawing
    display_image = original_image.copy()

    # Create MediaPipe image directly from the captured image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=original_image)

    # Recognize gestures in the input image
    recognition_result = recognizer.recognize(mp_image)

    # Draw hand landmarks and display gesture only for the right hand
    if recognition_result.gestures and recognition_result.hand_landmarks:
        # Check if the detected hand is the right hand
        if recognition_result.handedness[0][0].category_name == "Right":
            top_gesture = recognition_result.gestures[0][0]
            hand_landmarks = recognition_result.hand_landmarks[0]

            # Draw hand landmarks on the display image
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                for landmark in hand_landmarks
            ])
            mp_drawing.draw_landmarks(
                display_image,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Display gesture on the display image
            gesture_text = f"Right Hand: {top_gesture.category_name} ({top_gesture.score:.2f})"
            cv2.putText(display_image, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # If pointing up gesture is detected, save index finger tip coordinates
            if top_gesture.category_name == "Pointing_Up":
                index_finger_tip = hand_landmarks[8]  # Index 8 corresponds to the tip of the index finger
                x = int(index_finger_tip.x * capture_width)
                y = int(index_finger_tip.y * capture_height)
                index_finger_coords.append((x, y))

                # Draw a circle at the index finger tip on the display image
                cv2.circle(display_image, (x, y), 10, (0, 0, 255), -1)

            # Draw the trajectory of the index finger tip on the display image
            for i in range(1, len(index_finger_coords)):
                cv2.line(display_image, index_finger_coords[i-1], index_finger_coords[i], (255, 0, 0), 2)

            # If thumbs up gesture is detected, crop and save the image
            current_time = time.time()
            if top_gesture.category_name == "Thumb_Up" and len(index_finger_coords) > 0 and current_time - last_crop_time > 5:
                # Calculate bounding box with 100px margin on each side
                x_coords, y_coords = zip(*index_finger_coords)
                min_x, max_x = max(0, min(x_coords) - 100), min(capture_width, max(x_coords) + 100)
                min_y, max_y = max(0, min(y_coords) - 100), min(capture_height, max(y_coords) + 100)

                # Crop the original image
                cropped_image = original_image[min_y:max_y, min_x:max_x]

                # Save the cropped image with timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"cropped_{timestamp}.png"
                cv2.imwrite(filename, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
                print(f"Saved cropped image: {filename}")

                # Clear the coordinates
                index_finger_coords.clear()

                # Update last crop time and start display timer
                last_crop_time = current_time
                crop_display_start_time = current_time

        else:
            # If a hand is detected but it's not the right hand
            cv2.putText(display_image, "No right hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # If no hand is detected
        cv2.putText(display_image, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display cropped image and success message if within display duration
    if cropped_image is not None and time.time() - crop_display_start_time < crop_display_duration:
        # Resize cropped image to fit in the bottom right corner
        display_height, display_width = display_image.shape[:2]
        cropped_height, cropped_width = cropped_image.shape[:2]
        max_cropped_height = int(display_height * 0.3)
        max_cropped_width = int(display_width * 0.3)
        scale = min(max_cropped_height / cropped_height, max_cropped_width / cropped_width)
        resized_cropped = cv2.resize(cropped_image, (int(cropped_width * scale), int(cropped_height * scale)))

        # Calculate position for cropped image
        y_offset = display_height - resized_cropped.shape[0] - 10
        x_offset = display_width - resized_cropped.shape[1] - 10

        # Overlay cropped image on display image
        display_image[y_offset:y_offset+resized_cropped.shape[0], x_offset:x_offset+resized_cropped.shape[1]] = resized_cropped

        # Add success message
        cv2.putText(display_image, "Image cropped!", (x_offset, y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the image
    cv2.imshow('Right Hand Gesture Recognition', cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR))

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()