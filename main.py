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

def capture_window(window_title):
    hwnd = win32gui.FindWindow(None, window_title)
    if not hwnd:
        raise Exception(f'Window not found: {window_title}')

    left, top, right, bot = win32gui.GetClientRect(hwnd)
    w = right - left
    h = bot - top

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

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
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Specify the window title to capture
window_title = "Casting"  # Change this to the title of the window you want to capture

try:
    window_rect = get_window_rect(window_title)
except Exception as e:
    print(f"Error: {e}")
    exit(1)

cv2.namedWindow("Window Capture Gesture Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Window Capture Gesture Recognition", window_rect["width"], window_rect["height"])

while True:
    # Capture window content
    image = capture_window(window_title)
    
    if image is None:
        print("Failed to capture window content")
        continue

    # Create MediaPipe image directly from the captured image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # Recognize gestures in the input image
    recognition_result = recognizer.recognize(mp_image)

    # Draw hand landmarks and display gesture
    if recognition_result.gestures and recognition_result.hand_landmarks:
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
        gesture_text = f"{top_gesture.category_name} ({top_gesture.score:.2f})"
        cv2.putText(image, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the image
    cv2.imshow('Window Capture Gesture Recognition', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()