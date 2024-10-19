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
import requests
import io
import threading

# OCR server details
OCR_SERVER_URL = "http://192.168.86.115:5000/ocr"

# Bounding box padding
BOUNDING_BOX_PADDING = 100

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

def send_image_to_ocr(image, callback):
    try:
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Save image to a byte buffer
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Prepare the file for sending
        files = {'image': ('image.png', img_byte_arr, 'image/png')}

        # Send POST request to OCR server
        response = requests.post(OCR_SERVER_URL, files=files)

        if response.status_code == 200:
            ocr_text = response.text  # Get raw text from response
            callback("OCR processing successful", ocr_text)
        else:
            callback(f"OCR processing failed: {response.status_code}", '')
    except Exception as e:
        callback(f"Error sending image to OCR: {str(e)}", '')

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the GestureRecognizer with lower thresholds
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    num_hands=1,  # Detect only one hand
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
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

def main_loop():
    # Initialize variables
    index_finger_coords = deque(maxlen=100)
    last_crop_time = 0
    cropped_image = None
    crop_display_start_time = 0
    crop_display_duration = 10
    cooldown_duration = 5
    ocr_status = ""
    ocr_status_time = 0
    ocr_status_duration = 5
    ocr_result_text = ""

    def ocr_callback(status, text):
        nonlocal ocr_status, ocr_result_text, ocr_status_time
        ocr_status = status
        ocr_result_text = text
        ocr_status_time = time.time()

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

        # Initialize gesture_text
        gesture_text = "No hand detected"

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

                # Update gesture text
                gesture_text = f"Right Hand: {top_gesture.category_name} ({top_gesture.score:.2f})"

                # If pointing up gesture is detected, save index finger tip coordinates with timestamp
                if top_gesture.category_name == "Pointing_Up":
                    index_finger_tip = hand_landmarks[8]  # Index 8 corresponds to the tip of the index finger
                    x = int(index_finger_tip.x * capture_width)
                    y = int(index_finger_tip.y * capture_height)
                    index_finger_coords.append((x, y, time.time()))

                # Check if thumbs up gesture is detected and cooldown has passed
                current_time = time.time()
                if top_gesture.category_name == "Thumb_Up" and len(index_finger_coords) > 0 and current_time - last_crop_time > cooldown_duration:
                    # Filter coordinates that are less than 5 seconds old
                    valid_coords = [(x, y) for x, y, t in index_finger_coords if current_time - t <= 5]

                    if valid_coords:
                        # Calculate bounding box with BOUNDING_BOX_PADDING on each side
                        x_coords, y_coords = zip(*valid_coords)
                        min_x = max(0, min(x_coords) - BOUNDING_BOX_PADDING)
                        max_x = min(capture_width, max(x_coords) + BOUNDING_BOX_PADDING)
                        min_y = max(0, min(y_coords) - BOUNDING_BOX_PADDING)
                        max_y = min(capture_height, max(y_coords) + BOUNDING_BOX_PADDING)

                        # Crop the original image
                        cropped_image = original_image[min_y:max_y, min_x:max_x]

                        # Save the cropped image with timestamp
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"cropped_{timestamp}.png"
                        cv2.imwrite(filename, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
                        print(f"Saved cropped image: {filename}")

                        # Send the cropped image to OCR server using a separate thread
                        ocr_thread = threading.Thread(target=send_image_to_ocr, args=(cropped_image, ocr_callback))
                        ocr_thread.start()

                        ocr_status = "Sending to OCR server..."
                        ocr_status_time = current_time

                        # Clear the coordinates
                        index_finger_coords.clear()

                        # Update last crop time and start display timer
                        last_crop_time = current_time
                        crop_display_start_time = current_time

            else:
                # If a hand is detected but it's not the right hand
                gesture_text = "No right hand detected"

        # Remove coordinates older than 5 seconds
        current_time = time.time()
        index_finger_coords = deque([(x, y, t) for x, y, t in index_finger_coords if current_time - t <= 5], maxlen=100)

        # Draw the trajectory of the index finger tip on the display image
        for i in range(1, len(index_finger_coords)):
            cv2.line(display_image, (index_finger_coords[i-1][0], index_finger_coords[i-1][1]), 
                     (index_finger_coords[i][0], index_finger_coords[i][1]), (255, 0, 0), 2)

        # Display croppable status
        time_since_last_crop = current_time - last_crop_time
        if time_since_last_crop > cooldown_duration:
            status_text = "CROPPABLE"
            status_color = (0, 255, 0)  # Green
        else:
            time_left = max(0, cooldown_duration - time_since_last_crop)
            status_text = f"NOT CROPPABLE (Cooldown: {time_left:.1f}s)"
            status_color = (0, 0, 255)  # Red

        # Create a semi-transparent overlay for the status background
        overlay = display_image.copy()
        cv2.rectangle(overlay, (0, 0), (400, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, display_image, 0.5, 0, display_image)

        # Display gesture text and status on the semi-transparent background
        cv2.putText(display_image, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_image, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        # Display OCR sending status
        if current_time - ocr_status_time < ocr_status_duration:
            cv2.putText(display_image, ocr_status, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Display cropped image, OCR result, and success message if within display duration
        if cropped_image is not None and current_time - crop_display_start_time < crop_display_duration:
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

            # Display OCR result text with black background
            if ocr_result_text:
                # Split the OCR result text into lines
                lines = ocr_result_text.split('\n')
                text_height = len(lines) * 30  # Assuming 30 pixels per line
                text_width = max(len(line) * 10 for line in lines)  # Assuming 10 pixels per character

                # Create a black background for the text
                text_bg = np.zeros((text_height + 20, text_width + 20, 3), dtype=np.uint8)
                
                # Position the text background in the bottom left corner
                bg_y_offset = display_height - text_height - 30
                display_image[bg_y_offset:bg_y_offset+text_height+20, 10:text_width+30] = text_bg

                # Add OCR text
                for i, line in enumerate(lines):
                    
                    cv2.putText(display_image, line, (20, bg_y_offset + 30 + i * 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the image
        cv2.imshow('Right Hand Gesture Recognition', cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR))

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()