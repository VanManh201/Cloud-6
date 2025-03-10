import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import tensorflow as tf
import boto3
import time


# Cấu hình AWS
S3_BUCKET = "handwriting-recognition-bucket"
AWS_REGION = "us-east-1"  
s3 = boto3.client("s3")

# Load the handwriting recognition model
model = tf.keras.models.load_model('emnist_handwriting_model.h5')  

# Hàm nhận diện chữ viết tay từ ảnh
def recognize_handwriting(image_path):
    image = Image.open(image_path).convert('L')  # Chuyển sang grayscale
    image = image.resize((28, 28))  # Resize ảnh
    image = np.array(image) / 255.0  # Chuẩn hóa ảnh
    image = image.reshape(1, 28, 28, 1)  # Định dạng lại cho mô hình

    # Dự đoán
    pred = model.predict(image)
    predicted_label = np.argmax(pred, axis=1)[0]
    confidence = pred[0][predicted_label] * 100
    print(f'Số dự đoán: {predicted_label} (confidence: {confidence:.2f}%)')
    

# Initialize MediaPipe Hands.
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Open the webcam.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Tăng chiều rộng
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Tăng chiều cao


# Create a blank image for drawing.
drawing_image = np.zeros((480, 640, 3), dtype=np.uint8)

# Variables to store the previous position of the index finger tip.
prev_x, prev_y = None, None

# Define the color palette.
colors = {
    'red': (0, 0, 255),
    'yellow': (0, 255, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    
}
color_names = list(colors.keys())
current_color = colors['red']

def draw_color_palette(image):
    """Draw color palette on the top right horizontally"""
    height, width, _ = image.shape
    palette_x = width - 540  # Start position from the right side
    spacing = 20  # Spacing between color boxes
    box_size = 100  # Size of the color boxes
    
    # Draw the color boxes
    for i, color_name in enumerate(color_names):
        color = colors[color_name]
        # Draw color box
        cv2.rectangle(image, 
                      (palette_x + i * (box_size + spacing), 10), 
                      (palette_x + box_size + i * (box_size + spacing), 10 + box_size), 
                      color, 
                      -1)
        

def check_if_hand_open(hand_landmarks, image_shape):
    """Check if all five fingers are open."""
    image_height, image_width = image_shape[:2]
    fingers_tips_ids = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                        mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
                        mp_hands.HandLandmark.PINKY_TIP]
    fingers_tips = [hand_landmarks.landmark[tip].y * image_height for tip in fingers_tips_ids]
    fingers_mcp = [hand_landmarks.landmark[mcp].y * image_height for mcp in
                   [mp_hands.HandLandmark.THUMB_CMC, mp_hands.HandLandmark.INDEX_FINGER_MCP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_MCP,
                    mp_hands.HandLandmark.PINKY_MCP]]
    return all(tip < mcp for tip, mcp in zip(fingers_tips, fingers_mcp))

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    screenshot_counter = 0  # Counter for screenshot filenames
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        drawing_image = cv2.resize(drawing_image, (image.shape[1], image.shape[0]))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=2)
                )

                if check_if_hand_open(hand_landmarks, image.shape):
                    drawing_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
                    continue

                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                x1 = int(index_finger_tip.x * image.shape[1])
                y1 = int(index_finger_tip.y * image.shape[0])
                x2 = int(thumb_tip.x * image.shape[1])
                y2 = int(thumb_tip.y * image.shape[0])

                palette_x = image.shape[1] - 540
                spacing = 20
                box_size = 100
                for i, color_name in enumerate(color_names):
                    if palette_x + i * (box_size + spacing) <= x1 <= palette_x + box_size + i * (box_size + spacing) and 10 <= y1 <= 10 + box_size:
                        current_color = colors[color_name]

                if abs(x1 - x2) > 50 and abs(y1 - y2) > 50 and y2 > y1 and x2 > x1:
                    if prev_x is not None and prev_y is not None:
                        cv2.line(drawing_image, (prev_x, prev_y), (x1, y1), current_color, 5)
                    prev_x, prev_y = x1, y1
                else:
                    prev_x, prev_y = None, None
        else:
            prev_x, prev_y = None, None

        draw_color_palette(image)

        combined_image = cv2.addWeighted(image, 0.5, drawing_image, 0.5, 0)

        flipped_image = cv2.flip(combined_image, 1)  # Flip for display only

    # Add text at the top right of the screen
        text1 = "Dien toan dam may - 010100087101"
        text2 = "Nhom 6"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2

        # Calculate text sizes
        text1_size = cv2.getTextSize(text1, font, font_scale, font_thickness)[0]
        text2_size = cv2.getTextSize(text2, font, font_scale, font_thickness)[0]

        # Move the text to the top right
        text1_x = flipped_image.shape[1] - text1_size[0] - 50  # Align to the right, 50px from the right edge
        text1_y = 50  # First text line's y-coordinate

        text2_x = flipped_image.shape[1] - text2_size[0] - 50  # Align to the right, 50px from the right edge
        text2_y = text1_y + 40  # Second text line is 40 pixels below the first one

        # Render both text lines
        cv2.putText(flipped_image, text1, (text1_x, text1_y), font, font_scale, (255, 255, 255), font_thickness)
        cv2.putText(flipped_image, text2, (text2_x, text2_y), font, font_scale, (255, 255, 255), font_thickness)

        cv2.imshow('MediaPipe Hands Drawing', flipped_image)
        # Check for the "s" key to save a screenshot and recognize handwriting.
    
        key = cv2.waitKey(5)
        if key & 0xFF == ord('s'):
            screenshot_filename = f'drawing_screenshot_{screenshot_counter}.png'
            # Get the dimensions of the combined image
            height, width, _ = combined_image.shape
            # Define a smaller size for the square crop
            crop_size = min(height, width) // 2  # Make the crop size smaller (half the original size)
            # Calculate the starting coordinates to move the crop to the top left
            start_x = 0  # Start from the left edge
            start_y = 0  # Start from the top edge
            # Crop the image to a square
            cropped_image = combined_image[start_y:start_y + crop_size, start_x:start_x + crop_size]
            
            # Flip the cropped image
            flipped_cropped_image = cv2.flip(cropped_image, 1)  # Flip horizontally

            cv2.imwrite(screenshot_filename, flipped_cropped_image)  # Save the flipped cropped image

            print(f"Screenshot saved as {screenshot_filename}")

            # Run handwriting recognition on the saved screenshot
            recognize_handwriting(screenshot_filename)

             # Upload ảnh lên AWS S3
            s3.upload_file(screenshot_filename, S3_BUCKET, screenshot_filename)
            print(f"✅ Đã upload {screenshot_filename} lên S3!")

            screenshot_counter += 1
        elif key & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
