import cv2
import numpy as np
import tensorflow as tf
import boto3
from PIL import Image
import time

# Cấu hình AWS
S3_BUCKET = "handwriting-recognition-bucket"
AWS_REGION = "us-east-1"  
s3 = boto3.client("s3")

# Load mô hình nhận diện chữ số viết tay
model = tf.keras.models.load_model("emnist_handwriting_model.h5")

def process_image(filename):
    """ Tiền xử lý ảnh trước khi đưa vào model """
    image = Image.open(filename).convert("L")  
    original_width, original_height = image.size
    
    # Tạo ảnh vuông với padding để giữ tỉ lệ
    max_size = max(original_width, original_height)
    new_image = Image.new("L", (max_size, max_size), 0)  
    new_image.paste(image, ((max_size - original_width) // 2, (max_size - original_height) // 2))

    # Resize về 28x28
    image_resized = new_image.resize((28, 28), Image.Resampling.LANCZOS)

    # Chuẩn hóa ảnh
    image_array = np.array(image_resized) / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)  # Định dạng input cho model

    return image_array

# Mở camera
cap = cv2.VideoCapture(0)
predicted_label = None  # Biến lưu kết quả dự đoán

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Hiển thị số nhận diện trên camera
    if predicted_label is not None:
        result_text = f"Số: {predicted_label}"
        cv2.putText(frame, result_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hiển thị khung hình camera
    cv2.imshow("Handwriting Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):  # Nhấn 's' để chụp ảnh
        filename = f"handwriting_{int(time.time())}.png"
        cv2.imwrite(filename, frame)
        print(f"📸 Đã lưu ảnh: {filename}")

        # Nhận diện số viết tay
        image = process_image(filename)
        pred = model.predict(image)
        predicted_label = np.argmax(pred, axis=1)[0]
        confidence = pred[0][predicted_label] * 100

        # Upload ảnh lên AWS S3
        s3.upload_file(filename, S3_BUCKET, filename)
        print(f"✅ Đã upload {filename} lên S3!")

        print(f"🔢 Number: {predicted_label} ({confidence:.2f}%)")

    elif key == ord("q"):  # Nhấn 'q' để thoát
        break

cap.release()
cv2.destroyAllWindows()
