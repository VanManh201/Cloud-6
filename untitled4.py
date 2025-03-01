

import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from PIL import Image

# Tải dữ liệu MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Chuẩn hóa dữ liệu (chuyển đổi sang định dạng 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Thêm kênh màu (vì ảnh trắng đen có 1 kênh)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # Lớp cuối với 10 lớp (số 0-9)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy on test data: {test_accuracy:.2f}')

# Chọn một mẫu ngẫu nhiên từ tập kiểm tra
sample_index = np.random.randint(0, x_test.shape[0])
plt.imshow(x_test[sample_index].reshape(28, 28), cmap='gray')
plt.show()

# Dự đoán
pred = model.predict(x_test[sample_index:sample_index+1])
predicted_label = np.argmax(pred, axis=1)
print(f'Predicted label: {predicted_label}')

model.save('emnist_handwriting_model.h5')

# Chọn một mẫu ngẫu nhiên từ tập kiểm tra để kiểm tra thử
sample_index = np.random.randint(0, x_test.shape[0])
plt.imshow(x_test[sample_index].reshape(28, 28), cmap='gray')
plt.title("Ảnh mẫu")
plt.axis("off")
plt.show()

# Dự đoán nhãn cho mẫu đã chọn
pred = model.predict(x_test[sample_index:sample_index+1])
predicted_label = np.argmax(pred, axis=1)[0]  # Chọn nhãn có xác suất cao nhất

# In ra kết quả dự đoán
print(f'Predicted label: {predicted_label}')

# Tạo cửa sổ chọn file
def load_image():
    Tk().withdraw()  # Ẩn cửa sổ chính
    file_path = filedialog.askopenfilename()  # Mở hộp thoại chọn file
    return file_path

# Chọn ảnh từ máy tính
file_path = load_image()

# Đọc và hiển thị ảnh
image = Image.open(file_path).convert('L')  # Chuyển sang ảnh grayscale
image = image.resize((28, 28))  # Chuyển đổi kích thước thành 28x28
plt.imshow(image, cmap='gray')
plt.title("Ảnh đã tải lên")
plt.axis("off")
plt.show()

# Chuẩn hóa ảnh để đưa vào mô hình
image = np.array(image) / 255.0  # Chuẩn hóa ảnh
image = image.reshape(1, 28, 28, 1)  # Định dạng lại cho mô hình

# Dự đoán nhãn cho ảnh đã tải lên
pred = model.predict(image)
predicted_label = np.argmax(pred, axis=1)[0]
confidence = pred[0][predicted_label] * 100

# In ra kết quả dự đoán
print(f'Số dự đoán: {predicted_label} (confidence: {confidence:.2f}%)')

# In xác suất cho từng nhãn
print("\nProbabilities for each label (0-9):")
for i, probability in enumerate(pred[0]):
    print(f'Số {i}: {probability * 100:.2f}%')