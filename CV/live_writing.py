import cv2
import os
import numpy as np
import torch
import sys
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("/Users/sarayu.madakasira/CNN/"))

from MNIST_CNN.model import ConvNet

# Load the trained model
model = ConvNet()
model.load_state_dict(torch.load('/Users/sarayu.madakasira/CNN/MNIST_CNN/model.pth', map_location=torch.device('cpu')))
model.eval()

# OpenCV setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
kernel = np.ones((5, 5), np.uint8)
x1, y1 = 0, 0
noise_thresh = 800  # Only draw if the contour area is greater than this threshold
predicted_digit = None  # Stores the latest prediction

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Load color range (assumes hsv_value.npy exists)
    hsv_value = np.load('hsv_value.npy')
    lower_range, upper_range = hsv_value[0], hsv_value[1]
    
    # Masking
    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Contours detection
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > noise_thresh:
        c = max(contours, key=cv2.contourArea)
        x2, y2, w, h = cv2.boundingRect(c)
        if x1 == 0 and y1 == 0:
            x1, y1 = x2, y2
        else:
            cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 8)
        x1, y1 = x2, y2
    else:
        x1, y1 = 0, 0
    
    # Overlay the drawing on the live video frame
    frame = cv2.add(frame, canvas)
    
    # Display the predicted digit if available
    if predicted_digit is not None:
        cv2.putText(frame, f"Predicted: {predicted_digit}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Live Writing', frame)
    
    # Key press actions
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        predicted_digit = None  # Clear prediction when canvas is cleared
    elif key == ord('p'):
        # Crop and preprocess digit from the canvas
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_canvas, 50, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(thresh)
        digit = thresh[y:y+h, x:x+w]
        
        # Center and resize the digit to 28x28
        h_digit, w_digit = digit.shape
        pad_size = max(h_digit, w_digit)
        padded_digit = np.zeros((pad_size, pad_size), dtype=np.uint8)
        x_offset = (pad_size - w_digit) // 2
        y_offset = (pad_size - h_digit) // 2
        padded_digit[y_offset:y_offset+h_digit, x_offset:x_offset+w_digit] = digit
        digit = cv2.resize(padded_digit, (28, 28))
        
        # Show the preprocessed input image briefly
        cv2.imshow('Model Input', digit)
        cv2.waitKey(500)  # Display for 500ms
        
        # Normalize input and prepare for prediction
        digit = digit.astype(np.float32) / 255.0
        digit = torch.tensor(digit).unsqueeze(0).unsqueeze(0)
        
        # Predict digit
        with torch.no_grad():
            output = model(digit)
            prediction = torch.argmax(output, dim=1).item()
        
        predicted_digit = prediction  # Update global prediction
        print(f"Predicted Digit: {prediction}")

cap.release()
cv2.destroyAllWindows()
