
import cv2
import os
import numpy as np
import torch
import sys

sys.path.append(os.path.abspath("/Users/sarayu.madakasira/CNN/"))  

from MNIST_CNN.model import ConvNet  

from pathlib import Path

model = ConvNet() 
model.load_state_dict(torch.load('/Users/sarayu.madakasira/CNN/MNIST_CNN/model.pth', map_location=torch.device('cpu')))
model.eval() 
hsv_value = np.load('/Users/sarayu.madakasira/CNN/CV/hsv_value.npy')

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

kernel = np.ones((5, 5), np.uint8)

canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

x1, y1 = 0, 0
noise_thresh = 800

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_range = hsv_value[0]
    upper_range = hsv_value[1]
    mask = cv2.inRange(hsv, lower_range, upper_range)


    mask = cv2.GaussianBlur(mask, (5, 5), 0)  
    mask = cv2.erode(mask, None, iterations=2)  
    mask = cv2.dilate(mask, None, iterations=2)  

    cv2.imshow("Detected Mask", mask)  
    cv2.waitKey(1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x2, y2, w, h = cv2.boundingRect(largest_contour)

        if w > 5 and h > 5:  
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 4)
            x1, y1 = x2, y2

            digit_roi = mask[y2:y2+h, x2:x2+w]
            digit_resized = cv2.resize(digit_roi, (28, 28))
            digit_resized = digit_resized.astype("float32") / 255.0 
            digit_tensor = torch.from_numpy(digit_resized).unsqueeze(0).unsqueeze(0)
            print(f"Tensor Shape: {digit_tensor.shape}")  
            

            with torch.no_grad():
                output = model(digit_tensor)
                predicted_digit = torch.argmax(output).item()
                print(f"Model Prediction: {predicted_digit}") 

            cv2.putText(frame, f"Predicted: {predicted_digit}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA) 

    else:
        x1, y1 = 0, 0

    frame = cv2.add(canvas, frame)
    cv2.imshow('Screen_Pen', frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):  
        break
    elif key & 0xFF == ord('c'):  
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()
