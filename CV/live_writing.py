# import cv2
# import numpy as np

# loadFromSys = True

# if loadFromSys:
# 	hsv_value = np.load('hsv_value.npy')

# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4,720)

# kernel = np.ones((5, 5), np.int8)

# canvas = np.zeros((720, 1280, 3))

# x1 = 0
# y1 = 0

# noise_thresh = 800

# while True:
# 	_, frame = cap.read()
# 	frame = cv2.flip(frame, 1)

# 	if canvas is not None:
# 		canvas = np.zeros_like(frame)

# 	hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

# 	if loadFromSys:
# 		lower_range = hsv_value[0]
# 		upper_range = hsv_value[1]

# 	mask = cv2.inRange(hsv, lower_range, upper_range)

# 	mask = cv2.erode(mask, kernel, iterations = 1)
# 	mask = cv2.dilate(mask, kernel, iterations = 2)

# 	contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 	if contours  and cv2.contourArea(max(contours, key = cv2.contourArea)) > noise_thresh:
# 		c = max(contours, key = cv2.contourArea)
# 		x2, y2 ,w, h = cv2.boundingRect(c)

# 		if x1 == 0 and y1 == 0:
# 			x1,y1 = x2,y2
# 		else:
# 			canvas = cv2.line(canvas, (x1,y1), (x2,y2), [0,255,0], 4)

# 		x1,y1 = x2,y2
	
# 	else:
# 		x1,y1 = 0, 0

# 	frame = cv2.add(canvas, frame)

# 	stacked = np.hstack((canvas, frame))
# 	cv2.imshow('Screen_Pen', cv2.resize(stacked, None, fx = 0.6, fy = 0.6))

# 	if cv2.waitKey(1) == 10:
# 		break

# 	#Clear the canvas when 'c' is pressed
# 	if cv2.waitKey(1) & 0xFF == ord('c'):
# 		canvas = None

# cv2.destroyAllWindows()
# cap.release()

import cv2
import os
import numpy as np
import torch
import sys




sys.path.append(os.path.abspath("/Users/sarayu.madakasira/CNN/"))  # Add CNN to path

from MNIST_CNN.model import ConvNet  # ✅ FIXED: Import ConvNet properly

from pathlib import Path


# Load trained CNN model
model = ConvNet() 
model.load_state_dict(torch.load('/Users/sarayu.madakasira/CNN/MNIST_CNN/model.pth', map_location=torch.device('cpu')))
model.eval() 
# Load HSV values
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

    frame = cv2.flip(frame, 1)  # Flip horizontally

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_range = hsv_value[0]
    upper_range = hsv_value[1]
    mask = cv2.inRange(hsv, lower_range, upper_range)

    # mask = cv2.erode(mask, kernel, iterations=1)
    # mask = cv2.dilate(mask, kernel, iterations=2)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)  # Smooth edges
    mask = cv2.erode(mask, None, iterations=2)  # Remove small noise
    mask = cv2.dilate(mask, None, iterations=2)  # Strengthen detection

    cv2.imshow("Detected Mask", mask)  # ✅ Debug: Show what the camera detects
    cv2.waitKey(1)



    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x2, y2, w, h = cv2.boundingRect(largest_contour)

        if w > 5 and h > 5:  # Ensure it's a valid digit
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 4)
            x1, y1 = x2, y2

            # Extract the drawn digit and process it
            digit_roi = mask[y2:y2+h, x2:x2+w]
            cv2.imshow("Extracted Digit", digit_roi)  # ✅ Debug: Check if digit is extracted correctly
            cv2.waitKey(1)
            # digit_resized = cv2.resize(digit_roi, (28, 28))
            # digit_resized = digit_resized / 255.0  # Normalize
            # digit_tensor = torch.tensor(digit_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            digit_resized = cv2.resize(digit_roi, (28, 28))
            digit_resized = digit_resized.astype("float32") / 255.0  # Normalize
            digit_tensor = torch.from_numpy(digit_resized).unsqueeze(0).unsqueeze(0)
            print(f"Tensor Shape: {digit_tensor.shape}")  # ✅ Debug: Ensure shape is (1,1,28,28)
            # Extract the drawn digit and process it
            


            # Get CNN prediction
            with torch.no_grad():
                output = model(digit_tensor)
                predicted_digit = torch.argmax(output).item()
                print(f"Model Prediction: {predicted_digit}") 

            # Display prediction
            cv2.putText(frame, f"Predicted: {predicted_digit}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)  # Larger, green, and smooth text

    else:
        x1, y1 = 0, 0

    frame = cv2.add(canvas, frame)
    cv2.imshow('Screen_Pen', frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):  # Press 'q' to quit
        break
    elif key & 0xFF == ord('c'):  # Press 'c' to clear the canvas
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()
