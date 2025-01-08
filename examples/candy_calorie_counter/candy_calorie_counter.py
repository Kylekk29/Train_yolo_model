##### Candy Calorie Counter #####

# Author: Evan Juras, EJ Technology Consultants, https://ejtech.io

# Description:
# This script uses a custom YOLO candy detection model to locate and identify pieces of candy in a live camera view.
# It references a dictionary storing nutritional info about each piece of candy, and tallies the total amount
# of sugar and calories from the candy present in the camera's view.

# Import necessary packages
import os
import sys
import cv2
from ultralytics import YOLO

# Define path to model and other user variables
model_path = 'yolo11s_candy_model.pt'  # Path to model
min_thresh = 0.50                      # Minimum detection threshold
cam_index = 0                          # Index of USB camera
imgW, imgH = 1280, 720                 # Resolution to run USB camera at
record = False                         # Record result video

# Create dictionary to hold info about candy calories and sugar. Each entry is stored as {'candy_type': [calories, grams sugar]}
# These were taken from the Costco candy packaging: https://www.costco.com/.product.100333887.html and https://www.costco.com/.product.100688986.html
nutrition_info = {'MMs_peanut': [90, 9], 'MMs_regular': [130, 17], 'airheads': [150, 25], 'gummy_worms': [110, 16], 'milky_way': [80, 10], 'nerds': [45, 12],
              'skittles': [60, 11], 'snickers': [80, 9], 'starbust': [120, 16], 'three_musketeers': [130, 20], 'twizzlers': [90, 12]}

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('WARNING: Model path is invalid or model was not found.')
    sys.exit()

# Load the model into memory and get labemap
model = YOLO(model_path, task='detect')
labels = model.names

# Initialize camera
cap = cv2.VideoCapture(cam_index)
ret = cap.set(3, imgW)
ret = cap.set(4, imgH)

# Set up recording
if record == True:
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (imgW,imgH))

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Begin inference loop
while True:

    # Grab frame from counter
    ret, frame = cap.read()
    if (frame is None) or (not ret):
        print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
        break

    # Run inference on frame with tracking enabled (tracking helps object to be consistently detected in each frame)
    results = model.track(frame, verbose=False)

    # Extract results
    detections = results[0].boxes

    # Initialize variable to hold every candy detected in this frame
    candies_detected = []

    # Go through each detection and get bbox coords, confidence, and class
    for i in range(len(detections)):

        # Get bounding box coordinates
        # Ultralytics returns results in Tensor format, which have to be converted to a regular Python array
        xyxy_tensor = detections[i].xyxy.cpu() # Detections in Tensor format in CPU memory
        xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
        xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # Get bounding box confidence
        conf = detections[i].conf.item()

        # Draw box if confidence threshold is high enough
        if conf > 0.5:

            # Draw box around object
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

            # Draw label for object
            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text

            # Add object to list of detected candies
            candies_detected.append(classname)
    
    # Process list of candies that have been detected to determine total amount of sugar and calories
    total_calories = 0
    total_sugar = 0

    for candy_name in candies_detected:

        # Reference candy nutrition dictionary, and add that candy's calories and sugar to the running total
        calories, sugar = nutrition_info[candy_name]
        total_calories += calories
        total_sugar += sugar

    # Draw number of candy pieces detect, total calories, and total sugar on frame
    cv2.rectangle(frame, (10, 10), (450, 130), (50,50,50), cv2.FILLED) # Rectangle to draw text on
    cv2.putText(frame, f'Number of candies: {len(candies_detected)}', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,102,51), 2)
    cv2.putText(frame, f'Total calories: {total_calories}', (20,75), cv2.FONT_HERSHEY_SIMPLEX, 1, (51,204,51), 2)
    cv2.putText(frame, f'Total sugar (g): {total_sugar}', (20,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,204,255), 2)

    # Display results
    cv2.imshow('Candy detection results',frame) # Display image
    if record: recorder.write(frame) # Record frame to video (if enabled)

    # Poll for user keypress and wait 5ms before continuing to next frame
    key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'): # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'): # Press 's' to pause inference
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): # Press 'p' to save a picture of results on this frame
        cv2.imwrite('capture.png',frame)

# Clean up
cap.release()
if record: recorder.release()
cv2.destroyAllWindows()
