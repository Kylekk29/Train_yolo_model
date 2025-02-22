# YOLO detection script on Picamera

import os
import sys
import argparse
import glob
import time
import gpiozero

import cv2
import numpy as np
from ultralytics import YOLO

### Set user-defined parameters and program parameters

# User-defined parameters
model_path = 'yolo11n_ncnn_model'	# Path to model file or folder
cam_source = 'usb0' 				# Options: 'usb0' for USB camera, 'picamera0' for Picamera
min_thresh = 0.5 					# Minimum detection threshold
resW, resH = 1280, 720				# Resolution to run camera at
record = False						# Enables recording if True

# Program parameters
# Define Raspberry Pi GPIO pin to toggle
gpio_pin = 14

# Define box coordinates where we want to look for a person. If a person is present in this box for enough frames, toggle GPIO to turn light on.
pbox_xmin = 540
pbox_ymin = 160
pbox_xmax = 760
pbox_ymax = 450

# Set detection bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

### Initialize YOLO model, GPIO, and camera

# Set up Raspberry Pi GPIO
led = gpiozero.LED(gpio_pin)

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit()

# Load the model into memory and get labemap
model = YOLO(model_path, task='detect')
labels = model.names

# Set up recording
if record:
    record_name = 'demo6.avi'
    record_fps = 5
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Initialize Picamera or USB camera depending on user input
if 'usb' in cam_source:
    cam_type = 'usb'
    cam_idx = int(cam_source[3:])
    cam = cv2.VideoCapture(cam_idx)
    ret = cam.set(3, resW)
    ret = cam.set(4, resH)

elif 'picamera' in cam_source:
    from picamera2 import Picamera2
    cam_type = 'picamera'
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cam.start()

else:
    print('Invalid input for cam_source variable! Use "usb0" or "picamera0". Exiting program.')
    sys.exit()


# Initialize frame rate variables 
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200

# Initialize control and status variables
consecutive_detections = 0
gpio_state = 0

### Begin main inference loop
while True:

    t_start = time.perf_counter()

    # Grab frame from USB camera or Picamera (depending on user selection)
    if cam_type == 'usb':
        ret, frame = cam.read()

    elif cam_type == 'picamera':
        frame_bgra = cam.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR) # Remove alpha channel

    # Check to make sure frame was received
    if (frame is None):
        print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
        break

    ### Run inference on frame and parse detections
    
    # Run inference on frame with tracking enabled (tracking helps object to be consistently detected in each frame)
    results = model.track(frame, verbose=False)

    # Extract results
    detections = results[0].boxes

    # Initialize array to hold locations of "person" detections
    person_locations = []

    # Go through each detection and get bbox coords, confidence, and class
    for i in range(len(detections)):

        # Get bounding box coordinates
        # Ultralytics returns results in Tensor format, which have to be converted to a regular Python array
        xyxy_tensor = detections[i].xyxy.cpu() # Detections in Tensor format in CPU memory
        xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
        xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int
        
        # Calculate center coordinates from xyxy coordinates
        cx = int((xmin + xmax)/2)
        cy = int((ymin + ymax)/2)

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # Get bounding box confidence
        conf = detections[i].conf.item()

        # Draw box if confidence threshold is high enough
        if conf > 0.5:

            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text

            # If this object is a person, append their coordinates to running list of person detections
            if classname == 'person':
                person_locations.append([cx, cy])
                # Draw a cirle there too (and make it change color based on number of consecutive detections)
                color_intensity = 30*consecutive_detections
                cv2.circle(frame, (cx, cy), 7, (0,color_intensity,color_intensity), -1) 

    ### Logic to trigger GPIO change
    
    # Initialize flag to indicate whether person is in the desired location this frame (set as False)
    person_in_pbox = False
    
    # Go through person detections to check if any are within desired box location
    for person_xy in person_locations:
        
        person_cx, person_cy = person_xy # Get center coordinates for this person
        
        # This big conditional checks if the person's center_x/center_y coordinates are within the box coordinates
        if (person_cx > pbox_xmin) and (person_cx < pbox_xmax) and (person_cy > pbox_ymin) and (person_cy < pbox_ymax):
            person_in_pbox = True

    # If there is a person in the box, increment consecutive detection count by 1 (but not above 15)
    if person_in_pbox == True:
        consecutive_detections = min(8, consecutive_detections + 1) # Prevents this variable from going above 15 
    
    # If not, decrease consecutive detection count by 1 (but not below 0)
    else:
        consecutive_detections = max(0, consecutive_detections - 1)
    
    # If consecutive detections are high enough AND the GPIO is currently off, turn GPIO on!
    if consecutive_detections >= 8 and gpio_state == 0:
        gpio_state = 1
        led.on() # Sets GPIO pin to HIGH (3.3V) state
    
    # Conversely, if consecutive detections are back to 0 AND the GPIO is currently on, turn GPIO off!
    if consecutive_detections <= 0 and gpio_state == 1:
        gpio_state = 0
        led.off() # Sets GPIO pin to LOW (0V) state
    
    
    ### Display results

    # Draw framerate
    cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (20,30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2)
    
    # Draw rectangle around the detection box where we are looking for a person
    cv2.rectangle(frame, (pbox_xmin, pbox_ymin), (pbox_xmax, pbox_ymax), (0,255,255), 2)
    
    # Draw GPIO status on frame
    if gpio_state == 0:
        cv2.putText(frame, 'Light currently OFF.', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2)
    elif gpio_state == 1:
        cv2.putText(frame, 'Person detected in box! Turning light ON.', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 3)
        cv2.putText(frame, 'Person detected in box! Turning light ON.', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    # Display detection results
    cv2.imshow('YOLO detection results',frame) # Display image
    if record: recorder.write(frame)

    # Wait 5ms before moving to next frame and check for user keypress.
    key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'): # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'): # Press 's' to pause inference
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): # Press 'p' to save a picture of results on this frame
        cv2.imwrite('capture.png',frame)
    
    # Calculate FPS for this frame
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # Append FPS result to frame_rate_buffer (for finding average FPS over multiple frames)
    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    # Calculate average FPS for past frames
    avg_frame_rate = np.mean(frame_rate_buffer)


# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if record: recorder.release()
if cam_type == 'usb': cam.release()
if cam_type == 'picamera': cam.stop()
cv2.destroyAllWindows()
