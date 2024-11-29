import cv2
from ultralytics import YOLO

# Load the model into memory and get labemap
model = YOLO('/content/model.pt')
labels = model.names

# Load an image
img_path = 'test.jpg'
img = cv2.imread(img_path)

# Run inference on image
results = model(img)

# Extract results
detections = results[0].boxes

# Initialize variable for basic object counting example
object_count = 0

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

      cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,255,0), 2)

      label = f'{classname}: {conf*100:.2f}%'
      labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
      label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
      cv2.rectangle(img, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255,255,255), cv2.FILLED) # Draw white box to put label text in
      cv2.putText(img, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text

      # Basic example: count the number of objects in the image that match the first class in the labelmap
      if classname == labels[0]:
        object_count = object_count + 1

# Display detection results
cv2.putText(img, f'Number of {labels[0]}s: {object_count}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2) # Draw total number of detected quarters in top left corner of image
cv2.imshow('YOLO detection results',img) # Display image

# Wait for keypress then close all windows
cv2.waitKey()
cv2.destroyAllWindows()
