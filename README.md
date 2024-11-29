# Train-and-Deploy-YOLO-Models
Tutorials and examples showing how to train and deploy Ultralytics YOLO models.

Colab notebook for training YOLO models to be added soon.

## Usage Instructions
The `yolo_detect.py` script provides a basic example that shows how to load a model, run inference on an image source, parse the inference results, and display boxes around each detected class in the image. This script shows how to work with YOLO models in Python, and it can be used as a starting point for more advanced applications. 

To run inference with a yolov8s model on a USB camera at 1280x720 resolution, issue:

```
yolo_detect.py --model yolov8s.pt --source usb0 --resolution 1280x720
```

Here are all the arguments for yolo_detect.py:

- `model`: Path to a model file (e.g. `my_model.pt`). If the model isn't found, it will default to using `yolov8s.pt`.
- `source`: Source to run inference on. The options are:
    - Image file (example: `test.jpg`)
    - Folder of images (example: `my_images/test`)
    - Video file (example: `testvid.mp4`)
    - Index of a connected USB camera (example: `usb0`)
- `thresh` (optional): Minimum confidence threshold for displaying detected objects. Default value is 0.5 (example: `0.4`)
- `resolution` (optional): Resolution in WxH to display inference results at. If not specified, the program will match use the source resolution. (example: `1280x720`)
