# Run YOLO detection on Raspberry Pi using a Picamera

Using a Picamera with YOLO is a little different than a standard USB camera because we have to use the Picamera2-Python library to read from the Picamera. Unfortunately, Picamera2-Python can't be easily installed in a virtual environment. Instead, we have to install the Ultralytics and OpenCV libraries in the native Python environment. This is fine but it's also a little scary because it can cause version conflicts with existing Python libraries.

**So, it is highly recommended to start with a fresh install of the latest version of 64-bit Raspberry Pi OS if you are using the Picamera with YOLO!**

Anyway, follow the steps below to run YOLO detection on a Picamera!

## Requirements
Raspberry Pi 4 or Raspberry Pi 5

## Setup Steps

1. Use [Raspberry Pi Imager](https://www.raspberrypi.com/software/) to flash the latest version of 64-bit Raspberry Pi OS onto an SD card.
2. Plug in a monitor, keyboard, mouse, and Picamera into the Raspberry Pi and turn it on.
3. Open a terminal and update the Pi using `sudo apt update` and `sudo apt upgrade`.
4. Raspberry Pi OS comes pre-installed with the necessary Picamera libraries. Verify your Picamera is working by running `rpicam-hello` in the terminal. A window should appear showing the live Picamera view. If it doesn't try replugging the camera. If it still doesn't work, your camera may be broken :frowning_face:.
5. Install OpenCV with `apt` by issuing `sudo apt install -y python3-opencv opencv-data`.
6. Install the Ultralytics and NCNN libraries by issuing `pip3 install ultralytics ncnn --break-system-packages`. This may take up to 15 minutes. Make sure you have a stable internet connection. If it fails, try re-running the command again.
7. Make a directory named `yolo` and move into it by issuing `mkdir ~/yolo && cd ~/yolo`.
8. Download the `yolo11n.pt` model by issuing `yolo detect predict model=yolo11n.pt`. It will download into the `yolo` folder. You can also use a custom model (e.g. `my_model.pt`) like the one trained in the [YOLO Training Notebook](https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb) in this repository.
9. Export the model to NCNN format by ussing `yolo export model=yolo11n.pt format=ncnn`. (Using NCNN allows the model to run much faster on the Raspberry Pi.)
10. Download the `yolo_detect_picamera.py` script by issuing `wget https://raw.githubusercontent.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/refs/heads/main/examples/picamera/yolo_detect_picamera.py`.

## Run Detection Script
Now you're ready to run the script! Run it using the following command, where `--model` points to the NCNN model folder and `--resolution` is the resolution you want to run the camera at.

```
yolo_detect_picamera.py --model=yolo11n_ncnn_model --resolution=640x480
```

A window will appear showing the Picamera's live view with detections drawn on each frame.
