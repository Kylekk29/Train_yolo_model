# Toggle Raspberry Pi GPIO - Smart Lamp Example
One helpful way to use YOLO detections is to control real-world devices when certain events occur. The [toggle_pi_gpio.py](toggle_pi_gpio.py) example shows how to set up a "smart lamp" that turns on when a person is detected within a certain area of the camera's view. This example is a useful starting point to see how to toggle the Raspberry Pi's General-Purpose Input/Output (GPIO) pins using YOLO and Python.

<p align="center">
  <img src="../../doc/smart-lamp-example.png">
</p>

This README shows how connect the Raspberry Pi to an [AC Power Relay](https://amzn.to/3WJASk8), which is controlled with the Pi's GPIO to turn devices on and off when certain objects are detected. It also shows how to configure and run the toggle_pi_gpio.py script on the Pi.

## Requirements

Hardware required:
* [Raspberry Pi 5](https://amzn.to/3Qo4wrX) or Raspberry Pi 4
* [DIGITAL LOGGERS IoT AC Power Relay](https://amzn.to/3WJASk8) - this is an easy-to-use and safe device for turning wall-connected devices (such as lights) on and off with the Pi
* USB webcam (such as the [Logitech c920](https://amzn.to/40Q6PK7)) or a Picamera (such as the [Picamera Module 3](https://amzn.to/3PXfggn)

Software setup:
* Before running this example, set up the Raspberry Pi to run YOLO models by following the instructions in our [How to Run YOLO Detection Models on the Raspberry Pi](https://www.ejtech.io/learn/yolo-on-raspberry-pi) article.
* After following the steps in the article, you should have a folder named `yolo` that contains a `yolo_detect.py` script and a `yolo11n_ncnn_model` folder.
* You should be able to run `yolo_detect.py --model=yolo11n_ncnn_model --source=usb0` and see a window display showing detection results on the live webcam feed.



