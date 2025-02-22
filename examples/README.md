## YOLO Application Examples
This folder contains example code showing different applications you can build around YOLO models.

### Candy Calorie Counter
The [candy_calorie_counter](candy_calorie_counter) example uses a custom YOLO model that's trained to identify popular types of candy (Skittles, Snickers, etc). When candy is placed in front of the camera, the application checks the number of calories and grams of sugar contained in each piece of candy, and it reports the total calories and sugar. It's a basic example of how to use detected object classes to look up information about each object.

### Using YOLO With Multiple Cameras
The [multi_camera](multi_camera) example shows an efficient way to run YOLO models on multiple camera streams using Python multiprocessing.

### Toggle Raspberry Pi GPIO - Smart Lamp
The [toggle_pi_gpio](toggle_pi_gpio) example shows how to set up a "smart lamp" that turns on when a person is detected within a certain area of the camera's view. This example is a useful starting point to see how to toggle the Raspberry Pi's General-Purpose Input/Output (GPIO) pins using YOLO and Python. It also shows how to work with detected object coordinates and make decisions based on where an object is located on the screen.
