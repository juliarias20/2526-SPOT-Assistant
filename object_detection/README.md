# Object Detection
## About
- This module is to be used on its own for general object detection or with other modules for autonomous assitant functions.

## Installation
- Current working environment
    - Windows 11
    - Python 3.10.11

- Install the required libraries: `pip install -r object_detection/requirements.txt`
    - If you want to use GPU and CUDA - go to https://pytorch.org/get-started/locally/ and pick the version that fits your CUDA version.

## Usage
- `network_compute_server.py` is the main program that will start up a server with the object detection model and link it to SPOT to be used by other scripts.
    - To set up the server:
        - `py network_compute_server.py -m models/x_block_model/x_block_model.pt 192.168.80.3`
    - The default name of the server is currently `fetch-server` but can be changed with the command above and the option `-n` or `--name`.
    - The model can be changed by changing the path after the `-m`. 
    - Multiple models can be added by repeating `-m` option.

- `fetch.py` is currently the test code to use the model - basically ripped from the fetch tutorial. If this can fully use the model then the this code can be used as a blueprint for creating code utilizing models from servers.
    - To use:
        - `py fetch.py -s fetch-server -m x_block_model -o x-block 192.168.80.3`
    - If you changed the server name, then simply replace `fetch-server` with the name.
    - The file currently has the name of the object detection hardcoded in - if you want to use a different model that detects different objects, change the appropriate name (currently `x-block`) at line 210.
    - `-o` option specifies the object to be detected and grabbed.

- `live_cam_detection.py` lets you see through one of the cameras while in control of the robot.
    - To use:
        - `py live_cam_detection.py -i frontleft_fisheye_image -s fetch-server -m x_block_model 192.168.80.3`
    - You can change the camera to another by changing the `frontleft_fisheye_image` from above.
    - The default name of the server is currently `fetch-server` but can be changed with the command above and the option `-n` or `--name`.
    - The model can be changed by changing the model name after the `-m`.
    - Additional cameras can be used by adding more `-i` arguments.
    - `-o` is an optional argument that specifies object to be detected. It is only used by vocabulary based models.

- To create new object detection models, use the notebook `YOLO_SPOT.ipynb`, open in Google Colab, and follow the directions there. 
    - Once you have downloaded the new model, make sure to create a directory under `/models` with the desired name for the model and put the `.pt` file there.

## Dataset
- We have uploaded our datasets in Roboflow: https://universe.roboflow.com/spot-datasets-ectut

## Image Sources
    - `back_depth`
    - `back_depth_in_visual_frame`
    - `back_fisheye_image`
    - `frontleft_depth`
    - `frontleft_depth_in_visual_frame`
    - `frontleft_fisheye_image`
    - `frontright_depth`
    - `frontright_depth_in_visual_frame`
    - `frontright_fisheye_image`
    - `hand_color_image`
    - `hand_color_in_hand_depth_frame`
    - `hand_depth`
    - `hand_depth_in_hand_color_frame`
    - `hand_image`
    - `left_depth`
    - `left_depth_in_visual_frame`
    - `left_fisheye_image`
    - `right_depth`
    - `right_depth_in_visual_frame`
    - `right_fisheye_image`