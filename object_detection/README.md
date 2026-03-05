# About
- This is the Object Detection module of the SPOT Assistant Project.
- Still a work in progress! 

# Installation
- Current working environment
    - Windows 11
    - Python 3.10.11

- Install the required libraries: `pip install -r object_detection/requirements.txt`
    - If you want to use GPU and CUDA - go to https://pytorch.org/get-started/locally/ and pick the version that fits your CUDA version.

# Usage
- `network_compute_server.py` is the main program that will start up a server with the object detection model and link it to SPOT to be used by other scripts.
    - To set up the server:
        - `py network_compute_server.py -m models/yolov5/yolov5.pt 192.168.80.3`
    - The default name of the server is currently `fetch-server` but can be changed with the command above and the option `-n` or `--name`.

- `fetch.py` is currently the test code to use the model - basically ripped from the fetch tutorial. If this can fully use the model then the this code can be used as a blueprint for creating code utilizing models from servers.
    - To use:
        - `py fetch.py -s fetch-server -m yolov5 192.168.80.3`
    - If you changed the server name, then simply replace `fetch-server` with the name.