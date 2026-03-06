"""
Program: watter_bottle_grasp.py

Summary: Using Manipulation API Service, SPOT will take an image, present it to the user
         and respond autonomously to pick up the object of focus. The program will utilize methods
         and structure from arm_door.py, arm_grasp.py, and arm_walk_to_object.py, which are documented 
         in the Boston Dynamics SPOT SDK Software.

Methods needed:

"arm_walk_to_object"
    1. walk_to_object -- request SPOT to walk over to object to prepare for manipulation
    2. mouse_response -- wait for user to click on object of focus in the image presented
    3. float(x) -- converting integer to float value

"arm_grasp"
    4. verify_estop -- verify that the robot is safe to operate and is not in emergency stop mode
    5. arm_grasp -- command SPOT to take an image, wait for user selection, and pick up the object
    6. grasp_options -- specify whether to grasp from above or side angle.

Robot initialization functions:
    7. power_on -- power on the robot
    8. power_off -- power off the robot

"""
import argparse
import math
import sys
import time

import cv2
import numpy as np

from bosdyn import geometry
from bosdyn.api import basic_command_pb2, geometry_pb2, manipulation_api_pb2
from bosdyn.api.manipulation_api_pb2 import (ManipulationApiFeedbackRequest, ManipulationAPIRequest, WalkToObjectInImage)
from google.protobuf import wrappers_pb2

from bosdyn.client import create_standard_sdk, frame_helpers
from bosdyn.client.door import DoorClient
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.util import add_base_arguments, authenticate, setup_logging

def power_on(robot):
    try:
        robot.logger.info('Powering on robot...')
        robot.power_on(timeout_sec=20)

        if robot.is_powered_on():
            print('Robot already powered on.')
        robot.logger.info('Robot powered on.')
    except Exception as e:
        print("Error while powering on SPOT: " + str(e))

def power_off(robot):
    try:
        robot.logger.info('Powering off robot...')
        robot.power_off(cut_immediately=False, timeout_sec=20)

        if robot.is_powered_on():
            print('Robot power off failed.')
        robot.logger.info('Robot safely powered off.')
    except Exception as e:
        print("Error while powering off SPOT: " + str(e))

def stand(robot):
    robot.logger.info('Sending "stand" command to SPOT')
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    blocking_stand(command_client, timeout_sec = 10)
    robot.logger.info('SPOT is standing!')

def walk_to_object(config):

    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path)

    SPOT_USERNAME = os.getenv("SPOT_USERNAME")
    SPOT_PASSWORD = os.getenv("SPOT_PASSWORD")
    SPOT_IP = os.getenv("SPOT_IP")

    if not all([SPOT_USERNAME, SPOT_PASSWORD, SPOT_IP]):
        raise RuntimeError("Missing SPOT_USERNAME, SPOT_PASSWORD, or SPOT_IP environment variables.")

    sdk = bosdyn.client.create_standard_sdk("InitialDemoClient")
    robot = sdk.create_robot(SPOT_IP)

    robot.authenticate(SPOT_USERNAME, SPOT_PASSWORD)
    robot.time_sync.wait_for_sync()

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease_client.acquire()
    lease_keepalive = LeaseKeepAlive(lease_client, must_acquire=False, return_at_exit=True)
    
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        self.power_on(robot)

        self.stand(robot)

        #Take a picture with a camera with the target object in sight
        robot.logger.info('Getting image from: %s', config.image_source)
        image_responses = image_client.get_image_from_sources([config.image_source])

        if len(image_responses) != 1: #Number of images taken does not = 1
            print(f'Invalid number of images: {len(image_responses)}')
            print(image_responses)
            assert False

        # Convert the image into a presentable format for user-click
        image = image_responses[0] #Obtain the first image
        if image.shot.image.pixel_format == image.pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            dtype = np.uint16
        else:
            dtype = np.uint8
        img = np.fromstring(image.shot.image.data, dtype=dtype)
        if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
                img = img.reshape(image.shot.image.rows, image.shot.image.cols)
        else:
            img = cv2.imdecode(img, -1)

        # Show image to user and wait for click
        robot.logger.info('Click on an object to walk up to...')
        image_title = 'Click to walk up to an object'
        cv2.namedWindow(image_title)
        cv2.setMouseCallback(image_title, cv_mouse_callback) ## INCLUDE THIS METHOD

        global g_image_click, g_image_display
        g_image_display = img
        cv2.imshow(image_title, g_image_display)
        while g_image_click is None:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                # Quit
                print('"q" pressed, exiting.')
                exit(0)

        robot.logger.info('Walking to object at image location (%s, %s)', g_image_click[0],
                          g_image_click[1])

        walk_vec = geometry_pb2.Vec2(x=g_image_click[0], y=g_image_click[1])

        # Optionally populate the offset distance parameter.
        if config.distance is None:
            offset_distance = None
        else:
            offset_distance = wrappers_pb2.FloatValue(value=config.distance)

        # Build the proto
        walk_to = manipulation_api_pb2.WalkToObjectInImage(
            pixel_xy=walk_vec, transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            frame_name_image_sensor=image.shot.frame_name_image_sensor,
            camera_model=image.source.pinhole, offset_distance=offset_distance)

        # Ask the robot to pick up the object
        walk_to_request = manipulation_api_pb2.ManipulationApiRequest(
            walk_to_object_in_image=walk_to)

        # Send the request
        cmd_response = manipulation_api_client.manipulation_api_command(
            manipulation_api_request=walk_to_request)

        # Get feedback from the robot
        while True:
            time.sleep(0.25)
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id)

            # Send the request
            response = manipulation_api_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)

            print('Current state: ',
                  manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state))

            if response.current_state == manipulation_api_pb2.MANIP_STATE_DONE:
                break

        robot.logger.info('Finished.')
        robot.logger.info('Sitting down and turning off.')

        self.power_off(robot)

def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click, g_image_display
    clone = g_image_display.copy()
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    else:
        # Draw some lines on the image.
        # print('mouse', x, y)
        color = (30, 30, 30)
        thickness = 2
        image_title = 'Click to walk up to something'
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow(image_title, clone)

def arg_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{repr(x)} not a number')
    return x

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-i', '--image-source', help='Get image from source',
                        default='frontleft_fisheye_image')
    parser.add_argument('-d', '--distance', help='Distance from object to walk to (meters).',
                        default=None, type=arg_float)
    options = parser.parse_args()

    try:
        walk_to_object(options)
        return True
    except Exception as e:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception('Threw an exception: ' + str(e))
        return False


if __name__ == '__main__':
    if not main():
        sys.exit(1)