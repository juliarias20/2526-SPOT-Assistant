"""
Program: Initial Demo

Summary: Given a set of data, the program will utilize an image and CV library
         to confirm visual conditions and mobilize SPOT to symbolize 
         robotic integration.

Improvements: 
    1. Include more complex actions for SPOT using the SPOT SDK module. 
    2. Transition from analyzing color to analyzing image of an object.
        a. Potentially needs: GPU model integration + response from model ---> how is the reponse portrayed in data?

Mobilization Code: SPOT SDK --> arm_walk_to_object.py

"""

# Values obtained from SPOT SDK file
import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand

import argparse
import sys
import time
import os
import cv2
import numpy as np

def main():

    # Establish SPOT variables from .env file 
    SPOT_USERNAME = os.getenv('SPOT_USERNAME')
    SPOT_PASSWORD = os.getenv('SPOT_PASSWORD')
    SPOT_IP = os.getenv('SPOT_IP')

    if not all(SPOT_USERNAME, SPOT_PASSWORD, SPOT_IP):
        raise ValueError("SPOT credentials not found in file. Please try again.")
    
    # Create SPOT SDK variables for robot configuration
    sdk = bosdyn.client.create_standard_sdk('StandClient')
    robot = sdk.create_robot(SPOT_USERNAME)
    bostyn.client.util.authenticate(robot)

    # Authenticate robot with environmental variables
    robot.authenticate(SPOT_USERNAME, SPOT_PASSWORD)
    robot.time_sync.wait_for_sync()

    print("Starting demo...")

    #Establish data collection by importing the image
    cv_image = cv2.imread("IMAGE NAME")

    # Convert to RGB formatting for program to analyze
    cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    #Split into three variables: R, G, B
    R, G, B = cv2.split(cv_image_rgb)

    # Create a boolean value to decide where red is dominant
    # Thresholds acn be modified depending on lighting and Spot’s camera exposure
    red_mask = (R > 100) & (R > G * 1.3) & (R > B * 1.3)

    # Convert boolean mask to uint8 for integer analysis
    red_mask = red_mask.astype(np.uint8) * 255

    # Count red pixels to identify the ratio (analyzed)
    red_pixels = cv2.countNonZero(red_mask)
    total_pixels = image_rgb.shape[0] * image_rgb.shape[1]
    red_ratio = red_pixels / total_pixels

    # if data confirmed visual conditions: 
    # Mobilize SPOT to symbolize robotic integration  ---->  SPOT will stand: 
    if red_radio > #include number: 
        robot.logger.info('Sending "stand" command to SPOT')
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec = 10)
        robot.logger.info('SPOT is standing!')
    else:
        print("No signficant amount of red detected within the image provided. SPOT integration failed.")   

if __name__ == "__main__":
    main()
