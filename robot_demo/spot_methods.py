"""
OBJECTIVE: Organizes a list of methods for SPOT to utilize upon intialization.

Methods:
    1. power_on :: intializes motors and turns on SPOT 
    2. power_off :: stops motors and safely shuts down SPOT
    4. stand :: sends the stand command to SPOT
    
    5. run_model :: main method to test "FETCH" tutorial methods 
    6. get_img_find_obj :: used to capture image and uses the model to find the object
    5. get_bounding_box_image :: based on the image captured and the model analysis, create a bounding box for SPOT to estimate coordinates
    6. find_center_px :: used to find the center of bounding box for grasp and motor control
    7. grasp_obj :: uses arm control to grasp the object 
    8. walk_to_obj :: allows SPOT to walk to object detected based on bounding box results
    9. pose_dist :: calculates the relative distance between SPOT and the target object 
"""
import argparse
import sys
import time
import os
import cv2
import numpy as np
import math
from dotenv import load_dotenv
from pathlib import Path

from bosdyn import geometry
import bosdyn.client
import bosdyn.client.util
from bosdyn.api import (basic_command_pb2, image_pb2, geometry_pb2, manipulation_api_pb2, network_compute_bridge_pb2)
from bosdyn.api.manipulation_api_pb2 import (ManipulationApiFeedbackRequest, ManipulationAPIRequest, WalkToObjectInImage)
from google.protobuf import wrappers_pb2

from bosdyn.client import create_standard_sdk, frame_helpers
from bosdyn.client.door import DoorClient
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.network_compute_bridge_client import (ExternalServerError,
                                                         NetworkComputeBridgeClient)
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.util import add_base_arguments, authenticate, setup_logging
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, math_helpers

kImageSources = [
    'frontleft_fisheye_image', 'frontright_fisheye_image', 'left_fisheye_image',
    'right_fisheye_image', 'back_fisheye_image'
]

def power_on(robot):
    try:
        robot.logger.info("Powering on SPOT...")
        robot.power_on(timeout_sec = 20)

        if robot.is_powered_on():
            print("SPOT is already powered on.")
        robot.logger.info("SPOT is powered on.")

    except Exception as e:
        print("Error while powering on SPOT: " + str(e))

def power_off(robot):
    try:
        robot.logger.info("Powering off SPOT...")
        robot.power_off(cut_immediately=False, timeout_sec = 20)

        if robot.is_powered_on():
            print("SPOT power off failed.")
        robot.logger.info("SPOT safely powered off.")
    except Exception as e:
        print("Error while powering off SPOT " + str(e))

def stand(robot):
    robot.logger.info('Sending "stand" command to SPOT')
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    blocking_stand(command_client, timeout_sec = 10)
    robot.logger.info('SPOT is now standing.')

def run_model(options, kImageSources, robot):    
    # Use lease client to take motor control from tablet
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease_client.acquire()
    lease_keepalive = LeaseKeepAlive(lease_client, must_acquire=False, return_at_exit=True)

    # Declare clients to be utilized for arm/robot manipulation, model analysis, and robot state
    network_compute_client = robot.ensure_client(NetworkComputeBridgeClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            # Store the position of the hand at the last toy drop point.
            vision_tform_hand_at_drop = None

            while True:
                holding_toy = False
                while not holding_toy:
                    # Capture an image and run ML on it.
                    object, image, vision_tform_object = get_img_find_obj(
                        network_compute_client, options.ml_service, options.model,
                        options.confidence_object, kImageSources, 'x-block')

                    if object is None:
                        # Didn't find anything, keep searching.
                        continue

                    # If we have already dropped the toy off, make sure it has moved a sufficient amount before
                    # picking it up again
                    if vision_tform_hand_at_drop is not None and pose_dist(
                            vision_tform_hand_at_drop, vision_tform_object) < 0.5:
                        print('Found object, but it hasn\'t moved.  Waiting...')
                        time.sleep(1)
                        continue

                    print('Found object...')
                    
                    # The ML result is a bounding box.  Find the center.
                    (center_px_x, center_px_y) = find_center_px(object.image_properties.coordinates)
                    
                    # Once found, walk to object using the coordinates calculated.
                    walk_to_obj(center_px_x, center_px_y, robot, options, image, manipulation_api_client)

                    #Once the robot has reached the target, use the coordinates to grasp the object
                    grasp_obj(center_px_x, center_px_y, options, image, object, manipulation_api_client, robot_state_client, command_client)
    
def pose_dist(pose1, pose2):
    diff_vec = [pose1.x - pose2.x, pose1.y - pose2.y, pose1.z - pose2.z]
    return np.linalg.norm(diff_vec)

def get_img_find_obj(network_compute_client, server, model, confidence, image_sources, label):
    for source in image_sources:
        # Build a network compute request for this image source.
        image_source_and_service = network_compute_bridge_pb2.ImageSourceAndService(
            image_source=source)

        # Input data:
        #   model name
        #   minimum confidence (between 0 and 1)
        #   if we should automatically rotate the image
        input_data = network_compute_bridge_pb2.NetworkComputeInputData(
            image_source_and_service=image_source_and_service, model_name=model,
            min_confidence=confidence, rotate_image=network_compute_bridge_pb2.
            NetworkComputeInputData.ROTATE_IMAGE_ALIGN_HORIZONTAL)

        # Server data: the service name
        server_data = network_compute_bridge_pb2.NetworkComputeServerConfiguration(
            service_name=server)

        # Pack and send the request.
        process_img_req = network_compute_bridge_pb2.NetworkComputeRequest(
            input_data=input_data, server_config=server_data)

        try:
            resp = network_compute_client.network_compute_bridge_command(process_img_req)
        except ExternalServerError:
            # This sometimes happens if the NCB is unreachable due to intermittent wifi failures.
            print('Error connecting to network compute bridge. This may be temporary.')
            return None, None, None

        best_obj = None
        highest_conf = 0.0
        best_vision_tform_obj = None

        img = get_bounding_box_image(resp)
        image_full = resp.image_response

        # Show the image
        cv2.imshow("Fetch", img)
        cv2.waitKey(15)

        if len(resp.object_in_image) > 0:
            for obj in resp.object_in_image:
                # Get the label
                obj_label = obj.name.split('_label_')[-1]
                if obj_label != label:
                    continue
                conf_msg = wrappers_pb2.FloatValue()
                obj.additional_properties.Unpack(conf_msg)
                conf = conf_msg.value

                try:
                    vision_tform_obj = frame_helpers.get_a_tform_b(
                        obj.transforms_snapshot, frame_helpers.VISION_FRAME_NAME,
                        obj.image_properties.frame_name_image_coordinates)
                except bosdyn.client.frame_helpers.ValidateFrameTreeError:
                    # No depth data available.
                    vision_tform_obj = None

                if conf > highest_conf and vision_tform_obj is not None:
                    highest_conf = conf
                    best_obj = obj
                    best_vision_tform_obj = vision_tform_obj

        if best_obj is not None:
            return best_obj, image_full, best_vision_tform_obj

    return None, None, None

def get_bounding_box_image(response):
    dtype = np.uint8
    img = np.fromstring(response.image_response.shot.image.data, dtype=dtype)
    if response.image_response.shot.image.format == image_pb2.Image.FORMAT_RAW:
        img = img.reshape(response.image_response.shot.image.rows,
                          response.image_response.shot.image.cols)
    else:
        img = cv2.imdecode(img, -1)

    # Convert to BGR so we can draw colors
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw bounding boxes in the image for all the detections.
    for obj in response.object_in_image:
        conf_msg = wrappers_pb2.FloatValue()
        obj.additional_properties.Unpack(conf_msg)
        confidence = conf_msg.value

        polygon = []
        min_x = float('inf')
        min_y = float('inf')
        for v in obj.image_properties.coordinates.vertexes:
            polygon.append([v.x, v.y])
            min_x = min(min_x, v.x)
            min_y = min(min_y, v.y)

        polygon = np.array(polygon, np.int32)
        polygon = polygon.reshape((-1, 1, 2))
        cv2.polylines(img, [polygon], True, (0, 255, 0), 2)

        caption = "{} {:.3f}".format(obj.name, confidence)
        cv2.putText(img, caption, (int(min_x), int(min_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

    return img

def find_center_px(polygon):
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    for vert in polygon.vertexes:
        if vert.x < min_x:
            min_x = vert.x
        if vert.y < min_y:
            min_y = vert.y
        if vert.x > max_x:
            max_x = vert.x
        if vert.y > max_y:
            max_y = vert.y
    x = math.fabs(max_x - min_x) / 2.0 + min_x
    y = math.fabs(max_y - min_y) / 2.0 + min_y
    print(f"x: {x}, y: {y}")
    return (x, y)

def grasp_obj(center_px_x, center_px_y, options, image, object, manipulation_api_client, robot_state_client, command_client):
    # Request Pick Up on that pixel.
    pick_vec = geometry_pb2.Vec2(x=center_px_x, y=center_px_y)
    print(pick_vec)
    grasp = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=pick_vec,
        transforms_snapshot_for_camera=image.shot.transforms_snapshot,
        frame_name_image_sensor=image.shot.frame_name_image_sensor,
        camera_model=image.source.pinhole)
    
    """
    # We can specify where in the gripper we want to grasp. About halfway is generally good for
    # small objects like this. For a bigger object like a shoe, 0 is better (use the entire
    # gripper)
    grasp.grasp_params.grasp_palm_to_fingertip = 0.5

    # Tell the grasping system that we want a top-down grasp.

    # Add a constraint that requests that the x-axis of the gripper is pointing in the
    # negative-z direction in the vision frame.

    # The axis on the gripper is the x-axis.
    axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

    # The axis in the vision frame is the negative z-axis
    axis_to_align_with_ewrt_vision = geometry_pb2.Vec3(x=0, y=0, z=-1)

    # Add the vector constraint to our proto.
    constraint = grasp.grasp_params.allowable_orientation.add()
    constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
        axis_on_gripper_ewrt_gripper)
    constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
        axis_to_align_with_ewrt_vision)

    # We'll take anything within about 15 degrees for top-down or horizontal grasps.
    constraint.vector_alignment_with_tolerance.threshold_radians = 0.25

    # Specify the frame we're using.
    grasp.grasp_params.grasp_params_frame_name = frame_helpers.VISION_FRAME_NAME
    """
    # Optionally add a grasp constraint.  This lets you tell the robot you only want top-down grasps or side-on grasps.
    add_grasp_constraint(options, grasp, robot_state_client)

    # Build the proto
    grasp_request = manipulation_api_pb2.ManipulationApiRequest(
        pick_object_in_image=grasp)

    # Send the request
    print('Sending grasp request...')
    cmd_response = manipulation_api_client.manipulation_api_command(
        manipulation_api_request=grasp_request)

    # Wait for the grasp to finish
    grasp_done = False
    failed = False
    time_start = time.time()
    while not grasp_done:
        feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd_response.manipulation_cmd_id)

        # Send a request for feedback
        response = manipulation_api_client.manipulation_api_feedback_command(
            manipulation_api_feedback_request=feedback_request)

        current_state = response.current_state
        current_time = time.time() - time_start
        print(
            'Current state ({time:.1f} sec): {state}'.format(
                time=current_time,
                state=manipulation_api_pb2.ManipulationFeedbackState.Name(
                    current_state)), end='                \r')
        sys.stdout.flush()

        failed_states = [
            manipulation_api_pb2.MANIP_STATE_GRASP_FAILED,
            manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION,
            manipulation_api_pb2.MANIP_STATE_GRASP_FAILED_TO_RAYCAST_INTO_MAP,
            manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_WAITING_DATA_AT_EDGE
        ]
        failed = current_state in failed_states
        grasp_done = current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED or failed

        time.sleep(0.1)

    holding_toy = not failed

    # Move the arm to a carry position.
    print('')
    print('Grasp finished.')
    carry_cmd = RobotCommandBuilder.arm_carry_command()
    command_client.robot_command(carry_cmd)

    # Wait for the carry command to finish
    time.sleep(0.75)

def add_grasp_constraint(config, grasp, robot_state_client):
    # There are 3 types of constraints:
    #   1. Vector alignment
    #   2. Full rotation
    #   3. Squeeze grasp
    #
    # You can specify more than one if you want and they will be OR'ed together.

    # For these options, we'll use a vector alignment constraint.
    use_vector_constraint = config.force_top_down_grasp or config.force_horizontal_grasp

    # Specify the frame we're using.
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

    if use_vector_constraint:
        if config.force_top_down_grasp:
            # Add a constraint that requests that the x-axis of the gripper is pointing in the
            # negative-z direction in the vision frame.

            # The axis on the gripper is the x-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

            # The axis in the vision frame is the negative z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)
    if config.force_horizontal_grasp:
            # Add a constraint that requests that the y-axis of the gripper is pointing in the
            # positive-z direction in the vision frame.  That means that the gripper is constrained to be rolled 90 degrees and pointed at the horizon.

            # The axis on the gripper is the y-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)

            # The axis in the vision frame is the positive z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=1)

            # Add the vector constraint to our proto.
            constraint = grasp.grasp_params.allowable_orientation.add()
            constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
                axis_on_gripper_ewrt_gripper)
            constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
                axis_to_align_with_ewrt_vo)

            # We'll take anything within about 10 degrees for top-down or horizontal grasps.
            constraint.vector_alignment_with_tolerance.threshold_radians = 0.17
    elif config.force_squeeze_grasp:
        # Tell the robot to just squeeze on the ground at the given point.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.squeeze_grasp.SetInParent()


def walk_to_obj(center_px_x, center_px_y, robot, config, image, manipulation_api_client):
    robot.logger.info("Walking to object at location (%s, %s)", center_px_x, center_px_y)

    walk_vec = geometry_pb2.Vec2(x=center_px_x, y=center_px_y)
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

    # Ask the robot to walk up to the object
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

    robot.logger.info('Arrived at object.')

def main(argv):

    """TEST PROGRAM FOR METHODS"""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-s', '--ml-service',
                        help='Service name of external machine learning server.', required=True)
    parser.add_argument('-m', '--model', help='Model name running on the external server.',
                        required=True)
    parser.add_argument('-p', '--person-model',
                        help='Person detection model name running on the external server.')
    parser.add_argument('-c', '--confidence-object',
                        help='Minimum confidence to return an object for the object (0.0 to 1.0)',
                        default=0.5, type=float)
    parser.add_argument('-e', '--confidence-person',
                        help='Minimum confidence for person detection (0.0 to 1.0)', default=0.6,
                        type=float)
    parser.add_argument('-d', '--distance', help='Distance from object to walk to (meters).',
                        default=None, type=float)
    parser.add_argument('-t', '--force-top-down-grasp',
                        help='Force the robot to use a top-down grasp (vector_alignment demo)',
                        action='store_true')
    parser.add_argument('-f', '--force-horizontal-grasp',
                        help='Force the robot to use a horizontal grasp (vector_alignment demo)',
                        action='store_true')
    parser.add_argument(
        '-r', '--force-45-angle-grasp',
        help='Force the robot to use a 45 degree angled down grasp (rotation_with_tolerance demo)',
        action='store_true')
    parser.add_argument('-s', '--force-squeeze-grasp',
                        help='Force the robot to use a squeeze grasp', action='store_true')
    
    options = parser.parse_args(argv)
    num = 0
    if options.force_top_down_grasp:
        num += 1
    if options.force_horizontal_grasp:
        num += 1
    if options.force_45_angle_grasp:
        num += 1
    if options.force_squeeze_grasp:
        num += 1


    # Establish environmental variables to initialize SPOT robot
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path)

    SPOT_USERNAME = os.getenv("SPOT_USERNAME")
    SPOT_PASSWORD = os.getenv("SPOT_PASSWORD")
    SPOT_IP = os.getenv("SPOT_IP")

    if not all([SPOT_USERNAME, SPOT_PASSWORD, SPOT_IP]):
        raise RuntimeError("Missing SPOT_USERNAME, SPOT_PASSWORD, or SPOT_IP environment variables.")
 
    sdk = bosdyn.client.create_standard_sdk("ModelTestClient")
    robot = sdk.create_robot(SPOT_IP)

    robot.authenticate(SPOT_USERNAME, SPOT_PASSWORD)
    robot.time_sync.wait_for_sync()

    stand(robot)

    run_model(options, kImageSources, robot)

