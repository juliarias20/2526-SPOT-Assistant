import sys
import argparse

import cv2
import numpy as np
from google.protobuf import wrappers_pb2

import bosdyn.client
import bosdyn.client.util
from bosdyn.client import frame_helpers
from bosdyn.api import network_compute_bridge_pb2, image_pb2
from bosdyn.client.network_compute_bridge_client import NetworkComputeBridgeClient

def get_obj_and_img(network_compute_client, server, model, confidence, sources):
    for source in sources:
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

        resp = network_compute_client.network_compute_bridge_command(process_img_req)
        
        # Put bounding boxes in the image
        img = get_bounding_box_image(resp)

        # Show the image
        cv2.imshow(f"Camera: {source}", img)
        
        # Wait for 15 milliseconds
        cv2.waitKey(15)

def get_bounding_box_image(response):
    dtype = np.uint8
    img = np.fromstring(response.image_response.shot.image.data, dtype=dtype)
    if response.image_response.shot.image.format == image_pb2.Image.FORMAT_RAW:
        img = img.reshape(response.image_response.shot.image.rows,
                          response.image_response.shot.image.cols)
    else:
        img = cv2.imdecode(img, -1)

    # Convert to BGR so we can draw colors
    if response.image_response.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
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

        caption = "{} {:.3f}".format(obj.name.split('_label_')[-1], confidence)
        cv2.putText(img, caption, (int(min_x), int(min_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

    return img

def main(argv):
    # ARGUMENTS
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-i', '--image-sources', help='Get image from source(s)',
                        action='append')
    parser.add_argument('-s', '--ml-service',
                        help='Service name of external machine learning server.', required=True)
    parser.add_argument('-m', '--model', help='Model name running on the external server.',
                        required=True)
    parser.add_argument('-c', '--confidence-object',
                        help='Minimum confidence to return an object for the detection (0.0 - 1.0)',
                        default=0.5, type=float)
    
    options = parser.parse_args(argv)
    
    # Register robot
    sdk = bosdyn.client.create_standard_sdk('SpotLiveDetectionClient')
    sdk.register_service_client(NetworkComputeBridgeClient)
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    
    robot.time_sync.wait_for_sync()
    
    network_compute_client = robot.ensure_client(NetworkComputeBridgeClient.default_service_name)
    
    # Main loop
    while True:
        get_obj_and_img(network_compute_client, options.ml_service, options.model, 
                        options.confidence_object, options.image_sources)
        
if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)