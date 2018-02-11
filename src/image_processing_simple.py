#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('barc')
import sys
import rospy
import cv2
from std_msgs.msg import String, Int32, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from scipy import signal
from math import sqrt, atan, pi

point_num = 0

# Parameter Default Values
display_image = False
publish_image = False
calibrate_transform = False
image_calibrated = True
calibration_pts = None

def print_point_callback(event, x, y, flags, param):
    """
    This callback function calibrates the image perspective transform
    """
    global point_num
    global image_calibrated
    global calibration_pts
    global transformation
    global image_processor_global
    if event == cv2.EVENT_LBUTTONDOWN:
        point_num = point_num + 1
        if (point_num == 1):
            print('Start at upper left corner. Select points clockwise.')
            calibration_pts = np.float32([[x + 0.0,y + 0.0]])
        elif (point_num <= 4):
            calibration_pts = np.append(calibration_pts, np.float32([[x + 0.0,y + 0.0]]),axis = 0)
        
        if (point_num == 4):
            # Apply Projection Transform
            # ref points [TOPLEFT, TOPRIGHT, BOTTOMRIGHT, BOTTOMLEFT]
            ref_pts = np.float32([[image_processor_global.x_offset,0], \
                [image_processor_global.x_width + image_processor_global.x_offset,0], \
                [image_processor_global.x_width + image_processor_global.x_offset, image_processor_global.y_height], \
                [image_processor_global.x_offset, image_processor_global.y_height]])

            image_processor_global.projection_dim = (image_processor_global.x_width + image_processor_global.x_offset * 2, image_processor_global.y_height)
            image_processor_global.projection_transform = cv2.getPerspectiveTransform(calibration_pts, ref_pts) 
            image_calibrated = True
            cv2.destroyWindow("Calibrate Image Transform")
        display_val ='Pt{}: x:{} y:{}'.format(point_num, x, y)
        print(display_val)

def find_offset_in_lane(img,x,y,width):
    """
    Returns the difference in x and y positions
    operates on pixels. Return value is pixel offset from nominal
    """
    x_left = x
    x_right = x
    while(x_left > 0 and not img[y, x_left]):
        x_left = x_left - 1
    while(x_right < width and not img[y, x_right]):
        x_right = x_right + 1
    return (x_left, x_right)



class image_processor:
    """
    This class takes image messages from the USB Camera and converts them to a cv2 format
    subsequently it converts the image to grayscale and performs a perspective and hanning transform.
    Finally, it outputs a delta value indicating the offset of the vehicle from the center of the lane
    """

    def __init__(self):
        global display_image, publish_image, calibrate_transform
        global calibration_pts
        
        #Create ROS Interfaces
        self.offset_lane_pub = rospy.Publisher("lane_offset", Float32,queue_size=10)
        self.cvfail_pub = rospy.Publisher("cv_abort", Int32, queue_size=10)
       
        self.image_pub = rospy.Publisher("cv_image", Image, queue_size = 10)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/barc_cam/image_raw", Image,self.callback_image)
        
        #Get Launch File Parameters and configure node
        calibrate_transform = rospy.get_param("/image_processing/calibrate_transform")
        display_image = rospy.get_param("/image_processing/display_image")
        publish_image = rospy.get_param("/image_processing/publish_image")
        global image_calibrated
        
        # Projection Transform Parameters
        self.x_offset = 100
        self.x_width = 200
        self.y_height = 400
        if calibrate_transform:
            image_calibrated = False
            cv2.namedWindow("Calibrate Image Transform")
            cv2.setMouseCallback("Calibrate Image Transform", print_point_callback)
        else:
            image_calibrated = True
            calibration_pts = np.float32( ([rospy.get_param("/image_processing/upperLeftX"), rospy.get_param("/image_processing/upperLeftY")], \
                                [rospy.get_param("/image_processing/upperRightX"), rospy.get_param("/image_processing/upperRightY")], \
                                [rospy.get_param("/image_processing/lowerRightX"), rospy.get_param("/image_processing/lowerRightY")], \
                                [rospy.get_param("/image_processing/lowerLeftX"), rospy.get_param("/image_processing/lowerLeftY")]))
        
            # Apply Projection Transform
            # ref points [TOPLEFT, TOPRIGHT, BOTTOMRIGHT, BOTTOMLEFT]
            ref_pts = np.float32([[self.x_offset,0], \
                [self.x_width + self.x_offset,0], \
                [self.x_width + self.x_offset, self.y_height], \
                [self.x_offset, self.y_height]])

            self.projection_dim = (self.x_width + self.x_offset * 2, self.y_height)
            self.projection_transform = cv2.getPerspectiveTransform(calibration_pts, ref_pts) 

        self.minimumContourSize = 10
        self.image_threshold = 20
        self.y_offset_pixels_cg = 70
        self.num_nonvalid_img = 0
        self.num_failed_img_before_abort = 30
        self.half_vehicle_width_pixels = (260 // 16) * 6

    def callback_image(self,data):
        """
        Callback for incoming image message
        """
        # Global Variables
        global display_image, publish_image, image_calibrated
        
        # Convert ROS Image Message to CV2 image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        (rows,cols,channels) = cv_image.shape
        # Convert Color Image to Grayscale
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        if image_calibrated:

            # Perform Line Detection
            edges = cv2.Canny(gray_image,20,100)
            
            # Find Contours
            contours, heirarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Remove small contours (noise) in edge detected image
            contours = [contour for contour in contours if (cv2.contourArea(contour) < self.minimumContourSize)]
            cv2.drawContours(edges, contours, -1, color = 0)
            
            # Apply Projection Transform
            edges = cv2.warpPerspective(edges, self.projection_transform, self.projection_dim) 

            # Apply Thresholding
            #ret, edges = cv2.threshold(edges, self.image_threshold, 255, cv2.THRESH_BINARY)

            # Convert to RGB to visualize lane detection points
            if display_image or publish_image:
                backtorgb = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
           
            ## Lane Detection
            height, width = edges.shape
            
            # We assume that our vehicle is placed in the center of the lane initially
            offset = 0 

            y_base = 20 # this represents 3" in front of the car


            #for i in range(height // y_increment):
            index_y = height - y_base
            index_x = (width)//2
            if index_x >= width:
                index_x = width - 1
            if index_x < 0:
                index_x = 0 
            x_left, x_right = find_offset_in_lane(edges, index_x, index_y, width)  
            
            midpoint = (x_right + x_left)//2
            offset = midpoint - width//2
            # ~~~~ FILTERING ~~~~
            # perform median filter to remove extremities
            if display_image or publish_image:

                cv2.circle(backtorgb, (x_right, index_y), 3, (0,255,255), -1)
                cv2.circle(backtorgb, (x_left, index_y), 3, (0,255,255), -1)
                cv2.circle(backtorgb, (midpoint, index_y), 3, (0,255,0),-1)
                cv2.circle(backtorgb, (width//2, index_y-5), 3, (0,0,255),-1)

                fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
                cv2.putText(backtorgb, str(offset), (25, index_y-5), fontFace, .5,(0,0,255))
        

            ## Make Steering Calculations
            angle_adjacent = 20; #experimentally determined

            ## Publish vehicle steering directions 
            self.offset_lane_pub.publish(Float32(offset))
        
            if display_image:
                cv2.imshow("Image window", backtorgb)
            if publish_image:
                try:
                    self.image_pub.publish(self.bridge.cv2_to_imgmsg(backtorgb, "bgr8"))
                except CvBridgeError as e:
                    print(e)
        else :
            if display_image:
                cv2.imshow("Calibrate Image Transform", gray_image)
            if publish_image:
                try:
                    self.image_pub.publish(self.bridge.cv2_to_imgmsg(gray_image, "bgr8"))
                except CvBridgeError as e:
                    print(e)
        if display_image:
            cv2.waitKey(3)

def shutdown_func():
    cv2.destroyAllWindows() 

image_processor_global = None

def main(args):
    rospy.on_shutdown(shutdown_func)
    global image_processor_global
    image_processor_global = image_processor()
    rospy.init_node('image_processing', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
