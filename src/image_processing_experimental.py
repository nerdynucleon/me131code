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
import kernel

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

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/image_raw", Image,self.callback_image)
        
        #Get Launch File Parameters and configure node
        calibrate_transform = rospy.get_param("/image_processing/calibrate_transform")
        display_image = rospy.get_param("/image_processing/display_image")
        publish_image = rospy.get_param("/image_processing/publish_image")
        global image_calibrated
        
        if publish_image:
            self.image_pub = rospy.Publisher("cv_image", Image, queue_size = 10)
        
        # Projection Transform Parameters
        self.x_offset = 50
        self.x_width = 75
        self.y_height = 50
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

        self.black_lane = rospy.get_param("/image_processing/black_lane")
        self.kernel = kernel.kernel_get(5,1,11,5,self.black_lane)
        self.quartile = 95.0
        self.prev_offset = 0

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
        if display_image:
            cv2.imshow("Raw Image", gray_image)
            cv2.waitKey(3) 
        
        if image_calibrated:
            #IPM
            ipm = cv2.warpPerspective(gray_image, self.projection_transform, self.projection_dim) 
            if display_image:
                cv2.imshow("IPM", ipm)
                cv2.waitKey(3) 
            # filter with kernel
            filtered = signal.fftconvolve(self.kernel,ipm) 
            threshold  = np.percentile(filtered, self.quartile)
            idx = filtered < threshold 
            filtered[idx] = 0
            idx = filtered >= threshold
            filtered[idx] = 255
            filtered = np.array(filtered, dtype=np.uint8)
            kernel = np.ones((1,2))
            filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
            # Convert to RGB to visualize lane detection points
            if display_image:
                cv2.imshow("End Product", filtered)
                cv2.waitKey(3) 
            if display_image or publish_image:
                backtorgb = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
           
            ## Lane Detection
            height, width = filtered.shape
            
            # Change nominal reference point based on previous reference point
            index_x = width//2 + self.prev_offset 
            index_y = 10
            center_lane = []
            yvalid = []
            while index_y < height:
                x_left, x_right = find_offset_in_lane(filtered, index_x, height - index_y, width)  
                # Nonvalid signal
                if x_left == 0 or x_right == width - 1:
                    index_y += 1
                    continue;
                centerlanex = ( x_left + x_right ) // 2
                center_lane += [centerlanex]
                yvalid += [index_y]
                index_y += 1
                index_x += centerlanex - index_x
            
            offset_lane = offset - width//2
            self.prev_offset = offset_lane

            if display_image or publish_image:
                cv2.circle(backtorgb, (x_left, height - index_y), 3, (0,255,255), -1)
                cv2.circle(backtorgb, (x_right, height - index_y), 3, (0,255,255), -1)
                cv2.circle(backtorgb, (offset, height - index_y), 3, (255,0,255), -1)

            self.offset_lane_pub.publish(Float32(offset_lane))
            
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
