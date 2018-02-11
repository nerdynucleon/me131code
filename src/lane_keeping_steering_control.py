#!/usr/bin/env python

# ---------------------------------------------------------------------------
# Licensing Information: You are free to use or extend these projects for 
# education or reserach purposes provided that (1) you retain this notice
# and (2) you provide clear attribution to UC Berkeley, including a link 
# to http://barc-project.com
#
# Attibution Information: The barc project ROS code-base was developed
# at UC Berkeley in the Model Predictive Control (MPC) lab by Jon Gonzales
# (jon.gonzales@berkeley.edu). The cloud services integation with ROS was developed
# by Kiet Lam  (kiet.lam@berkeley.edu). The web-server app Dator was 
# based on an open source project by Bruce Wootton
# ---------------------------------------------------------------------------
from __future__ import division
import rospy
from data_service.msg import TimeData
from barc.msg import ECU
from std_msgs.msg import Int32, Float32
from math import pi,sin,radians
import time
import serial
from numpy import zeros, hstack, cos, array, dot, arctan
from input_map import angle_2_servo, servo_2_angle

# Steering Offset Global Variables
pixel_offset = 0.0
pixel_offset_prev = 0.0
ei = 0.0

def offset_callback(data):
    # Store current and previous Lane Pixel Offset
    global pixel_offset
    global pixel_offset_prev
    if data is not None:
        pixel_offset_prev = pixel_offset
        pixel_offset = float(data.data)
    return 0

def steering_command(rateHz, Kp, Kd, Ki):
    global pixel_offset
    global pixel_offset_prev
    global ei
    # Current Error, Derivative of Error, Intergral of Error
    e = pixel_offset
    ed = e - pixel_offset_prev
    ei = ei + (e / float(rateHz))
    # Calulate Turn Angle
    turn = (e * Kp) + (ed * Kd) + (ei * Ki)
    # Saturate Turn Value
    if (turn < -30.0):
        turn = -30.0
    elif (turn > 30.0):
        turn = 30.0
    return angle_2_servo(-turn), -turn

#############################################################
def main_auto():
    # initialize ROS node
    rospy.init_node('auto_mode', anonymous=True)
    nh = rospy.Publisher('ecu', ECU, queue_size = 10)
    steering_offset_subscriber = rospy.Subscriber("lane_offset", Float32, offset_callback) 
	
    # set node rate
    rateHz  = 50
    rate 	= rospy.Rate(rateHz)

    # specify test and test options
    v_x_pwm 	= rospy.get_param("steering_controller/v_x_pwm")
    
    # Get Controller Parameters
    Kp 	= rospy.get_param("steering_controller/Kp")
    Kd 	= rospy.get_param("steering_controller/Kd")
    Ki 	= rospy.get_param("steering_controller/Ki")

    # main loop
    while not rospy.is_shutdown():
        # get command signal
        servoCMD, steering_angle = steering_command(rateHz, Kp, Kd, Ki)
        # send command signal 
        ecu_cmd = ECU(v_x_pwm, servoCMD)
        nh.publish(ecu_cmd)
        
        # wait
        rate.sleep()

#############################################################
if __name__ == '__main__':
	try:
		main_auto()
	except rospy.ROSInterruptException:
		pass
