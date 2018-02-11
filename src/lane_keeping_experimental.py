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
from barc.msg import ECU, Encoder
from std_msgs.msg import Int32, Float32
from math import pi,sin,radians
import time
import serial
from numpy import zeros, hstack, cos, array, dot, arctan
from input_map import angle_2_servo, servo_2_angle

def offset_callback(data):
    # Store current and previous Lane Pixel Offset
    global pixel_offset
    if data is not None:
        pixel_offset = data.data


# Speed Estimator Global Variables
n_FL, n_FR, n_FL_prev, n_FR_prev = zeros(4)
t0 = time.time()
v_x_curr = 0.0
v_x_prev = 0.0
d_f = 0.0 
r_tire = 0.0319
dx_magnets = 2 * pi * r_tire / 4.0

def enc_callback(data):
    global v_x_curr, v_x_prev
    global d_f, t0, dx_magnets
    global n_FL, n_FR, n_FL_prev, n_FR_prev
    if data is not None:
        n_FL = data.FL
        n_FR = data.FR

    # compute time elapsed
    tf = time.time()
    dt = tf - t0

    # if enough time elapse has elapsed, estimate v_x
    dt_min = 0.20
    if dt >= dt_min:
        # compute speed :  speed = distance / time
        v_FL = float(n_FL- n_FL_prev)*dx_magnets/dt
        v_FR = float(n_FR- n_FR_prev)*dx_magnets/dt

        # update encoder v_x, v_y measurements
        # only valid for small slip angles, still valid for drift?
        v_x_curr 	= (v_FL + v_FR)/2.0*cos(radians(d_f))
        # Low Pass Filter
        v_x_curr    = 0.75 * v_x_curr + 0.25 * v_x_prev
        v_x_prev    = v_x_curr
        # update old data
        n_FL_prev   = n_FL
        n_FR_prev   = n_FR
        t0 	        = time.time()

# Global Steering Control Variables
e_prev = 0
pixel_offset = 0.0
ei = 0.0

def steering_command(rateHz, Kp, Kd, Ki):
    global pixel_offset
    global e_prev
    global ei, d_f
    e = pixel_offset
    ed = e - e_prev
    ei = ei + (e / float(rateHz))
    # Calulate Turn Angle
    turn = (e * Kp) + (ed * Kd) + (ei * Ki)
    
    # Saturate Turn Value
    if (turn < -30.0):
        turn = -30.0
    elif (turn > 30.0):
        turn = 30.0
    d_f = turn
    e_prev = e
    return angle_2_servo(-turn), -turn


# Global Speed Control Variables
max_torque = 0
eSprev = 0.0
eiS = 0.0

def speed_command(speed_desired, rateHz, KpS, KdS, KiS):
    global eiS, v_x_curr, max_torque

    #Calculate Error
    e = speed_desired - v_x_curr
    ed = (e - eSprev) * rateHz
    eiS = eiS + (e / float(rateHz))

    #Calculate Torque Value
    torque = 94 + ((e * KpS) + (ed * KdS) + (ei * KiS))
    #Saturate Torque Value
    if (torque > max_torque):
        torque = max_torque
    elif (torque <  90):
        torque = 90
    return torque
    #return torque


nh = None
#############################################################
def main_auto():
    global nh, v_x_curr, max_torque
    # initialize ROS node
    rospy.init_node('auto_mode', anonymous=True)
    nh = rospy.Publisher('ecu', ECU, queue_size = 10)
     
    steering_offset_subscriber = rospy.Subscriber("lane_offset", Float32, offset_callback)
    rospy.Subscriber('encoder', Encoder, enc_callback)
    
    #XXX:DATALOGGING
    torque_pub = rospy.Publisher('torque', Float32, queue_size = 10)
    steering_angle_pub = rospy.Publisher('steering_angle', Float32, queue_size = 10)
    speed_pub = rospy.Publisher('speed', Float32, queue_size = 10)

    # set node rate
    rateHz  = 50
    rate 	= rospy.Rate(rateHz)

    # Get Desired Speed
    speed_desired = rospy.get_param("speed_controller/speed_desired")
    max_torque = rospy.get_param("speed_controller/max_torque")

    # Get Speed Controller Parameters
    KpS = rospy.get_param("speed_controller/KpS")
    KdS = rospy.get_param("speed_controller/KdS")
    KiS = rospy.get_param('speed_controller/KiS')
    
    # Get Turning Controller Parameters
    Kp 	= rospy.get_param("speed_controller/Kp")
    Kd 	= rospy.get_param("speed_controller/Kd")
    Ki 	= rospy.get_param("speed_controller/Ki")

    # main loop
    while not rospy.is_shutdown():
        # get command signal
        speedCMD = speed_command(speed_desired, rateHz, KpS, KdS, KiS)
        servoCMD, angle = steering_command(rateHz, Kp, Kd, Ki)
        
        # send command signal 
        ecu_cmd = ECU(speedCMD, servoCMD)
        nh.publish(ecu_cmd)

        #XXX:DATALOGGING
        torque_pub.publish(speedCMD)
        steering_angle_pub.publish(angle)
        speed_pub.publish(v_x_curr)
        
        # wait
        rate.sleep()

#############################################################
if __name__ == '__main__':	
    try:
        main_auto()
    except rospy.ROSInterruptException:
        pass
