# coding: utf8

import os
import numpy as np
from Params import RLParams
from utils import *
import time
import pinocchio as pin

#from cpuMLP import PolicyMLP3, StateEstMLP2
# from numpy_mlp import MLP2
# from cpuMLP import Interface
from ControllerRL import ControllerRL

from Joystick import Joystick

import torch

params = RLParams()
np.set_printoptions(precision=3, linewidth=400)

def control_loop():
    """
    Main function that calibrates the robot, get it into a default waiting position then launch
    the main control loop once the user has pressed the Enter key
    """

    # Load RL policy
    q_init = np.array([  0., 0.9, -1.64,
                0., 0.9, -1.64,

                0., 0.9, -1.64,
                0., 0.9 , -1.64 ])
    params.q_init = q_init
    policy = ControllerRL("tmp_checkpoints/policy_1.pt", q_init)

    policy.update_observation(np.zeros((12,1)),
                                  np.zeros((12,1)),
                                  np.zeros((4,1)),
                                  np.zeros((3,1)))
    policy.forward()
    #p_tlicy = RLController(weight_path='./tmp_checkpoints/sym_pose/policy-07-05-23-16-12/full_2000.npy', use_state_est= True)

    #policies trained with 3vel + energy penalty
    # symmetric policy trained with 9cm foot cl
    #policy = RLController(weight_path='./tmp_checkpoints/sym_pose/energy/policy-07-28-00-10-01/full_2000.npy', use_state_est= True)

    # symmetric policy trained with 6cm foot cl
    #policy = RLController(weight_path='./tmp_checkpoints/sym_pose/energy/6cm/policy-07-29-10-59-14/full_2000.npy', use_state_est=True)
    # policy = RLController(weight_path='./tmp_checkpoints/sym_pose/energy/6cm/policy-08-03-01-20-47/full_2000.npy', use_state_est=True)

    # Run full c++ Interface
   #policy = Interface()
    polDirName = "tmp_checkpoints/sym_pose/energy/6cm/w2/"
    #estDirName = "tmp_checkpoints/state_estimation/symmetric_state_estimator.txt"
   # policy.initialize(polDirName, estDirName, params.q_init.copy())

    # Define joystick
    if params.USE_JOYSTICK:
        joy = Joystick()
        joy.update_v_ref(0, 0)

    if params.USE_PREDEFINED:
        params.USE_JOYSTICK = False
        v_ref = 0.0  # Starting reference velocity
        alpha_v_ref = 0.03

    if params.LOGGING or params.PLOTTING:
        from Logger import Logger
        mini_logger = Logger(logSize=int(params.max_steps), SIMULATION=params.SIMULATION)

    # INITIALIZATION ***************************************************
    device, logger, qc = initialize(params, params.q_init, np.zeros((12,)), 100000)



    # Init Histories **********************************************
    # device.parse_sensor_data()

    # device.joints.set_position_gains(policy.P)
    # device.joints.set_velocity_gains(policy.D)
    # device.joints.set_desired_positions(policy.q_des)
    # device.joints.set_desired_velocities(np.zeros((12,)))
    # device.joints.set_torques(np.zeros((12,)))

    decimation = int(params.control_dt/params.dt)
 
    # for j in range(decimation):
    #     device.send_command_and_wait_end_of_cycle(params.dt)
    #     device.parse_sensor_data()

    #import pudb; pudb.set_trace()
    # RL LOOP ***************************************************
    k = 0
    while (not device.is_timeout and k < params.max_steps/decimation):

        # Update sensor data (IMU, encoders, Motion capture)
        policy.update_observation(device.joints.positions.reshape((-1, 1)),
                                  device.joints.velocities.reshape((-1, 1)),
                                  device.imu.attitude_quaternion.reshape((-1, 1)),
                                  device.imu.gyroscope.reshape((-1, 1)))

        policy.forward()

        # Set desired quantities for the actuators
        device.joints.set_position_gains(policy.P)
        device.joints.set_velocity_gains(policy.D)
        device.joints.set_desired_positions(policy.q_des)
        device.joints.set_desired_velocities(np.zeros((12,)))
        device.joints.set_torques(np.zeros((12,)))

        # Send command to the robot
        for j in range(decimation):
            if params.USE_JOYSTICK:
                joy.update_v_ref(k*10 + j + 1, 0)
            device.parse_sensor_data()
            device.send_command_and_wait_end_of_cycle(params.dt)
            if params.LOGGING or params.PLOTTING:
                mini_logger.sample(device, policy, policy.q_des, policy.get_observation(),
                                   policy.get_computation_time(), qc)

        # Increment counter
        k += 1
        if params.USE_JOYSTICK:
            joy.update_v_ref(k, 0)
            vx = joy.v_ref[0,0]
            vx = 0 if abs(vx) < 0.3 else vx
            wz = joy.v_ref[-1,0]
            wz = 0 if abs(wz) < 0.3 else wz
            vy = joy.v_ref[1,0] * int(params.enable_lateral_vel)
            vy = 0 if abs(vy) < 0.3 else vy
            policy.vel_command = np.array([vx, vy, wz])
            # print(vx, wz, joy.v_ref)
        elif params.USE_PREDEFINED:
            t_rise = 100  # rising time to max vel
            t_duration = 500  # in number of iterations
            if k < t_rise + t_duration:
                v_max = 1.0  # in m/s
                v_gp = np.min([v_max * (k / t_rise), v_max])
            else:
                alpha_v_ref = 0.1
                v_gp = 0.0  # Stop the robot
            v_ref = alpha_v_ref * v_gp + (1 - alpha_v_ref) * v_ref  # Low-pass filter
            policy.vel_command = np.array([v_ref, 0, 0])  
        elif k > 0 and k % 300 ==0:
            vx = np.random.uniform(-0.5 , 1.5)
            vx = 0 if abs(vx) < 0.3 else vx
            wz = np.random.uniform(-1, 1)
            wz = 0 if abs(wz) < 0.3 else wz
            policy.vel_command = np.array([vx, 0, wz])

        if params.record_video and k % 10==0:
            save_frame_video(int(k//10), './recordings/')

    if device.is_timeout:
        print("Time out detected..............")

    # DAMPING TO GET ON THE GROUND PROGRESSIVELY *********************
    damping(device, params)

    # FINAL SHUTDOWN *************************************************
    shutdown(device, params)

    if params.LOGGING:
        mini_logger.saveAll(suffix = "_" + (polDirName.split("/")[-2]).replace(".", "-"))
        print("log saved")
    if params.LOGGING or params.PLOTTING:
        mini_logger.plotAll(params.dt, None)
    return 0



def main():
    """
    Main function
    """

    if True or  not params.SIMULATION:  # When running on the real robot
        os.nice(-20)  #  Set the process to highest priority (from -20 highest to +20 lowest)

    # import cProfile
    # import time

    # profiler = cProfile.Profile()
    # profiler.run("control_loop()")
    # profiler.print_stats("calls")
    control_loop()
    quit()


if __name__ == "__main__":
    main()
