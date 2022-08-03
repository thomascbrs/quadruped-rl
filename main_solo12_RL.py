# coding: utf8

import os
import numpy as np
from Params import RLParams
from utils import *
import time
import pinocchio as pin

#from cpuMLP import PolicyMLP3, StateEstMLP2
from numpy_mlp import MLP2
from cpuMLP import Interface

from Joystick import Joystick

params = RLParams()
np.set_printoptions(precision=3, linewidth=400)
USE_JOYSTICK = False

class RLController():
    def __init__(self, weight_path, use_state_est=False):
        """
        Args:
            params: store control parameters
        """
        # Control gains
        self.P =  np.array([3.0, 3.0, 3.0]*4) 
        self.D = np.array([0.2, 0.2, 0.2]*4)

        # Policy definition and loading parameters and scaling
        iteration = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
        weight_dir = weight_path.rsplit('/', 1)[0] + '/'

        #self.policy = PolicyMLP3()
        #self.policy.load(weight_path)
        self.policy = MLP2(132, 12, [256,128,32])
        self.policy.load_state_dict(np.load(weight_path, allow_pickle=True).item())

        self.obs_mean  = np.loadtxt(weight_dir + "/mean" + str(iteration) + ".csv",dtype=np.float32)
        self.obs_var = np.loadtxt(weight_dir + "/var" + str(iteration) + ".csv",dtype=np.float32)

        # state estimator if neededp
        #self.state_estimator = StateEstMLP2()
        #self.state_estimator.load('./checkpoints/state_estimation/state_estimator.txt')

        # non symmetric state est
        #self.state_estimator = MLP2(123, 11, [256,128])
        #self.state_estimator.load_state_dict(np.load('./tmp_checkpoints/state_estimation/state_estimator.npy', allow_pickle=True).item())

        # symmetric state est
        self.state_estimator = MLP2(123, 3, [256,128])
        self.state_estimator.load_state_dict(np.load('./tmp_checkpoints/state_estimation/symmetric_state_estimator.npy', allow_pickle=True).item())

        # History Buffers
        self.num_history_stack = 6
        self.q_pos_error_hist = np.zeros((self.num_history_stack, 12))
        self.qd_hist = np.zeros((self.num_history_stack, 12))
        self.previous_action = params.q_init.copy()
        self.preprevious_action = params.q_init.copy()

        self.pTarget12 = np.zeros((12,))

        self.vel_command = np.array([0.5,0,0.])

        self.state_est_obs = np.zeros((123,))
        self._obs = np.zeros((132,))

    def forward(self):

        self._obs[:] = np.clip((self._obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8), -10, 10)

        self.pTarget12[:] = params.q_init + 0.3 * self.policy.forward(self._obs).clip(-np.pi, np.pi)
        return self.pTarget12.copy()

    def update_observation(self, device):
        self.update_history(device) 

        self.state_est_obs[:] =  np.hstack([pin.Quaternion(device.imu.attitude_quaternion.reshape((-1,1))).toRotationMatrix()[2,:],
                                    device.joints.positions.flatten(),
                                    device.joints.velocities.flatten(),
                                    self.previous_action,
                                    self.preprevious_action,
                                    self.q_pos_error_hist[0],
                                    self.qd_hist[0],
                                    self.q_pos_error_hist[2],
                                    self.qd_hist[2],
                                    self.q_pos_error_hist[4],
                                    self.qd_hist[4]])

        self._obs[:] = np.hstack([pin.Quaternion(device.imu.attitude_quaternion.reshape((-1,1))).toRotationMatrix()[2,:],
                                  self.state_estimator.forward(self.state_est_obs)[:3],
                                  device.imu.gyroscope.flatten(),
                                  self.vel_command,
                                  device.joints.positions.flatten(),
                                  device.joints.velocities.flatten(),
                                  self.previous_action,
                                  self.preprevious_action,
                                  self.q_pos_error_hist[0],
                                  self.qd_hist[0],
                                  self.q_pos_error_hist[2],
                                  self.qd_hist[2],
                                  self.q_pos_error_hist[4],
                                  self.qd_hist[4]])

    def update_history(self, device):
        tmp = self.q_pos_error_hist.copy()
        self.q_pos_error_hist[:-1,:] = tmp[1:,:]
        self.q_pos_error_hist[-1,:] = self.pTarget12 - device.joints.positions

        tmp = self.qd_hist.copy()
        self.qd_hist[:-1,:] = tmp[1:,:]
        self.qd_hist[-1,:] = device.joints.velocities

        self.preprevious_action[:] = self.previous_action.copy()
        self.previous_action[:] = self.pTarget12.copy()

    def get_observation(self):
        return self._obs

def control_loop():
    """
    Main function that calibrates the robot, get it into a default waiting position then launch
    the main control loop once the user has pressed the Enter key
    """

    # Load RL policy
    #policy = RLController(weight_path='./tmp_checkpoints/sym_pose/policy-07-01-12-42-53/full_2000.npy', use_state_est= True)
    #p_tlicy = RLController(weight_path='./tmp_checkpoints/sym_pose/policy-07-05-23-16-12/full_2000.npy', use_state_est= True)

    #policies trained with 3vel + energy penalty
    # symmetric policy trained with 9cm foot cl
    #policy = RLController(weight_path='./tmp_checkpoints/sym_pose/energy/policy-07-28-00-10-01/full_2000.npy', use_state_est= True)

    # symmetric policy trained with 6cm foot cl
    #policy = RLController(weight_path='./tmp_checkpoints/sym_pose/energy/6cm/policy-07-29-10-59-14/full_2000.npy', use_state_est=True)
    # policy = RLController(weight_path='./tmp_checkpoints/sym_pose/energy/6cm/policy-08-03-01-20-47/full_2000.npy', use_state_est=True)

    # Run full c++ Interface
    policy = Interface()
    polDirName = "tmp_checkpoints/sym_pose/energy/6cm/policy-08-03-01-20-47/"
    estDirName = "tmp_checkpoints/state_estimation/symmetric_state_estimator.txt"
    policy.initialize(polDirName, estDirName, params.q_init.copy())

    # Define joystick
    joy = Joystick()
    joy.update_v_ref(0, 0)

    if params.LOGGING:
        from Logger import Logger
        mini_logger = Logger(logSize=int(params.max_steps))

    # INITIALIZATION ***************************************************
    device, logger, qc = initialize(params, params.q_init, np.zeros((12,)), 100000)


    # Init Histories **********************************************
    device.parse_sensor_data()
    policy.pTarget12 = params.q_init.copy()
    policy.update_observation(device.joints.positions.reshape((-1, 1)),
                              device.joints.velocities.reshape((-1, 1)),
                              device.imu.attitude_euler.reshape((-1, 1)),
                              device.imu.gyroscope.reshape((-1, 1)))

    device.joints.set_position_gains(policy.P)
    device.joints.set_velocity_gains(policy.D)
    device.joints.set_desired_positions(policy.pTarget12)
    device.joints.set_desired_velocities(np.zeros((12,)))
    device.joints.set_torques(np.zeros((12,)))
 
    for j in range(int(params.control_dt/params.dt)):
        device.send_command_and_wait_end_of_cycle(params.dt)
        device.parse_sensor_data()

    #import pudb; pudb.set_trace()
    # RL LOOP ***************************************************
    k = 0
    while (not device.is_timeout and k < params.max_steps/10):

        # Update sensor data (IMU, encoders, Motion capture)
        policy.update_observation(device.joints.positions.reshape((-1, 1)),
                                  device.joints.velocities.reshape((-1, 1)),
                                  device.imu.attitude_euler.reshape((-1, 1)),
                                  device.imu.gyroscope.reshape((-1, 1)))

        q_des = policy.forward()

        # Set desired quantities for the actuators
        device.joints.set_position_gains(policy.P)
        device.joints.set_velocity_gains(policy.D)
        device.joints.set_desired_positions(q_des)
        device.joints.set_desired_velocities(np.zeros((12,)))
        device.joints.set_torques(np.zeros((12,)))

        # Send command to the robot
        for j in range(int(params.control_dt/params.dt)):
            joy.update_v_ref(k*10 + j + 1, 0)
            device.parse_sensor_data()
            device.send_command_and_wait_end_of_cycle(params.dt)
            if params.LOGGING:
                mini_logger.sample(device, policy, q_des, None)

        # Increment counter
        k += 1
        if USE_JOYSTICK:
            vx = joy.v_ref[0,0]
            vx = 0 if abs(vx) < 0.3 else vx
            wz = joy.v_ref[-1,0]
            wz = 0 if abs(wz) < 0.3 else wz
            vy = joy.v_ref[1,0] * int(params.enable_lateral_vel)
            vy = 0 if abs(vy) < 0.3 else vy
            policy.vel_command = np.array([vx, vy, wz])
            #print(vx, wz, joy.v_ref)

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
        mini_logger.saveAll()
        print("log saved")
        mini_logger.plotAll(params, None)
    return 0


def main():
    """
    Main function
    """

    if not params.SIMULATION:  # When running on the real robot
        os.nice(-20)  #  Set the process to highest priority (from -20 highest to +20 lowest)
    control_loop()
    quit()


if __name__ == "__main__":
    main()
