# coding: utf8

import os
import numpy as np
from Params import RLParams
from utils import *

#from cpuMLP import PolicyMLP3, StateEstMLP2
# from numpy_mlp import MLP2
# from cpuMLP import Interface
from ControllerRL import ControllerRL

from Joystick import Joystick

PROFILER = False

np.set_printoptions(precision=3, linewidth=400)

class SoloRLDevice:

    def __init__(self, policy: ControllerRL, params: RLParams, run_name: str):
        self.policy = policy
        self.params = params
        self.run_name = run_name
        self._pre_init()
       
    def _pre_init(self):
        if not self.params.SIMULATION:
            # this must be called once before connecting to the robot
            # because the first call to PyTorch takes too much time
            # (causes desync with the robot)
            self.policy.update_observation(np.zeros((12,1)),
                                np.zeros((12,1)),
                                np.zeros((4,1)),
                                np.zeros((3,1)))
            self.policy.forward()

        params = self.params
        # Define joystick
        if params.USE_JOYSTICK:
            self.joystick = Joystick()
            self.joystick.update_v_ref()

        if params.USE_PREDEFINED:
            params.USE_JOYSTICK = False
            self.v_ref = 0.0  # Starting reference velocity
            self.alpha_v_ref = 0.03

        self.decimation = int(params.control_dt/params.dt)
        self.k = 0

    def init_robot_and_wait_floor(self):
         self.device, self.logger, _qc = initialize(self.params, self.params.q_init, np.zeros((12,)), 100000)

    def damping_and_shutdown(self):     
        device = self.device

        # DAMPING TO GET ON THE GROUND PROGRESSIVELY *********************
        damping(device, self.params)

        # FINAL SHUTDOWN *************************************************
        shutdown(device, self.params)

    def _parse_joystick_cmd(self):
        joystick = self.joystick
        joystick.update_v_ref()
        vx = joystick.v_ref[0,0]
        vx = 0 if abs(vx) < 0.3 else vx
        wz = joystick.v_ref[-1,0]
        wz = 0 if abs(wz) < 0.3 else wz
        vy = joystick.v_ref[1,0] * int(self.params.enable_lateral_vel)
        vy = 0 if abs(vy) < 0.3 else vy
        return np.array([vx, vy, wz])
    
    def _predefined_vel_cmd(self):
        t_rise = 100  # rising time to max vel
        t_duration = 500  # in number of iterations
        if self.k < t_rise + t_duration:
            v_max = 1.0  # in m/s
            v_gp = np.min([v_max * (self.k / t_rise), v_max])
        else:
            self.alpha_v_ref = 0.1
            v_gp = 0.0  # Stop the robot
        self.v_ref = self.alpha_v_ref * v_gp + (1 - self.alpha_v_ref) * self.v_ref  # Low-pass filter
        return np.array([self.v_ref, 0, 0]) 
    
    def _random_cmd(self):
        vx = np.random.uniform(-0.5 , 1.5)
        vx = 0 if abs(vx) < 0.3 else vx
        wz = np.random.uniform(-1, 1)
        wz = 0 if abs(wz) < 0.3 else wz
        return np.array([vx, 0, wz])

    def update_vel_command(self):
        params = self.params

        if params.USE_JOYSTICK:
            self.policy.vel_command = self._parse_joystick_cmd()
        
        elif params.USE_PREDEFINED:
            self.policy.vel_command = self._predefined_vel_cmd()
        
        elif self.k > 0 and self.k % 300 ==0:
            self.policy.vel_command = self._random_cmd()

    def run_cycle(self):
        policy = self.policy
        device = self.device
        decimation = self.decimation
        params = self.params

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
                self.joystick.update_v_ref()

            device.parse_sensor_data()
            device.send_command_and_wait_end_of_cycle(params.dt)

            if params.LOGGING or params.PLOTTING:
                self.logger.sample(policy, policy.q_des, policy.vel_command,
                                    policy.get_observation(), policy.get_computation_time())
                
        self.post_cycle_callback()
       
    def post_cycle_callback(self):
        self.k += 1
        if self.params.record_video and self.k % 10==0:
            save_frame_video(int(self.k//10), './recordings/')

    def save_plot_logs(self):
        params = self.params
        if params.LOGGING:
            self.logger.saveAll(suffix = "_" + self.run_name)
            print("log saved")

        if params.PLOTTING:
            self.logger.plotAll(params.dt, None)

    def control_loop(self):
        self.init_robot_and_wait_floor()
        while (not self.device.is_timeout and self.k < self.params.max_steps/self.decimation):
            self.run_cycle()
            self.update_vel_command()     
        
        if self.device.is_timeout:
            print("Time out detected..............")

        self.damping_and_shutdown()

def main():
    """
    Main function
    """

    params = RLParams()
    if not params.SIMULATION:  # When running on the real robot
        os.nice(-20)  #  Set the process to highest priority (from -20 highest to +20 lowest)

    q_init = np.array([  0., 0.9, -1.64,
                0., 0.9, -1.64,

                0., 0.9, -1.64,
                0., 0.9 , -1.64 ])
    params.q_init = q_init
    policy = ControllerRL("tmp_checkpoints/policy_1.pt", q_init)
    
    device = SoloRLDevice(policy, params, "solo")
    device.control_loop()
    device.save_plot_logs()

    quit()

if __name__ == "__main__":
    if PROFILER:
        import cProfile

        profiler = cProfile.Profile()
        profiler.enable()

    main()

    if PROFILER:
        profiler.disable()
        profiler.print_stats("calls")
