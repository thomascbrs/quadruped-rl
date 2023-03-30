import torch
import numpy as np
import pinocchio as pin
from time import time as clock

def ControllerRL():
    def __init__(self, filename, q_init):
        """
        Load a RL policy previously saved using torch.jit.save for cpu.
        Args:
            - filename (str) : path of the torch.jit model.
            - q_init (np.array(12)) : initial joint position
        """
        # Control gains
        self.P =  np.array([3.0, 3.0, 3.0]*4)
        self.D = np.array([0.2, 0.2, 0.2]*4)

        # Size
        self._Nobs = 235
        self._Nact = 12

        # Observation
        self.obs = np.zeros(self._Nobs)
        self.act = np.zeros(self._Nact)
        self.previous_action = np.zeros(self._Nact)


        # Load model
        model = torch.jit.load(filename)

        # Initial joint position
        self.q_init = q_init

        # timings
        self._t0 = 0
        self._t1 = 0

    def update_observation(self, joints_pos, joints_vel, imu_ori, imu_gyro):
        self._t0 = clock()

        self.obs[:] = np.hstack([np.zeros(3),
                                  imu_gyro.flatten(),
                                  imu_ori,
                                  self.vel_command,
                                  joints_pos.flatten() - self.q_init,
                                  joints_vel.flatten(),
                                  self.previous_action,
                                  np.zeros(187)])

        self._t1 = clock()
