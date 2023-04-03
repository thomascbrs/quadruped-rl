import torch
import numpy as np
import pinocchio as pin
from time import time as clock

class ControllerRL():
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
        self.vel_command = np.array([0.,0.,0.])

        # Scales 
        self.scale_actions = 0.3
        self.scale_cmd_lin_vel = 2.0
        self.scale_cmd_ang_vel = 0.25
        self.scale_cmd = np.array([self.scale_cmd_lin_vel,self.scale_cmd_lin_vel,self.scale_cmd_ang_vel])
        self.scale_ang_vel = 0.25
        self.scale_dof_pos = 1.0
        self.scale_dof_vel = 0.05
        self.clip_observations = 100.
        self.clip_actions = 100.

        # Load model
        self.model = torch.jit.load(filename)
        self.model.eval()

        # Initial joint position
        self.q_init = q_init

        # timings
        self._t0 = 0
        self._t1 = 0
        self._t2 = 0
        self._t3 = 0
        
    def forward(self):
        
        self._t2 = clock()

        obs = torch.from_numpy(self.obs).float().unsqueeze(0)            
        self.act = self.model(obs).detach().numpy().squeeze()       
        self.act = np.clip(self.act, -self.clip_actions, self.clip_actions)
        
        self._t3 = clock()

        self.q_des = self.act * self.scale_actions + self.q_init
        return self.q_des# self.q_init
    
    def update_observation(self, joints_pos, joints_vel, imu_ori, imu_gyro):
        self._t0 = clock()

        self.joints_pos = joints_pos
        self.joints_vel = joints_vel

        self.projected_gravity = imu_ori.flatten()
        self.base_ang_vel = imu_gyro.flatten()
        
        self.obs = np.hstack([np.zeros(3),
                                  self.base_ang_vel * self.scale_ang_vel,
                                  self.projected_gravity,
                                  self.vel_command * self.scale_cmd,
                                  (joints_pos.flatten() - self.q_init) * self.scale_dof_pos,
                                  joints_vel.flatten() * self.scale_dof_vel,
                                  self.act,
                                  -1.5 * np.ones(187)])
        self.obs = np.clip(self.obs, -self.clip_observations, self.clip_observations)

        self._t1 = clock()

    def get_observation(self):
        return self.obs
    
      
    def get_computation_time(self):
        # Computation time in us
        return (self._t3 - self._t2) * 1e6