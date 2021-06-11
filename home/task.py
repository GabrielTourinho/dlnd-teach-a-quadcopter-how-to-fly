import numpy as np
import pandas as pd
import seaborn as sns
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #reward = np.tanh(1.-.3*(abs(self.sim.pose[:3] - self.target_pos))).sum()
        
        target = self.target_pos[2]
        z = self.sim.pose[2]
        
        takeoff = z >= target
        
        crash = z <= 0
        
        #crash = self.sim.done and self.sim.runtime > self.sim.time
        
        distance = abs(z - target)
        
        if not crash:
            if takeoff: reward = 10   
            else: reward = np.tanh(1.-.03*distance) #1 - distance**0.4
        else: reward = -10
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
    
    def map_function(reward_function, x ,y):
        R = pd.DataFrame(np.zeros([len(x), len(y)]), index=y, columns=x)
        for xx in x:
            for yy in y:
                R[xx][yy] = reward_function(xx, yy)

        return R