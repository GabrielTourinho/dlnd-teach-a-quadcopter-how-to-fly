import numpy as np
import pandas as pd
import seaborn as sns

class RewardPlot():
    """Reward plotting class"""
    
    def get_reward(x, z, target=20):
        """Uses current pose of sim to return reward."""
        #reward = np.tanh(1.-.3*(abs(self.sim.pose[:3] - self.target_pos))).sum()

        takeoff = z >= target

        crash = z<=0

        distance = abs(z - target)

        if not crash:
            if takeoff: reward = 10
            else: 
                reward = np.tanh(1-.03*distance)
        else: reward = -10

        return reward


    def map_function(reward_function, x, y):
        R = pd.DataFrame(np.zeros([len(x), len(y)]), index=y, columns=x)
        for xx in x:
            for yy in y:
                R[xx][yy] = reward_function(xx, yy)

        return R