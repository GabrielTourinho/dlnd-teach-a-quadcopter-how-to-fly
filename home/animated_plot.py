from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# Make sure to change from notebook to inline after your tests
import time

class AnimatedPlot():
    def __init__(self):
        """Initialize parameters"""
        self.X, self.Y, self.Z = [], [], []

        self.fig = plt.figure(figsize = (14,8))
        self.ax = self.fig.add_subplot(111, projection='3d')

    def plot(self, task, i_episode=None):
        pose = task.sim.pose[:3]
        self.X.append(pose[0])
        self.Y.append(pose[1])
        self.Z.append(pose[2])
        self.ax.clear()
        if i_episode:
            plt.title("Episode {}".format(i_episode))

        if len(self.X)>1:
            self.ax.scatter(self.X[:-1], self.Y[:-1], self.Z[:-1], c='k', alpha=0.3)
        if task.sim.done and task.sim.runtime > task.sim.time:
            # Colision
            self.ax.scatter(pose[0], pose[1], pose[2], c='r', marker='*', linewidths=5)
        else:
            self.ax.scatter(pose[0], pose[1], pose[2], c='k', marker='s', linewidths=5)

        self.fig.canvas.draw()
        plt.pause(0.5)