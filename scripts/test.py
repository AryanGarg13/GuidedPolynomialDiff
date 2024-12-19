import sys
import os
import numpy as np
import pybullet as p
import pybullet_data
import time
# Add the IROS24 directory (parent of lib) to the system path
sys.path.append(os.path.abspath('/home/aadi_iiith/Desktop/RRC/IROS24/IROS24'))

# Now you should be able to import from lib
from lib.models import Diffusion, BernsteinTemporalUNet
from lib.priors import MPiNetPrior as Prior
from lib.envs import RobotEnvironment
from lib.guides.franka_volume_guide import VolumeGuide
from lib.metrics import MetricsCalculator

class PyBulletEnv:

    def __init__(self):
        self.points = []

    def spawn_points(self, points, color="green"):
        """Spawn points in the PyBullet environment."""
        for point in points:
            
            x = point[0]
            y = point[1]
            z = point[2]
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.02,
                rgbaColor=[0, 1, 0, 1] if color == "green" else [1, 0, 0, 1]
            )
            point_id = p.createMultiBody(baseVisualShapeIndex=visual_shape_id, basePosition=[x, y, z])
            self.points.append(point_id)

    def remove_all_points(self):
        """Remove all points from the PyBullet environment."""
        for point_id in self.points:
            p.removeBody(point_id)
        self.points = []

# Initialize PyBullet environment
env = PyBulletEnv()
obstacle_config = []

# env2 = RobotEnvironment(gui, manipulator)

# print("Imports successful")

# trajectory_path="/home/aadi_iiith/Desktop/RRC/IROS24/unguided_trajs.npy"
# trajectories= np.load(trajectory_path)
# print(trajectories.shape)
# guide = VolumeGuide(env = env,
#                     obstacle_config=obstacle_config,
#                     clearance = 0.,
#                     guide_type = "intersection",
#                     device = "cuda")
# scene_trajs = trajectories[0]
# trajectory = guide.cost(scene_trajs,0,0)
p.connect(p.GUI)

# Set up the environment
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Add path to PyBullet's data files
planeId = p.loadURDF("plane.urdf")  # Load a simple plane
p.setGravity(0, 0, -9.8)  # Set gravity
p.setTimeStep(1/240)  # Set simulation time step

for num in range(100,512):
    path = "/home/aadi_iiith/Desktop/Trajectories/"+str(num)
    trajectory = np.loadtxt(path)
    trajectory = trajectory.T
    env.remove_all_points()
    # env.spawn_points(coeffs, "red")
    env.spawn_points(trajectory)
    # env.spawn_points(traj, "blue")
    time.sleep(10)
# # Define a basic trajectory (e.g., a circular path)
# radius = 0.5
# num_points = 50
# trajectory = []

# # Generate points on a circle in the XY plane
# for i in range(num_points):
#     angle = 2 * np.pi * i / num_points
#     x = radius * np.cos(angle)
#     y = radius * np.sin(angle)
#     z = 0.1  # Slightly above the ground
#     trajectory.append([x, y, z])

# # Visualize the trajectory in PyBullet
# point_ids = []
# for point in trajectory:
#     visual_shape_id = p.createVisualShape(
#         shapeType=p.GEOM_SPHERE,
#         radius=0.02,
#         rgbaColor=[0, 0, 1, 1]  # Blue color
#     )
#     point_id = p.createMultiBody(baseVisualShapeIndex=visual_shape_id, basePosition=point)
#     point_ids.append(point_id)

# # Optional: Run the simulation for some time to visualize
# for _ in range(10000):
#     p.stepSimulation()
#     time.sleep(1/240)

# # Disconnect from PyBullet
# p.disconnect()
# file = "/home/aadi_iiith/Desktop/RRC/IROS24/IROS24/scripts/errors.txt"
# with open(file,'r') as fp:
#     values = [float(line.strip()) for line in fp]

# # Plot the values
# x = list(range(3,3+len(values)))
# plt.plot(x,values, marker='o')
# plt.title('MSE vs Number of Control Points')
# plt.xlabel('Control Points')
# plt.ylabel('MSE')
# plt.grid(True)
# plt.show()