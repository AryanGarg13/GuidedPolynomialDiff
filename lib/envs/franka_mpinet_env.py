import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data
import pybullet_planning as pp
import numpy as np
import time
import re
import os
import h5py
import lib.envs.planners as planners
import cv2


class RobotEnvironment:

    def __init__(self, gui = True, timestep = 1/480, manipulator = True, benchmarking = False):

        self.n = 50
        self.c = 7

        self.points_list = []
        
        self.client_id = bc.BulletClient(p.GUI if gui else p.DIRECT)            # Initialize the bullet client
        self.client_id.setAdditionalSearchPath(pybullet_data.getDataPath())     # Add pybullet's data package to path
        self.client_id.setTimeStep(timestep)                                    # Set simulation timestep
        self.client_id.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)        # Disable Shadows
        self.client_id.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)            # Disable Frame Axes

        self.timestep = timestep
        self.benchmarking = benchmarking
        self.num_collisions = 0
        p.setAdditionalSearchPath(pybullet_data.getDataPath())      # Add pybullet's data package to path   

        self.colors = {'black': np.array([0., 0., 0.]) / 255.0,
                       'blue': np.array([20, 30, 255]) / 255.0,  # blue
                       'green': np.array([0, 255, 0]) / 255.0,  # green
                       'dark_blue': np.array([78, 121, 167]) / 255.0,
                       'dark_green': np.array([89, 161, 79]) / 255.0,
                       'brown': np.array([156, 117, 95]) / 255.0,  # brown
                       'orange': np.array([242, 142, 43]) / 255.0,  # orange
                       'yellow': np.array([237, 201, 72]) / 255.0,  # yellow
                       'gray': np.array([186, 176, 172]) / 255.0,  # gray
                       'red': np.array([255, 0, 0]) / 255.0,  # red
                       'purple': np.array([176, 122, 161]) / 255.0,  # purple
                       'cyan': np.array([118, 183, 178]) / 255.0,  # cyan
                       'light_cyan': np.array([0, 255, 255]) / 255.0,
                       'light_pink': np.array([255, 105, 180]) / 255.0,
                       'pink': np.array([255, 157, 167]) / 255.0}  # pink
                       
        
        target = self.client_id.getDebugVisualizerCamera()[11]          # Get cartesian coordinates of the camera's focus
        self.client_id.resetDebugVisualizerCamera(                      # Reset initial camera position
            cameraDistance=1.5,
            cameraYaw=90,
            cameraPitch=-25,
            cameraTargetPosition=target,
        )
        
        p.resetSimulation()                             
        self.client_id.setGravity(0, 0, -9.8)           # Set Gravity

        self.plane = self.client_id.loadURDF("plane.urdf", basePosition=(0, 0, 0), useFixedBase=True)   # Load a floor
        
        self.client_id.changeDynamics(                  # Set physical properties of the floor
            self.plane,
            -1,
            lateralFriction=1.1,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )

        if manipulator:
            self.initialize_manipulator()

        self.obs_ids = []

    def initialize_manipulator(self, urdf_file = "franka_panda/panda.urdf", base_position = (0, 0, 0)):

        self.manipulator = self.client_id.loadURDF(urdf_file, basePosition = base_position, useFixedBase = True)
        self.joints = []

        for i in range(self.client_id.getNumJoints(self.manipulator)):

            info = self.client_id.getJointInfo(self.manipulator, i)

            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]

            if joint_name == "panda_grasptarget_hand":
                self.end_effector = joint_id

            if joint_type == self.client_id.JOINT_REVOLUTE:
                self.joints.append(joint_id)

        self.joint_lower_limits = np.array([-166*(np.pi/180), 
                                           -101*(np.pi/180), 
                                           -166*(np.pi/180), 
                                           -176*(np.pi/180),
                                           -166*(np.pi/180),
                                           -1*(np.pi/180),
                                           -166*(np.pi/180)])
        
        self.joint_upper_limits = np.array([166*(np.pi/180), 
                                           101*(np.pi/180), 
                                           166*(np.pi/180), 
                                           -4*(np.pi/180),
                                           166*(np.pi/180),
                                           215*(np.pi/180),
                                           166*(np.pi/180)])
        
        links_folder_path = 'assets/franka_panda/meshes/collision/'
        try:
            link_file_names = os.listdir(links_folder_path)
        except OSError as e:
            print(f"Error reading files in folder: {e}")
       
        self.link_meshes = {}
        self.link_dimensions = {}
        self.link_centers = {}

        self.link_index_to_name = ['link0', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7',
                                   'hand', 'finger', 'finger', 'finger', 'finger']

        for file_name in link_file_names:

            if file_name[-4:] == ".obj":
                vertices = []
                link_name = file_name[:-4]
                with open(links_folder_path + file_name, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('v '):
                            vertex = re.split(r'\s+', line)[1:4]
                            vertex = np.array([float(coord) for coord in vertex])
                            vertices.append(vertex)
                self.link_meshes[link_name] = np.array(vertices)
                max_point = np.max(self.link_meshes[link_name], axis = 0)
                min_point = np.min(self.link_meshes[link_name], axis = 0)
                self.link_dimensions[link_name] = max_point - min_point
                self.link_centers[link_name] = self.link_dimensions[link_name]/2 + min_point
    
    def get_mpinet_scene(self, index, data = 'val'):

        with h5py.File("assets/dataset/mpinet/" + data + ".hdf5", "r") as f:

            cuboid_centers = f['cuboid_centers'][index]
            cuboid_dims = f['cuboid_dims'][index]
            cuboid_quaternions = np.roll(f['cuboid_quaternions'][index], -1, axis = 1)

            cylinder_centers = f['cylinder_centers'][index]
            cylinder_heights = f['cylinder_heights'][index]
            cylinder_quaternions = np.roll(f['cylinder_quaternions'][index], -1, axis = 1)
            cylinder_radii = f['cylinder_radii'][index]

            num_cuboids = np.argmax(np.any(cuboid_dims == 0, axis = 1))
            num_cylinders = np.argmax(np.any(cylinder_heights == 0, axis = 1))

            cuboid_centers = cuboid_centers[:num_cuboids]
            cuboid_dims = cuboid_dims[:num_cuboids]
            cuboid_quaternions = cuboid_quaternions[:num_cuboids]

            cuboid_config = np.concatenate([cuboid_centers, cuboid_quaternions, cuboid_dims], axis = 1)
            
            cylinder_centers = cylinder_centers[:num_cylinders]
            cylinder_heights = cylinder_heights[:num_cylinders]
            cylinder_quaternions = cylinder_quaternions[:num_cylinders]
            cylinder_radii = cylinder_radii[:num_cylinders]

            cylinder_config = np.concatenate([cylinder_centers, cylinder_quaternions, cylinder_radii, cylinder_heights], axis = 1)

            cylinder_cuboid_dims = np.zeros((num_cylinders, 3))
            cylinder_cuboid_dims[:, 0] = 2 * cylinder_radii[:, 0]
            cylinder_cuboid_dims[:, 1] = 2 * cylinder_radii[:, 0]
            cylinder_cuboid_dims[:, 2] = cylinder_heights[:, 0]

            obstacle_centers = np.concatenate([cuboid_centers, cylinder_centers], axis = 0)
            obstacle_dims = np.concatenate([cuboid_dims, cylinder_cuboid_dims], axis = 0)
            obstacle_quaternions = np.concatenate([cuboid_quaternions, cylinder_quaternions], axis = 0)

            obstacle_config = np.concatenate([obstacle_centers, obstacle_quaternions, obstacle_dims], axis = 1)

            start = f['hybrid_solutions'][index][0, :]
            goal = f['hybrid_solutions'][index][-1, :]

            return obstacle_config, cuboid_config, cylinder_config, start, goal
    
    def spawn_cuboids(self, cuboid_config, color):

        for i in range(cuboid_config.shape[0]):
            
            vuid = self.client_id.createVisualShape(p.GEOM_BOX, 
                                        halfExtents = cuboid_config[i, 7:]/2,
                                        rgbaColor = np.hstack([self.colors[color], np.array([1.0])]))

            obs_id = self.client_id.createMultiBody(baseVisualShapeIndex = vuid, 
                                                basePosition = cuboid_config[i, :3], 
                                                baseOrientation = cuboid_config[i, 3:7])
            
            self.obs_ids.append(obs_id)

    def spawn_cylinders(self, cylinder_config, color):

        for i in range(cylinder_config.shape[0]):
            
            vuid = self.client_id.createVisualShape(p.GEOM_CYLINDER, 
                                        radius = cylinder_config[i, 7],
                                        length = cylinder_config[i, 8],
                                        rgbaColor = np.hstack([self.colors[color], np.array([1.0])]))

            obs_id = self.client_id.createMultiBody(baseVisualShapeIndex = vuid, 
                                                basePosition = cylinder_config[i, :3], 
                                                baseOrientation = cylinder_config[i, 3:7])
            
            self.obs_ids.append(obs_id)

    def spawn_spheres(self, sphere_config, color):

        for i in range(sphere_config.shape[0]):
            
            vuid = self.client_id.createVisualShape(p.GEOM_SPHERE, 
                                        radius = sphere_config[i, 7],
                                        rgbaColor = np.hstack([self.colors[color], np.array([1.0])]))

            obs_id = self.client_id.createMultiBody(baseVisualShapeIndex = vuid, 
                                                basePosition = sphere_config[i, :3], 
                                                baseOrientation = sphere_config[i, 3:7])
            
            self.obs_ids.append(obs_id)
    
    def spawn_collision_cuboids(self, cuboid_config, color = None, grasp = False):

        if grasp == True:
            baseMass = 0.001
            color = 'black'
        else:
            baseMass = 0.

        for i in range(cuboid_config.shape[0]):
            
            cuid = self.client_id.createCollisionShape(p.GEOM_BOX, 
                                                       halfExtents = cuboid_config[i, 7:]/2)
            
            vuid = self.client_id.createVisualShape(p.GEOM_BOX, 
                                                    halfExtents = cuboid_config[i, 7:]/2,
                                                    rgbaColor = np.hstack([self.colors[color], np.array([1.0])]))

            obs_id = self.client_id.createMultiBody(baseMass = baseMass,
                                                    baseCollisionShapeIndex = cuid,
                                                    baseVisualShapeIndex = vuid, 
                                                    basePosition = cuboid_config[i, :3], 
                                                    baseOrientation = cuboid_config[i, 3:7])
            
            self.obs_ids.append(obs_id)

    def spawn_collision_cylinders(self, cylinder_config, color = None, grasp = False):

        if grasp == True:
            baseMass = 0.001
            color = 'green'
        else:
            baseMass = 0.
            color = 'yellow'

        for i in range(cylinder_config.shape[0]):
            
            cuid = self.client_id.createCollisionShape(p.GEOM_CYLINDER, 
                                                       radius = cylinder_config[i, 7],
                                                       height = cylinder_config[i, 8])                                                   
            
            vuid = self.client_id.createVisualShape(p.GEOM_CYLINDER, 
                                                    radius = cylinder_config[i, 7],
                                                    length = cylinder_config[i, 8],
                                                    rgbaColor = np.hstack([self.colors[color], np.array([1.0])]))

            obs_id = self.client_id.createMultiBody(baseMass = baseMass,
                                                    baseCollisionShapeIndex = cuid, 
                                                    baseVisualShapeIndex = vuid, 
                                                    basePosition = cylinder_config[i, :3], 
                                                    baseOrientation = cylinder_config[i, 3:7])
            
            self.obs_ids.append(obs_id)

    def spawn_collision_spheres(self, sphere_config, color = None, grasp = False):

        if grasp == True:
            baseMass = 0.001
            color = 'red'
        else:
            baseMass = 0.
            color = 'yellow'
        
        for i in range(sphere_config.shape[0]):
            
            cuid = self.client_id.createCollisionShape(p.GEOM_SPHERE, 
                                                       radius = sphere_config[i, 7])                                                   
            
            vuid = self.client_id.createVisualShape(p.GEOM_SPHERE, 
                                                    radius = sphere_config[i, 7],
                                                    rgbaColor = np.hstack([self.colors[color], np.array([1.0])]))

            obs_id = self.client_id.createMultiBody(baseMass = baseMass,
                                                    baseCollisionShapeIndex = cuid, 
                                                    baseVisualShapeIndex = vuid, 
                                                    basePosition = sphere_config[i, :3], 
                                                    baseOrientation = sphere_config[i, 3:7])
            
            self.obs_ids.append(obs_id)
    
    def clear_obstacles(self):

        for id in self.obs_ids:
            self.client_id.removeBody(id)
        self.obs_ids = []
    
    def clip_joints(self, joints):

        return np.clip(joints, self.joint_lower_limits[np.newaxis, :, np.newaxis], self.joint_upper_limits[np.newaxis, :, np.newaxis])
    
    def draw_link_bounding_boxes(self):

        self.link_poses = []
        self.link_bounding_vertices = []
        
        self.link_bounding_objs = []

        for link_index in range(0, 11):     # The 12th link (i.e. 11th index) is the grasp target and is not needed   
                
            if link_index not in [8, 9]:    

                link_name = self.link_index_to_name[link_index+1]

                l, b, h = self.link_dimensions[link_name][0], self.link_dimensions[link_name][1], self.link_dimensions[link_name][2]

                if link_index == 7:
                    frame_pos, _ = self.client_id.getLinkState(self.manipulator, 7)[4:6]
                    _, frame_ori = self.client_id.getLinkState(self.manipulator, 10)[4:6]
                elif link_index != -1:
                    frame_pos, frame_ori = self.client_id.getLinkState(self.manipulator, link_index)[4:6]
                else:
                    frame_pos, frame_ori = self.client_id.getBasePositionAndOrientation(self.manipulator)
                world_transform = self.pose_to_transformation(np.array([*frame_pos, *frame_ori]))

                link_dimensions = self.link_dimensions[link_name].copy()
                if link_index == 10:
                    link_dimensions[1] *= 4

                world_link_center = (world_transform @ np.vstack((np.expand_dims(self.link_centers[link_name], 1), 1)))[:-1, 0]

            vertices = np.array([[-l/2, -b/2, -h/2],
                                [ l/2, -b/2, -h/2],
                                [ l/2,  b/2, -h/2],
                                [-l/2,  b/2, -h/2],
                                [-l/2, -b/2,  h/2],
                                [ l/2, -b/2,  h/2],
                                [ l/2,  b/2,  h/2],
                                [-l/2,  b/2,  h/2]])
            vertices = vertices + np.array([self.link_centers[link_name]])
            vertices = world_transform @ np.vstack((vertices.T, np.ones(8)))
            self.link_bounding_vertices.append(vertices.T[:, :-1])

            self.link_poses.append(np.array([*world_link_center, *frame_ori]))
            
            vuid = self.client_id.createVisualShape(p.GEOM_BOX, 
                                    halfExtents = link_dimensions/2,
                                    rgbaColor = np.hstack([self.colors['red'], np.array([1.0])]))
            
            obj_id = self.client_id.createMultiBody(baseVisualShapeIndex = vuid, 
                                                    basePosition = world_link_center, 
                                                    baseOrientation = frame_ori)
            
            self.link_bounding_objs.append(obj_id)
    
    def clear_bounding_boxes(self):

        for obj_id in self.link_bounding_objs:
            self.client_id.removeBody(obj_id)
    
    def get_joint_positions(self):

        return np.array([self.client_id.getJointState(self.manipulator, i)[0] for i in self.joints])
    
    def get_joint_velocities(self):

        return np.array([self.client_id.getJointState(self.manipulator, i)[1] for i in self.joints])

    def get_tf_mat(self, i, joint_angles):
        
        dh_params = np.array([[0, 0.333, 0, joint_angles[0]],
                    [0, 0, -np.pi / 2, joint_angles[1]],
                    [0, 0.316, np.pi / 2, joint_angles[2]],
                    [0.0825, 0, np.pi / 2, joint_angles[3]],
                    [-0.0825, 0.384, -np.pi / 2, joint_angles[4]],
                    [0, 0, np.pi / 2, joint_angles[5]],
                    [0.088, 0, np.pi / 2, joint_angles[6]],
                    [0, 0.107, 0, 0],
                    [0, 0, 0, -np.pi / 4],
                    [0.0, 0.1034, 0, 0]], dtype=np.float64)
        
        a = dh_params[i][0]
        d = dh_params[i][1]
        alpha = dh_params[i][2]
        theta = dh_params[i][3]
        q = theta

        return np.array([[np.cos(q), -np.sin(q), 0, a],
                        [np.sin(q) * np.cos(alpha), np.cos(q) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
                        [np.sin(q) * np.sin(alpha), np.cos(q) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                        [0, 0, 0, 1]])

    def draw_frame(self, transform, scale_factor = 0.2):

        unit_axes_world = np.array([[scale_factor, 0, 0], 
                                    [0, scale_factor, 0], 
                                    [0, 0, scale_factor],
                                    [1, 1, 1]])
        axis_points = ((transform @ unit_axes_world)[:3, :]).T
        axis_center = transform[:3, 3]

        l1 = self.client_id.addUserDebugLine(axis_center, axis_points[0], self.colors['red'], lineWidth = 4)
        l2 = self.client_id.addUserDebugLine(axis_center, axis_points[1], self.colors['green'], lineWidth = 4)
        l3 = self.client_id.addUserDebugLine(axis_center, axis_points[2], self.colors['blue'], lineWidth = 4)

        frame_id = [l1, l2, l3]

        return frame_id[:]
    
    def remove_frame(self, frame_id):

        for id in frame_id:
            self.client_id.removeUserDebugItem(id)
        
    def inverse_of_transform(self, matrix):

        # Extract the rotation part and translation part of the matrix
        rotation_part = matrix[:3, :3]
        translation_part = matrix[:3, 3]
        
        # Calculate the inverse of the rotation part
        inverse_rotation = np.linalg.inv(rotation_part)
        
        # Calculate the new translation by applying the inverse rotation
        inverse_translation = -inverse_rotation.dot(translation_part)
        
        # Create the inverse transformation matrix
        inverse_matrix = np.zeros_like(matrix)
        inverse_matrix[:3, :3] = inverse_rotation
        inverse_matrix[:3, 3] = inverse_translation
        inverse_matrix[3, 3] = 1.0
        
        return inverse_matrix.copy()
    
    def forward_kinematics(self, joint_angles):

        T_EE = np.identity(4)
        for i in range(7 + 3):
            T_EE = T_EE @ self.get_tf_mat(i, joint_angles)

        return T_EE
    
    def get_jacobian(self, joint_angles):

        T_EE = self.get_ee_tf_mat(joint_angles)

        J = np.zeros((6, 10))
        T = np.identity(4)
        for i in range(7 + 3):
            T = T @ self.get_tf_mat(i, joint_angles)

            p = T_EE[:3, 3] - T[:3, 3]
            z = T[:3, 2]

            J[:3, i] = np.cross(z, p)
            J[3:, i] = z

        return J[:, :7]
    
    def inverse_kinematics(self, point, euler_orientation = None):

        if type(euler_orientation) == type(None):
            point = self.client_id.calculateInverseKinematics(self.manipulator, self.end_effector, point)
        else:
            quat = self.client_id.getQuaternionFromEuler(euler_orientation)
            point = self.client_id.calculateInverseKinematics(self.manipulator, self.end_effector, point, quat)

        return point
    
    def get_end_effector_pose(self):

        pos, ori = self.client_id.getLinkState(self.manipulator, 11, computeForwardKinematics = 1)[:2]
        pose = np.array([*pos, *ori])

        return pose
    
    def get_end_effector_transformation(self):

        pose = self.get_end_effector_pose()        
        transform = self.pose_to_transformation(pose)

        return transform

    def pose_to_transformation(self, pose):

        pos = pose[:3]
        quat = pose[3:]

        rotation_matrix = self.quaternion_to_rotation_matrix(quat)

        transform = np.zeros((4, 4))
        transform[:3, :3] = rotation_matrix.copy()
        transform[:3, 3] = pos.copy()
        transform[3, 3] = 1

        return transform

    def euler_to_rotation_matrix(yaw, pitch, roll):
        
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

        Rx = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

        R = Rz @ (Ry @ Rx)
        
        return R
    
    def quaternion_to_rotation_matrix(self, quat):
        """
        Convert a quaternion to a rotation matrix.
        
        :param q: Quaternion [w, x, y, z]
        :return: 3x3 rotation matrix
        """
        # w, x, y, z = quat
        # rotation_matrix = np.array([[1 - 2*y**2 - 2*z**2,  2*x*y - 2*z*w,        2*x*z + 2*y*w],
        #                             [2*x*y + 2*z*w,        1 - 2*x**2 - 2*z**2,  2*y*z - 2*x*w],
        #                             [2*x*z - 2*y*w,        2*y*z + 2*x*w,        1 - 2*x**2 - 2*y**2]])
        
        mat = np.array(self.client_id.getMatrixFromQuaternion(quat))
        rotation_matrix = np.reshape(mat, (3, 3))

        return rotation_matrix
        
    def rotation_matrix_to_quaternion(self, R):
        
        trace = np.trace(R)

        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

        return np.array([qx, qy, qz, qw])
    
    def transformation_to_pose(self, T):

        pose = np.zeros((7,))
        pose[:3] = T[:3, 3]
        pose[3:] = self.rotation_matrix_to_quaternion(T[:3, :3])

        return pose.copy()
    
    def hold_object(self, obj_dims, disp):

        self.client_id.resetJointState(self.manipulator, self.joints[-1] + 3, obj_dims[1]/2)
        self.client_id.resetJointState(self.manipulator, self.joints[-1] + 4, obj_dims[1]/2)

        grasp_obj_config = np.zeros((1, 10))

        grasp_obj_config[0, 7:] = np.array(obj_dims)
        # axes = [perpendicular to hand plane (up from robot persp), 
        #         joint extension axis (right from robot persp), 
        #         pointing straight outwards]
        # max = [?, 0.08, ?]

        link_pose = np.zeros((7,))
        link_pose[:3], link_pose[3:] = self.client_id.getLinkState(self.manipulator, 11)[:2]

        transform = self.pose_to_transformation(link_pose)
        #transform[2, 3] = transform[2, 3] + disp

        # Angle in radians (90 degrees)
        angle_rad = np.radians(90)

        # Create the rotation matrix for a 90-degree rotation about the Y-axis
        rotation_matrix = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])

        grasp_obj_config[0, :3] = (transform @ np.array([[0., 0, disp, 1]]).T).T[0, :3]
        grasp_obj_config[0, 3:7] = self.rotation_matrix_to_quaternion(transform[:3, :3])
        # grasp_obj_config[0, 3:7] = self.rotation_matrix_to_quaternion(transform[:3, :3] @ rotation_matrix)

        # grasp_obj_config[0, :3], grasp_obj_config[0, 3:7] = self.client_id.getLinkState(self.manipulator, 11)[:2]
        # grasp_obj_config[0, 2] = grasp_obj_config[0, 2] + disp

        self.spawn_collision_cuboids(grasp_obj_config, grasp = True)
        grasp_obj_id = self.obs_ids[-1]

        # Create a fixed joint between the robot link and the object
        fixed_joint = self.client_id.createConstraint(
            parentBodyUniqueId = self.manipulator,
            parentLinkIndex = 11,  # Replace with the index of the link you want to attach to
            childBodyUniqueId = grasp_obj_id,
            childLinkIndex = -1,  # Use -1 for the base of the object
            jointType = p.JOINT_FIXED,
            jointAxis = [0, 0, 0],  # Unused for fixed joint
            parentFramePosition = [0, 0, 0],
            childFramePosition = [0, 0, 0],  # Relative to the object's origin
        )
    
    # def move_joints(self, target_joint_pos, speed = 0.01, timeout = 10):

    #     t0 = time.time()
    #     all_joints = []

    #     iters = 0
    #     # time.sleep(10)
    #     # while (time.time() - t0) < timeout:
    #     while iters < 500000:
          
    #         current_joint_pos = self.get_joint_positions()
    #         error = target_joint_pos - current_joint_pos
    #         # print(error)
    #         if all(np.abs(error) < 1e-2):
    #             for _ in range(10):
    #                 print("Reached")
    #                 self.client_id.stepSimulation()     # Give time to stop
    #             #     if self.benchmarking:
    #             #         self.check_collisions()
    #             #         if self.num_collisions > 0:
    #             #             break
    #             # if self.benchmarking:
    #             #     if self.num_collisions > 0:
    #             #             break
    #             return True, all_joints
    #         # print(target_joint_pos)
    #         norm = np.linalg.norm(error)
    #         vel = error / norm if norm > 0 else 0
    #         # print(norm)
    #         next_joint_pos = current_joint_pos + vel * speed

    #         all_joints.append(next_joint_pos)

    #         self.client_id.setJointMotorControlArray(           # Move with constant velocity
    #             bodyIndex = self.manipulator,
    #             jointIndices = self.joints,
    #             controlMode = p.POSITION_CONTROL,
    #             targetPositions = next_joint_pos,
    #             positionGains = 0.01*np.ones(len(self.joints)),
    #         )
    #         # self.spawn_single_point(next_joint_pos)
    #         # error = next_joint_pos - current_joint_pos
    #         # print(iters)
    #         self.client_id.stepSimulation()
    #         iters += 1
    #         # if self.benchmarking:     
    #         #     self.check_collisions()
    #         #     if self.num_collisions > 0:
    #         #         break

    #     print(f"Warning: move_joints exceeded {timeout} second timeout. Skipping.")

    #     return False, all_joints


    def move_joints(self, target_joint_pos, speed=0.0001, timeout=10):
        t0 = time.time()
        all_joints = []

        iters = 0
        max_iters = 10000  # Reduced iteration limit for more efficient testing

        while (time.time() - t0) < timeout and iters < max_iters:
            current_joint_pos = self.get_joint_positions()
            error = target_joint_pos - current_joint_pos

            # Check if we are within the tolerance range
            if np.all(np.abs(error) < 1e-2):
                for _ in range(10):
                    print("Reached target joint positions")
                    self.client_id.stepSimulation()  # Give time to stabilize
                return True, all_joints

            # Calculate the norm and handle potential numerical issues
            norm = np.linalg.norm(error)
            # if norm > 1e-3:
            #     vel = (error / norm) * speed
            # else:
            vel = error / norm if norm > 0 else 0
            
            # vel = np.zeros_like(error)

            # Calculate the next joint positions
            next_joint_pos = current_joint_pos + vel

            # Save the joint positions for debugging or analysis
            all_joints.append(next_joint_pos)

            # Apply the position control
            self.client_id.setJointMotorControlArray(
                bodyIndex=self.manipulator,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=next_joint_pos,
                positionGains= 0.01*np.ones(len(self.joints)),  # Increased gains for better responsiveness
            )

            # Step the simulation
            self.client_id.stepSimulation()
            iters += 1

            # Optional: Debugging information
            if iters % 100 == 0:  # Print every 100 iterations for readability
                print(f"Iteration {iters}: Norm = {norm}, Error = {error}")

        print(f"Warning: move_joints exceeded {timeout} seconds or {max_iters} iterations. Skipping.")

        return False, all_joints

    # def move_joints(self, target_joint_pos, speed = 0.1, timeout = 10):

    #     t0 = time.time()
    #     all_joints = []

    #     iters = 0

    #     # while (time.time() - t0) < timeout:
    #     while iters < 500000:
          
    #         current_joint_pos = self.get_joint_positions()
    #         error = target_joint_pos - current_joint_pos
    #         # print(error)
    #         if all(np.abs(error) < 1e-2):
    #             for _ in range(10):
    #                 print("Reached")
    #                 self.client_id.stepSimulation()     # Give time to stop
    #             #     if self.benchmarking:
    #             #         self.check_collisions()
    #             #         if self.num_collisions > 0:
    #             #             break
    #             # if self.benchmarking:
    #             #     if self.num_collisions > 0:
    #             #             break
    #             return True, all_joints
    #         # print(target_joint_pos)
    #         norm = np.linalg.norm(error)
    #         vel = error / norm if norm > 0 else 0
    #         # print(norm)
    #         next_joint_pos = current_joint_pos + vel * speed

    #         all_joints.append(next_joint_pos)

    #         self.client_id.setJointMotorControlArray(           # Move with constant velocity
    #             bodyIndex = self.manipulator,
    #             jointIndices = self.joints,
    #             controlMode = p.POSITION_CONTROL,
    #             targetPositions = next_joint_pos,
    #             positionGains = 0.01*np.ones(len(self.joints)),
    #         )
    #         # self.spawn_single_point(next_joint_pos)
    #         # error = next_joint_pos - current_joint_pos
    #         camera_position = [1.5, 1.5, 1.2]  # Adjusted position to get a wider view
    #         target_position = [0, 0, 0]  # Focus on the center of the scene
    #         up_vector = [0, 0, 1]  # Keep the up direction constant
        
    #         view_matrix = p.computeViewMatrix(camera_position, target_position, up_vector)
    #         projection_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.33, nearVal=0.1, farVal=100)

            


    #         width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
    #             width=640, height=480, viewMatrix=view_matrix, projectionMatrix=projection_matrix)

    #         # Debug: Check if rgb_img is None
    #         if rgb_img is None:
    #             print(f"No image captured for step {iters}")
    #             continue
            
    #         # Save the image using OpenCV
    #         rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGBA2BGR)
    #         image_path = f"images2/step_{iters:04d}.png"
    #         cv2.imwrite(image_path, rgb_img)

    #         # Debug: Confirm image is saved
    #         if os.path.isfile(image_path):
    #             print(f"Image saved: {image_path}")
    #         else:
    #             print(f"Failed to save image: {image_path}")
    #         self.client_id.stepSimulation()
    #         iters += 1
    #         # if self.benchmarking:     
    #         #     self.check_collisions()
    #         #     if self.num_collisions > 0:
    #         #         break

    #     print(f"Warning: move_joints exceeded {timeout} second timeout. Skipping.")

    #     return False, all_joints
    
    def go_home(self):

        for i, joint_ind in enumerate(self.joints):
            self.client_id.resetJointState(self.manipulator, joint_ind, 0)

    def check_collisions(self):

        # There are 12 links.

        # Collision Info format:
        # ((0, 0, 7, 6, -1, (0.6231421349501027, -0.20546030368880996, 0.4579269918307523), 
        #   (0.6231421349501027, -0.20546030368881, 0.45900286827236414), (-4.511815760890017e-15, 2.2559078804450087e-14, -1.0), 
        #   0.0010758764416118338, 5678.289055003103, 185.98474464217657, (0.0, 1.0, 2.2559078804450087e-14), 1407.3361669376961, 
        #   (1.0, 1.0178240730107782e-28, -4.511815760890017e-15)),)

        self.num_timesteps += 1

        for obs_id in range(len(self.obs_ids)):

            info = self.client_id.getContactPoints(self.manipulator, self.obs_ids[obs_id]) #, link_index)
            if len(info) > 0:
                self.num_collisions += 1
                break
    
    # def spawn_points(self, trajectory, color = 'black'):
    #     """
    #     Expected trajectory shape is (7, num_waypoints)
    #     """

    #     points = np.zeros((trajectory.shape[1] - 2, 3))
    #     for i in range(1, trajectory.shape[1] - 1):
    #         points[i-1] = self.forward_kinematics(trajectory[:, i])[:3, 3]
    #     colors = np.repeat([self.colors[color]], trajectory.shape[1] - 2, axis = 0)

    #     self.points_id = self.client_id.addUserDebugPoints(points, colors, 15)
    #     self.points_list.append(self.points_id)

    # def spawn_points(self, trajectory, color='black'):
    #     """
    #     Spawns points in the environment along the given trajectory.

    #     Expected trajectory shape is (7, num_waypoints).
    #     """
    #     points = np.zeros((trajectory.shape[1] - 2, 3))
    #     for i in range(1, trajectory.shape[1] - 1):
    #         # Calculate the 3D position of each point along the trajectory
    #         points[i-1] = self.forward_kinematics(trajectory[:, i])[:3, 3]

    #     # Define the color of the points
    #     colors = np.repeat([self.colors[color]], len(points), axis=0)

    #     # Add debug points to the environment
    #     self.points_id = self.client_id.addUserDebugPoints(points, colors, 15)
    #     self.points_list.append(self.points_id)
    # def spawn_points(self, trajectory, color='green'):
    #     """
    #     Spawns points in the environment along the given trajectory.
    #     """
    #     points = np.zeros((trajectory.shape[1] - 2, 3))
    #     for i in range(1, trajectory.shape[1] - 1):
    #         # Calculate the 3D position of each point along the trajectory
    #         points[i-1] = self.forward_kinematics(trajectory[:, i])[:3, 3]
    #         print(f"Point {i-1}: {points[i-1]}")  # Debug print

    #     # Define the color of the points
    #     colors = np.repeat([self.colors[color]], len(points), axis=0)

    #     # Add debug points to the environment
    #     if hasattr(self, 'points_list'):
    #         for point_id in self.points_list:
    #             self.client_id.removeUserDebugItem(point_id)
    #     self.points_id = self.client_id.addUserDebugPoints(points, colors, 15)
    #     self.points_list = [self.points_id]
    def spawn_single_point(self, joint_angles, color='green'):
        """
        Spawns a single point in the environment based on the given joint angles.
        Uses a visual shape to represent the point.
        """
        # Perform forward kinematics to get the 3D position from the joint angles
        position = self.forward_kinematics(joint_angles)[:3, 3]  # Extract the position from the transformation matrix
        # print(f"Calculated position from joint angles: {position}")

        # Define color
        rgbaColor = [0, 1, 0, 1] if color == "green" else [0, 0, 1, 1]

        # Remove previous points if any
        # if hasattr(self, 'point_ids'):
        #     for point_id in self.point_ids:
        #         self.client_id.removeBody(point_id)

        # Create the visual shape for the point
        visual_shape_id = self.client_id.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.02,
            rgbaColor=rgbaColor
        )

        # Spawn the point at the calculated position
        body_id = self.client_id.createMultiBody(
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position
        )

        # Store the point ID to remove later if needed
        self.point_ids = [body_id]

        # print(f"Spawned point at: {position}")

    def spawn_points(self, trajectory, color='black'):
        """
        Spawns points in the environment along the given trajectory.
        Uses visual shapes to represent points.
        """
        points = np.zeros((trajectory.shape[1] - 2, 3))
        for i in range(1, trajectory.shape[1] - 1):
            points[i-1] = self.forward_kinematics(trajectory[:, i])[:3, 3]
            print(f"Point {i-1}: {points[i-1]}")  # Debug print

        # Define color
        rgbaColor = [0, 1, 0, 1] if color == "green" else [1, 0, 0, 1]

        # Remove previous points if any
        if hasattr(self, 'point_ids'):
            for point_id in self.point_ids:
                self.client_id.removeBody(point_id)

        self.point_ids = []

        for point in points:
            visual_shape_id = self.client_id.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.02,
                rgbaColor=rgbaColor
            )
            body_id = self.client_id.createMultiBody(
                baseVisualShapeIndex=visual_shape_id,
                basePosition=point
            )
            self.point_ids.append(body_id)

        # Ensure the points are stored
        if not hasattr(self, 'point_ids'):
            self.point_ids = []

    def remove_all_points(self):

        for id in self.points_list:
            self.client_id.removeUserDebugItem(id)
        self.points_list = []

    def remove_all_text(self):

        self.client_id.removeAllUserDebugItems()

    def add_timestep_text(self, t, position = [1, 1, 1], size = 2, early_stop = False):

        if not early_stop:
            text = "Timestep = " + str(int(t))
            color = "black"
        else:
            text = "Timestep = " + str(int(t)) + ", Collision-free trajectory found!"
            color = "dark_green"
        
        self.client_id.addUserDebugText(text = text,
                                        textPosition = position,
                                        textColorRGB = self.colors[color],
                                        textSize = size,
                                        lifeTime = 0)
    
    # def execute_trajectory(self, trajectory,trajectories=None,currind=None,holding_object = False, obj_dims = None, disp = None):
            
    #     for i, joint_ind in enumerate(self.joints):
    #         self.client_id.resetJointState(self.manipulator, joint_ind, trajectory[i, 0])
        
    #     # if holding_object:
    #     #     self.hold_object(obj_dims, disp)

    #     # self.client_id.setJointMotorControlArray(           # Stabilize Robot
    #     #         bodyIndex = self.manipulator,
    #     #         jointIndices = self.joints,
    #     #         controlMode = p.POSITION_CONTROL,
    #     #         targetPositions = trajectory[:, 0],
    #     #         positionGains = np.zeros((len(self.joints))),
    #     #     )
        
    #     _ = input("Press Enter to Execute trajectory")

    #     for i in range(1, trajectory.shape[-1]):
    #         # appear_t = 2
    #         # spawn_t = 10
    #         # switch_t = 5

    #         # trans = self.forward_kinematics(trajectory[:,spawn_t])
    #         # pos = trans.squeeze()[:3,3]

    #         # if(i==appear_t):
    #         #     new_obstacle = np.array([[pos[0], pos[1], pos[2], 0, 0, 0.7061983, 0.70801413, 0.03, 0.03, 0.03]]) 
    #         #     self.spawn_cuboids(new_obstacle, color = "blue")

    #         time.sleep(0.4)
    #         # current_joints = np.array([self.client_id.getJointState(self.manipulator, i)[0] for i in self.joints])
    #         target_joints = trajectory[:, i]
    #         # print(f"Current Joints: {current_joints}")
    #         # print(f"Target Joints: {target_joints}")
    #         # print(f"Itr number: {i}")

    #         if any(target_joints <= self.joint_lower_limits) or any(target_joints >= self.joint_upper_limits):

    #             print("Joint Limits Exceeded")

    #         col_fn = planners.get_collision_fn(self.manipulator,self.joints,self,obstacles=self.obs_ids)
    #         if(col_fn(target_joints)):
    #             print("collision")
    #         # else:
    #         #     print("\rno col",end='')

    #         for i, joint_ind in enumerate(self.joints):
    #             self.client_id.resetJointState(self.manipulator, joint_ind, target_joints[i])

    #         # self.move_joints(target_joints)
    #         # self.client_id.stepSimulation()
                
    # def execute_trajectory(self, trajectory, trajectories=None, currind=None, holding_object=False, obj_dims=None, disp=None):
        
    #     # Create folder for images if it doesn't exist
    #     os.makedirs("images", exist_ok=True)

    #     self.spawn_points(trajectory)
    #     for i, joint_ind in enumerate(self.joints):
    #         self.client_id.resetJointState(self.manipulator, joint_ind, trajectory[i, 0])
        
    #     _ = input("Press Enter to Execute trajectory")

    #     for step in range(1, trajectory.shape[-1]):
    #         time.sleep(0.4)
    #         target_joints = trajectory[:, step]

    #         if any(target_joints <= self.joint_lower_limits) or any(target_joints >= self.joint_upper_limits):
    #             print("Joint Limits Exceeded")
            
    #         col_fn = planners.get_collision_fn(self.manipulator, self.joints, self, obstacles=self.obs_ids)
    #         if col_fn(target_joints):
    #             print("collision")

    #         for i, joint_ind in enumerate(self.joints):
    #             self.client_id.resetJointState(self.manipulator, joint_ind, target_joints[i])

    #         # Place camera and take a snapshot
    #         camera_position = [1, 1, 1]  # Example position
    #         target_position = [0, 0, 0]  # Example target (where the camera is looking)
    #         up_vector = [0, 0, 1]  # Up direction

    #         view_matrix = p.computeViewMatrix(camera_position, target_position, up_vector)
    #         projection_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1, nearVal=0.1, farVal=100)

    #         width, height, rgb_img, _, _ = p.getCameraImage(
    #             width=640, height=480, viewMatrix=view_matrix, projectionMatrix=projection_matrix)

    #         # Save image using OpenCV
    #         rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGBA2BGR)
    #         cv2.imwrite(f"images/step_{step:04d}.png", rgb_img)

    def execute_trajectory(self, trajectory, trajectories=None, currind=None, holding_object=False, obj_dims=None, disp=None,coeffs = False):

        # Create folder for images if it doesn't exist
        if(not coeffs):
            os.makedirs("images_final_trajectory", exist_ok=True)
        else:
            os.makedirs("images_denoised_coeffs", exist_ok=True)

        # Spawn points in the environment
        self.spawn_points(trajectory)

        for i, joint_ind in enumerate(self.joints):
            self.client_id.resetJointState(self.manipulator, joint_ind, trajectory[i, 0])

        _ = input("Press Enter to Execute trajectory")

        for step in range(1, trajectory.shape[-1]):
            time.sleep(0.4)
            target_joints = trajectory[:, step]

            if any(target_joints <= self.joint_lower_limits) or any(target_joints >= self.joint_upper_limits):
                print("Joint Limits Exceeded")

            col_fn = planners.get_collision_fn(self.manipulator, self.joints, self, obstacles=self.obs_ids)
            if col_fn(target_joints):
                print("collision")

            for i, joint_ind in enumerate(self.joints):
                self.client_id.resetJointState(self.manipulator, joint_ind, target_joints[i])

            # Camera placement
            camera_position = [1.5, 1.5, 1.2]  # Adjusted position to get a wider view
            target_position = [0, 0, 0]  # Focus on the center of the scene
            up_vector = [0, 0, 1]  # Keep the up direction constant
        
            view_matrix = p.computeViewMatrix(camera_position, target_position, up_vector)
            projection_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.33, nearVal=0.1, farVal=100)

            


            width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
                width=640, height=480, viewMatrix=view_matrix, projectionMatrix=projection_matrix)

            # Debug: Check if rgb_img is None
            if rgb_img is None:
                print(f"No image captured for step {step}")
                continue
            
            # Save the image using OpenCV
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGBA2BGR)
            if(not coeffs):
                image_path = f"images_final_trajectory/step_{step:04d}.png"
            else:
                image_path = f"images_denoised_coeffs/step_{step:04d}.png"

            cv2.imwrite(image_path, rgb_img)

            # Debug: Confirm image is saved
            if os.path.isfile(image_path):
                print(f"Image saved: {image_path}")
            else:
                print(f"Failed to save image: {image_path}")

    def excecute_stitching_nearest_multi(self, trajectory,trajectories,currtraj): # switch to nearest point in trajs which is collision free 
            
        for i, joint_ind in enumerate(self.joints):
            self.client_id.resetJointState(self.manipulator, joint_ind, trajectory[i, 0])
        
        
        spawn_t = 20
        trans = self.forward_kinematics(trajectory[:,spawn_t])
        pos = trans.squeeze()[:3,3]

        new_obstacle = np.array([[pos[0], pos[1], pos[2], 0, 0, 0.7061983, 0.70801413, 0.03, 0.03, 0.03]]) 
        # _ = input("Press Enter to start")

        i = 1
        while(i < trajectory.shape[-1]):
            window_size = 5
            if(i == 10):
                self.spawn_collision_cuboids(new_obstacle, color = "blue")

            time.sleep(0.3)
            target_joints = trajectory[:, i]
            ## collision window 
            
            win_col = False

            col_fn = planners.get_collision_fn(self.manipulator,self.joints,self,obstacles=self.obs_ids)
            check_joints = trajectory[:, i : i + min(window_size,trajectory.shape[-1] - i -1)].T
            for j in check_joints:
                if(col_fn(j)):
                    print("collision detected in forward window")
                    win_col = True
                    break
                
            if(win_col):
                stitch = None
                for ii,traj in enumerate(trajectories):
                    k = self.find_closest_pt(target_joints,traj.T)
                    check_joints = traj[:, k : k + min(window_size,traj.shape[-1] - k -1)].T
                    new_traj_col = False
                    for j in check_joints:
                        if(col_fn(j)):
                            print(ii,"th potential traj has collision detected in forward window")
                            new_traj_col = True
                            break
                    if(new_traj_col):
                        continue
                    
                    stitch = planners.plan_birrt(self,start=target_joints,goal=traj[:,k])
                    if(stitch is None):
                        print("aah no valid transition path found for this switch")
                        continue
                    trajectory = traj
                    target_joints = traj[:,k]
                    i = k+1
                    self.remove_all_points()
                    self.spawn_points(trajectory,'green')
                    self.spawn_points(np.array(stitch).T,'red')
                    break
                    
                if(stitch is None):
                    print("NO valid traj found, implement next win later, now just continuing")
                    i+=1
                    continue


                for kk in range(len(stitch)):
                    temptarget_joints = stitch[kk]
                    if any(temptarget_joints <= self.joint_lower_limits) or any(temptarget_joints >= self.joint_upper_limits):

                        print("Joint Limits Exceeded")

                    for f, joint_ind in enumerate(self.joints):
                        self.client_id.resetJointState(self.manipulator, joint_ind, temptarget_joints[f])
                        
                    time.sleep(0.4)

                # i = i + 20
                target_joints = trajectory[:, i]

            if any(target_joints <= self.joint_lower_limits) or any(target_joints >= self.joint_upper_limits):

                print("Joint Limits Exceeded")

            for a, joint_ind in enumerate(self.joints):
                self.client_id.resetJointState(self.manipulator, joint_ind, target_joints[a])
            
            i += 1

            # self.move_joints(target_joints)
            # self.client_id.stepSimulation()





    def excecute_stitching_same_only(self, trajectory,trajectories,currtraj):
            
        for i, joint_ind in enumerate(self.joints):
            self.client_id.resetJointState(self.manipulator, joint_ind, trajectory[i, 0])
        
        
        spawn_t = 20
        trans = self.forward_kinematics(trajectory[:,spawn_t])
        pos = trans.squeeze()[:3,3]

        new_obstacle = np.array([[pos[0], pos[1], pos[2], 0, 0, 0.7061983, 0.70801413, 0.03, 0.03, 0.03]]) 
        # _ = input("Press Enter to start")

        i = 1
        while(i < trajectory.shape[-1]):
            window_size = 5
            if(i == 10):
                self.spawn_collision_cuboids(new_obstacle, color = "blue")

            time.sleep(0.4)
            target_joints = trajectory[:, i]
            ## collision window 
            
            win_col = False

            col_fn = planners.get_collision_fn(self.manipulator,self.joints,self,obstacles=self.obs_ids)
            check_joints = trajectory[:, i : i + min(window_size,trajectory.shape[-1] - i -1)].T
            for j in check_joints:
                if(col_fn(j)):
                    print("collision detected in forward window")
                    win_col = True
                    break
                
            if(win_col):
                stitch = None
                for ii in range(i + min(window_size,trajectory.shape[-1] - i -1) , trajectory.shape[-1] - 1): # change this to another max size, rather than searching till the end
                    check_joints = trajectory[:, ii : ii + min(window_size,trajectory.shape[-1] - ii -1)].T
                    new_pt_col = False
                    for j in check_joints:
                        if(col_fn(j)):
                            # print(ii,"th potential traj has collision detected in forward window")
                            new_pt_col = True
                            break
                    if(new_pt_col):
                        continue
                    
                    stitch = planners.plan_birrt(self,start=target_joints,goal=trajectory[:,ii])
                    if(stitch is None):
                        print("aah no valid transition path found for this switch")
                        continue
                    # trajectory = traj
                    target_joints = trajectory[:,ii]
                    i = ii+1
                    # self.remove_all_points()
                    # self.spawn_points(trajectory,'black')
                    self.spawn_points(np.array(stitch).T,'red')
                    break
                    
                if(stitch is None):
                    print("NO valid traj found, implement next win later, now just continuing")
                    i+=1
                    continue


                for kk in range(len(stitch)):
                    temptarget_joints = stitch[kk]
                    if any(temptarget_joints <= self.joint_lower_limits) or any(temptarget_joints >= self.joint_upper_limits):

                        print("Joint Limits Exceeded")

                    for f, joint_ind in enumerate(self.joints):
                        self.client_id.resetJointState(self.manipulator, joint_ind, temptarget_joints[f])
                        
                    time.sleep(0.4)

                # i = i + 20
                target_joints = trajectory[:, i]

            if any(target_joints <= self.joint_lower_limits) or any(target_joints >= self.joint_upper_limits):

                print("Joint Limits Exceeded")

            for a, joint_ind in enumerate(self.joints):
                self.client_id.resetJointState(self.manipulator, joint_ind, target_joints[a])
            
            i += 1

            # self.move_joints(target_joints)
            # self.client_id.stepSimulation()



    

    def excecute_stitching(self, trajectory,trajectories,currtraj): 
            
        for i, joint_ind in enumerate(self.joints):
            self.client_id.resetJointState(self.manipulator, joint_ind, trajectory[i, 0])
        
        
        spawn_t = 20
        trans = self.forward_kinematics(trajectory[:,spawn_t])
        pos = trans.squeeze()[:3,3]

        new_obstacle = np.array([[pos[0], pos[1], pos[2], 0, 0, 0.7061983, 0.70801413, 0.03, 0.03, 0.03]]) 
        # _ = input("Press Enter to start")

        i = 1
        while(i < trajectory.shape[-1]):
            window_size = 5
            if(i == 10):
                self.spawn_collision_cuboids(new_obstacle, color = "blue")

            time.sleep(0.3)
            target_joints = trajectory[:, i]
            ## collision window 
            
            win_col = False

            col_fn = planners.get_collision_fn(self.manipulator,self.joints,self,obstacles=self.obs_ids)
            check_joints = trajectory[:, i : i + min(window_size,trajectory.shape[-1] - i -1)].T
            for j in check_joints:
                if(col_fn(j)):
                    print("collision detected in forward window")
                    win_col = True
                    break
                
            if(win_col):
                stitch = None
                for ii,traj in enumerate(trajectories):
                    k = self.find_closest_pt(target_joints,traj.T)
                    check_joints = traj[:, k : k + min(window_size,traj.shape[-1] - k -1)].T
                    new_traj_col = False
                    for j in check_joints:
                        if(col_fn(j)):
                            print(ii,"th potential traj has collision detected in forward window")
                            new_traj_col = True
                            break
                    if(new_traj_col):
                        continue
                    
                    stitch = planners.plan_birrt(self,start=target_joints,goal=traj[:,k])
                    if(stitch is None):
                        print("aah no valid transition path found for this switch")
                        continue
                    trajectory = traj
                    target_joints = traj[:,k]
                    i = k+1
                    self.remove_all_points()
                    self.spawn_points(trajectory,'green')
                    self.spawn_points(np.array(stitch).T,'red')
                    break
                    
                if(stitch is None):
                    print("NO valid traj found, now trying same traj new point")
                    
                    stitch = None
                    for ii in range(i + min(window_size,trajectory.shape[-1] - i -1) , trajectory.shape[-1] - 1): # change this to another max size, rather than searching till the end
                        check_joints = trajectory[:, ii : ii + min(window_size,trajectory.shape[-1] - ii -1)].T
                        new_pt_col = False
                        for j in check_joints:
                            if(col_fn(j)):
                                # print(ii,"th potential traj has collision detected in forward window")
                                new_pt_col = True
                                break
                        if(new_pt_col):
                            continue
                        
                        stitch = planners.plan_birrt(self,start=target_joints,goal=trajectory[:,ii])
                        if(stitch is None):
                            print("aah no valid transition path found for this switch")
                            continue
                        # trajectory = traj
                        target_joints = trajectory[:,ii]
                        i = ii+1
                        self.remove_all_points()
                        self.spawn_points(trajectory,'black')
                        self.spawn_points(np.array(stitch).T,'red')
                        break
                        
                    if(stitch is None):
                        print("NO valid traj found, implement next win later, now just continuing")
                        i+=1
                        continue



                for kk in range(len(stitch)):
                    temptarget_joints = stitch[kk]
                    if any(temptarget_joints <= self.joint_lower_limits) or any(temptarget_joints >= self.joint_upper_limits):

                        print("Joint Limits Exceeded")

                    for f, joint_ind in enumerate(self.joints):
                        self.client_id.resetJointState(self.manipulator, joint_ind, temptarget_joints[f])
                        
                    time.sleep(0.4)

                # i = i + 20
                target_joints = trajectory[:, i]

            if any(target_joints <= self.joint_lower_limits) or any(target_joints >= self.joint_upper_limits):

                print("Joint Limits Exceeded")

            for a, joint_ind in enumerate(self.joints):
                self.client_id.resetJointState(self.manipulator, joint_ind, target_joints[a])
            
            i += 1

            # self.move_joints(target_joints)
            # self.client_id.stepSimulation()




    

    def find_closest_pt(self,pt,traj):
        d_fn = pp.interfaces.planner_interface.joint_motion_planning.get_distance_fn(self.manipulator,self.joints)
        min_d = float('inf')
        min_i = 0
        for i , p in enumerate(traj):
            d = d_fn(pt,p)   
            if d < min_d:
                min_d = d
                min_i = i

        return min_i

            


    def benchmark_trajectory(self, trajectory, metrics_calculator):

        self.benchmarking = True

        self.num_timesteps = 0
        self.num_collisions = 0
        self.spawn_points(trajectory)
        for j, joint_ind in enumerate(self.joints):
            self.client_id.resetJointState(self.manipulator, joint_ind, trajectory[j, 0])

        self.client_id.setJointMotorControlArray(           # Stabilize Robot
                bodyIndex = self.manipulator,
                jointIndices = self.joints,
                controlMode = p.POSITION_CONTROL,
                targetPositions = trajectory[:, 0],
                positionGains = np.zeros((len(self.joints))),
            )

        dt = [0]

        for i in range(1, trajectory.shape[-1]):
            time.sleep(0.4)
            target_joints = trajectory[:, i]

            if any(target_joints <= self.joint_lower_limits) or any(target_joints >= self.joint_upper_limits):

                print("Joint Limits Exceeded")

            self.move_joints(target_joints)
            self.client_id.stepSimulation()
            self.check_collisions()
            if self.num_collisions > 0:
                break
            dt.append(self.num_timesteps * self.timestep)

        dt = np.diff(dt, n=1)[:, np.newaxis]

        collision_percentage = np.round((self.num_collisions / self.num_timesteps) * 100, 2)
        success = 1 if self.num_collisions == 0 else 0

        joint_smoothness, end_eff_smoothness = metrics_calculator.smoothness_metric(trajectory, self.timestep)
        joint_path_length, end_eff_path_length = metrics_calculator.path_length_metric(trajectory)

        self.benchmarking = False
        self.remove_all_points()
        return success, joint_path_length, end_eff_path_length, joint_smoothness[0], end_eff_smoothness[0]



        




