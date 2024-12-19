import numpy as np
import cv2
import time
import math
import os
import heapq as heap
from mazelib import Maze 
from mazelib.generate.Prims import Prims
import matplotlib.pyplot as plt
from collections import defaultdict
import torch.nn.functional as F
import torch

class Maze:

    def __init__(self):
        
        pass

    def generate_maze(self, maze_width, maze_height):

        if maze_height % 2 == 0 or maze_width % 2 == 0:
            raise Exception("Only Odd-numbered widths are allowed")
        
        maze = Maze()

        maze.generator = Prims((maze_height+1)//2 , (maze_width+1)//2)
        maze.generate()

        return maze.grid[1:-1, 1:-1]

    def convert_maze_to_image(self, generated_maze, image_width, image_height):

        maze_width = generated_maze.shape[1]
        maze_height = generated_maze.shape[0]
        
        cell_width = image_width // maze_width
        cell_height = image_height // maze_height

        pad_width_left = (image_width % maze_width) // 2
        pad_width_right = pad_width_left + ((image_width % maze_width) % 2)
        pad_length_up = (image_height % maze_height) // 2
        pad_length_down = pad_length_up + ((image_height % maze_height) % 2)

        maze_image = np.zeros((image_height - (pad_length_up + pad_length_down), image_width - (pad_width_left + pad_width_right)), dtype = np.uint8)

        for row in range(maze_height):
            for col in range(maze_width):

                if generated_maze[row, col] == 1:
                    maze_image[row * cell_height : (row+1) * cell_height, col * cell_width : (col+1) * cell_width] = 255
                elif generated_maze[row, col] == 0:
                    maze_image[row * cell_height : (row+1) * cell_height, col * cell_width : (col+1) * cell_width] = 0

        maze_image = np.pad(maze_image, ((pad_length_up, pad_length_down), (pad_width_left, pad_width_right)), 'edge')

        return maze_image.copy()
    
    def point_to_pixel(self, x, limit = 1024, bound = 1, return_int = True):
        """
        Image should always be square with equal and opposite bounds
        """
        if return_int == True:
            return ((limit / (2 * bound)) * (x + bound)).astype('int')
        else:
            return ((limit / (2 * bound)) * (x + bound))

    def pixel_to_point(self, p, limit = 1024, bound = 1):
        """
        Image should always be square with equal and opposite bounds
        """

        return (2 * bound / limit) * p - bound
    
    def plot_maze(self, maze):

        plt.imshow(np.rot90(1 - maze), cmap = 'gray')
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis (axis labels are reversed)")

    def get_env_image_tensors(self, mazes, device, image_width = 16, image_height = 16):

        maze_width = mazes.shape[-1]    #y
        maze_height = mazes.shape[-2]   #x
        
        cell_width = image_width // maze_width      #y
        cell_height = image_height // maze_height   #x

        pad_width_left = (image_width % maze_width) // 2
        pad_width_right = pad_width_left + ((image_width % maze_width) % 2)
        pad_length_up = (image_height % maze_height) // 2
        pad_length_down = pad_length_up + ((image_height % maze_height) % 2)

        images = np.kron(mazes, np.ones((1, cell_height, cell_width)))
        images = np.pad(images, ((0, 0), (pad_length_up, pad_length_down), (pad_width_left, pad_width_right)), 'edge')
        images = images[:, np.newaxis, :, :] * 255

        images = torch.tensor(images, dtype = torch.float32, device = device)

        return images
    
    def plot_trajectory_on_maze(self, points, maze, color = 'blue'):
        """
        points => (samples x 2 x length) array
        """

        points = np.array(points)
        
        pixel_coords = self.point_to_pixel(points, maze.shape[0], return_int = False)
        self.plot_maze(maze)
        plt.scatter(pixel_coords[:, 0, :] - 0.5, maze.shape[1] - 0.5 - pixel_coords[:, 1, :], color = color)

    def plot_point(self, image, point, color):

        center = self.point_to_pixel(point, return_int = True)
        radius = 8
        cv2.circle(image, center, radius, color, -1)
    
    def plot_trajectory(self, image, trajectory):

        # Start:
        self.plot_point(image, trajectory[:, 0], color = [0, 0, 255])

        # Trajectory:
        for i in range(1, trajectory.shape[-1] - 1):
            self.plot_point(image, trajectory[:, i], color = [255, 0, 0])

        # Goal:
        self.plot_point(image, trajectory[:, -1], color = [0, 0, 0])

class Maze2DPrior(Maze):

    def __init__(self, traj_len = 128):
        
        self.c = 2
        self.n = traj_len

        Maze.__init__(self)

    # ------------------ A-STAR FUNCTIONS ------------------ #
    
    def distance_between(self, node1, node2):
        
        return np.linalg.norm(np.array(node1) - np.array(node2))

    def is_goal(self, node, goal):

        if self.distance_between(node, goal) <= 0.1:
            return True
        else:
            return False
    
    def get_adjacent_nodes(self, node, maze, q_min, q_max):
        
        adj_nodes = np.zeros((0, 2))
        
        theta = np.linspace(start = 0, stop = 2*np.pi, num = 4 + 1)[:-1]
        x_diff = np.cos(theta).astype('int')
        y_diff = np.sin(theta).astype('int')
        temp_adj_nodes = np.zeros((theta.shape[0], 2))
        temp_adj_nodes[:, 0] = (node[0] + x_diff)
        temp_adj_nodes[:, 1] = (node[1] + y_diff)
        temp_adj_nodes = temp_adj_nodes.astype('int')
        
        for i in range(temp_adj_nodes.shape[0]):

            if np.all(temp_adj_nodes[i] >= q_min) and np.all(temp_adj_nodes[i] <= q_max):
                if not maze[temp_adj_nodes[i, 0], temp_adj_nodes[i, 1]]:
                    adj_nodes = np.concatenate([adj_nodes, [temp_adj_nodes[i]]], axis = 0)

        return adj_nodes.copy()
    
    def run_a_star(self, start, goal, maze, max_iter = 1000):

        start_time = time.time()
        q_min = np.array([0, 0])
        q_max = np.array(maze.shape) - 1
        
        # Get data on current node (now the start node):
        start, goal = tuple(start), tuple(goal)
        start_node = (self.distance_between(start, goal), start)

        # Initialize:
        iter = 0
        nodes = []
        heap.heapify(nodes)

        # Dictionary indicating actual costs of each node:
        actualCostMap = defaultdict(lambda: float('inf'))
        actualCostMap[start] = 0

        # Push the current node (start node) into heap
        heap.heappush(nodes, start_node)

        # Nodes that have to be explored:
        open_nodes = set()
        open_nodes.add(start)

        parentDict = {}
        closed_nodes = set()

        while iter < max_iter and len(nodes) > 0:

            iter += 1

            # Choose the node at the top of the heap to explore and remove it from open nodes:
            heurCost, curr_node = heap.heappop(nodes)
            open_nodes.remove(curr_node)

            # Check if the current node is the goal, if it is return it with its dictionary of parents:
            if self.is_goal(curr_node, goal):
                time_taken = time.time() - start_time
                #print("A-star took " + str(np.round(time_taken, 2)) + " seconds to converge")
                return curr_node, parentDict, time_taken, iter    
            
            # Add the current node to the list of closed nodes:
            closed_nodes.add(curr_node)

            # Get neighbouring nodes:
            adj_nodes = self.get_adjacent_nodes(curr_node, maze, q_min, q_max)

            # Loop through all neighbours:
            for i in range(adj_nodes.shape[0]):

                adj_node = tuple(adj_nodes[i])

                # Get the edge cost of the adjacent node using euclidean distance and add it to cost of current node
                newEdgeCost = actualCostMap[curr_node] + self.distance_between(curr_node, adj_node)
                # Get the heuristic cost, i.e. distance to goal
                heurCost = self.distance_between(adj_node, goal)
                # Get the previously stored cost of next point for comparison
                currCostofAdjPoint = actualCostMap[adj_node]

                # If we found a lower cost for an adjacent point, update the cost of that point:
                if currCostofAdjPoint > newEdgeCost:
                    
                    # Update parents dictionary, the cost map and get the total cost of this node
                    parentDict[adj_node] = curr_node
                    actualCostMap[adj_node] = newEdgeCost
                    totalCost = heurCost + newEdgeCost

                    # print("Pushed the node with total cost = ", totalCost)
                    # print("--------------------------------------------------------")

                    # If the node is not open for exploration, add it to open nodes and push it into the heap
                    if adj_node not in open_nodes:
                        open_nodes.add(adj_node)
                        heap.heappush(nodes, (totalCost, adj_node))

        time_taken = time.time() - start_time
        
        return curr_node, parentDict, time_taken, iter 
    
    def plan_a_star_path(self, start, goal, maze, max_iter = 1000):

        output = self.run_a_star(start, goal, maze, max_iter)

        start = tuple(start)

        if output == None:
            print("A-star did not find a path")
            return None
        
        else:
            
            end_node, parentDic, time_taken, num_iterations = output
            end_node = tuple(end_node)
            
            path = [end_node]

            while(end_node != start):
                
                end_node = tuple(parentDic[end_node])
                path.append(end_node)
            

            path.reverse()

            return np.array(path), time_taken, num_iterations
 
    # ------------------------------------------------------ #
    
    def generate_random_start_goal(self, maze):

        free_indices = np.array(np.where(1 - maze)).T

        start_index, goal_index = np.random.choice(np.arange(free_indices.shape[0]), 2, replace = False)
        start_pixel, goal_pixel = free_indices[start_index], free_indices[goal_index]

        start = np.array(start_pixel, dtype = np.int32)
        goal = np.array(goal_pixel, dtype = np.int32)

        return start, goal
    
    def expand_path(self, path):

        expanded_path = np.zeros((2 * path.shape[0] - 1, 2))

        i = np.arange(path.shape[0]) * 2
        j = i[:-1] + 1
        
        expanded_path[i, :] = path[:, :]
        expanded_path[j, :] = (path[:-1, :] + path[1:, :]) / 2

        return expanded_path.copy()
    
    def randomize_path(self, path):

        new_path = np.random.uniform(low = path, high = path + 1)

        path_diff = np.sum(np.abs(path[2:] - path[:-2]), axis = 1)
        turn_indices = np.where(path_diff == 2)[0].astype('int') + 1
        new_path[turn_indices, :] = path[turn_indices, :] #+ 0.25 * path[turn_indices, :]

        return new_path
    
    def bezier_function(self, t, points):

        P = points.reshape((1, -1, 2))
        
        out = np.zeros((P[:, 0, :].shape))
        n = P.shape[1]

        for i in range(n):

            out = out + P[:, i, :] * math.comb(n-1, i) * (t**i) * ((1-t)**(n-1-i))

        return out
    
    def generate_bezier_curve_from_points(self, P):

        # Define t values (parameter values)
        t = np.linspace(0, 1, self.n)

        # # Generate the bezier curve points
        curve_points = np.array([self.bezier_function(t_i, P)[0] for t_i in t])

        return curve_points.copy()
    
    def augment_maze(self, maze, p = [0.125, 0.125, 0.125, 0.125, 0.5]):

        augmented_maze = np.zeros(maze.shape)
        augment = np.random.choice([1, 2, 3, 4, 5], p = p)
        # Right shift:
        if augment == 1:
            augmented_maze[1:, :] = maze[:-1, :]
        # Up shift:
        elif augment == 2:
            augmented_maze[:, 1:] = maze[:, :-1]
        # Left shift:
        elif augment == 3:
            augmented_maze[:-1, :] = maze[1:, :]
        # Down shift:
        elif augment == 4:
            augmented_maze[:, :-1] = maze[:, 1:]
        # No shift:
        else:
            augmented_maze = maze.copy()

        return augmented_maze
    
    def plan_trajectory(self, num_samples, maze_index, maze_type = "15x15"):
        
        trajectories = np.zeros((num_samples, 2, self.n))

        maze_folder = "environments/2D_maze/" + maze_type + "/"
        
        for i in range(num_samples):

            maze = np.load(maze_folder + "maze" + str(maze_index) + ".npy")

            start, goal = self.generate_random_start_goal(maze)

            path, _, _ = self.plan_a_star_path(start, goal)

            for j in range(3):
                path = self.expand_path(path)

            trajectories[i] = self.generate_bezier_curve_from_points(path, self.n).T

        return trajectories

    def plan_random_trajectories(self, num_samples, maze_folder) -> (np.ndarray, np.ndarray):

        trajectories = np.zeros((num_samples, 2, self.n))

        _, _, maze_files = next(os.walk(maze_folder))
        num_mazes = len(maze_files)
        maze_indices = np.random.choice(range(1, num_mazes + 1), size = num_samples)

        dummy_maze = np.load(maze_folder + "maze1.npy")
        maze_size = dummy_maze.shape[0]

        environments = np.zeros((num_samples, maze_size, maze_size))
        
        for i in range(num_samples):

            maze = np.load(maze_folder + "maze" + str(maze_indices[i]) + ".npy")

            augmented_maze = self.augment_maze(maze)
            
            start, goal = self.generate_random_start_goal(augmented_maze)

            path, _, _ = self.plan_a_star_path(start, goal, augmented_maze)

            path = self.randomize_path(path)

            for j in range(2):
                path = self.expand_path(path)

            trajectories[i] = self.generate_bezier_curve_from_points(path).T
            environments[i] = augmented_maze.copy()

        trajectories = self.pixel_to_point(trajectories, maze_size)

        return trajectories, environments   

    def generate_trajectories(self, num_samples):

        trajectories = np.zeros((num_samples, 2, self.n))
        
        save_folder = "assets/dataset/5x5_maze_trajectories/"

        _, _, files = next(os.walk(save_folder))
        num_trajs = len(files)
        traj_indices = np.random.choice(range(1, num_trajs + 1), size = num_samples, replace = False)

        for i in range(num_samples):

            trajectories[i] = np.load(save_folder + "trajectory" + str(traj_indices[i]) + ".npy")

        return trajectories.copy()
    
    def sample_tasks(self, maze_folder, num_samples, repeat = False):

        _, _, maze_files = next(os.walk(maze_folder))
        num_mazes = len(maze_files)
        maze_indices = np.random.choice(range(1, num_mazes + 1), size = num_samples, replace = repeat)

        maze = np.load(maze_folder + "maze" + str(maze_indices[0]) + ".npy")
        augmented_maze = self.augment_maze(maze)
        start, goal = self.generate_random_start_goal(augmented_maze)

        maze_size = maze.shape[-1]

        envs = np.zeros((num_samples, maze_size, maze_size))
        start_pixels = np.zeros((num_samples, 2))
        goal_pixels = np.zeros((num_samples, 2))

        envs[0, :, :] = augmented_maze
        start_pixels[0, :] = start
        goal_pixels[0, :] = goal

        for i in range(1, num_samples):

            maze = np.load(maze_folder + "maze" + str(maze_indices[i]) + ".npy")
            augmented_maze = self.augment_maze(maze)
            start, goal = self.generate_random_start_goal(augmented_maze)

            envs[i, :, :] = augmented_maze
            start_pixels[i, :] = start
            goal_pixels[i, :] = goal

        start_pixels = np.random.uniform(low = start_pixels, high = start_pixels + 1)
        goal_pixels = np.random.uniform(low = goal_pixels, high = goal_pixels + 1)

        start_points = self.pixel_to_point(start_pixels, limit = maze_size)
        goal_points = self.pixel_to_point(goal_pixels, limit = maze_size)

        return envs, start_points, goal_points

    def embed_paths_in_images(self, images, paths, device):
        '''
        paths -> (batch, 2, traj_len)
        '''
        img_size = images.size()[-1]
        # Convert the path into pixel coordinates:
        pixel_paths = self.point_to_pixel(paths, limit = img_size, return_int = False)
        # Convert the paths to singular indices. Paths are in (b, 2, 128) format

        path_1d_indices = torch.floor(pixel_paths[:, 0, :]) * img_size + torch.floor(pixel_paths[:, 1, :])

        # Make an image indices array:
        img_1d_indices = torch.reshape(torch.arange(img_size**2, dtype = torch.float32, device = device), (img_size, img_size))
        traj_embed = torch.abs(img_1d_indices.unsqueeze(0).unsqueeze(0) - path_1d_indices.unsqueeze(-1).unsqueeze(-1))
        traj_embed = torch.clamp(torch.min(traj_embed, axis = 1)[0], 0, 1)
        traj_embed = (1 - traj_embed).unsqueeze(1)
        # Checked through plots, everything is fine and aligns till here.

        out = torch.cat((images, traj_embed), dim = 1)

        return out
    
    def check_collision(self, envs, paths):

        env_size = envs.shape[-1]
        pixel_paths = self.point_to_pixel(paths, limit = env_size, return_int = False)

        path_1d_indices = np.floor(pixel_paths[:, 0, :]) * env_size + np.floor(pixel_paths[:, 1, :])

        env_1d_indices = np.reshape(np.arange(env_size**2), (env_size, env_size))
        traj_embed = np.abs(env_1d_indices[np.newaxis, np.newaxis, :, :] - path_1d_indices[:, :, np.newaxis, np.newaxis])
        traj_embed = np.clip(np.min(traj_embed, axis = 1), 0, 1)
        traj_embed = 1 - traj_embed

        overlap = np.multiply(envs, traj_embed)
        collisions = np.clip(np.sum(overlap, axis = (1, 2)), 0, 1)[:, np.newaxis]
        # 1 for collision, 0 for no collisions

        return collisions

        

    