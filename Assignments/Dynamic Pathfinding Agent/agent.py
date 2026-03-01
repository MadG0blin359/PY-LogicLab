import random
from settings import *
import pygame

class Agent:
    def __init__(self, start_node, path_nodes):
        self.current_node = start_node
        self.path = path_nodes
        self.path_index = 0
        if len(self.path) > 0 and self.path[0] == start_node:
            self.path_index = 1 # Skip start node if it's the first in path
        
        # We also want to keep the agent's color visible
        # We can draw the agent on top of the grid or have the grid node colored as agent
        
    def spawn_dynamic_obstacle(self, grid_obj, spawn_prob=0.05):
        """Randomly spawn a new obstacle if dynamic mode is on"""
        if random.random() < spawn_prob:
            # Pick a random node that is not start, goal, or already a wall
            # Also don't spawn exactly on the agent
            while True:
                r = random.randint(0, grid_obj.rows - 1)
                c = random.randint(0, grid_obj.cols - 1)
                node = grid_obj.get_node(r, c)
                
                if (not node.is_start_node() and 
                    not node.is_goal_node() and 
                    not node.is_barrier() and
                    node != self.current_node):
                    
                    node.make_barrier()
                    grid_obj.update_all_neighbors() # Important to update neighbors
                    return True # Spawned
            
        return False
        
    def move_step(self):
        """Moves one step along the path if possible"""
        if self.path_index >= len(self.path):
            return True, True # Done moving (Reached end)
            
        next_node = self.path[self.path_index]
        
        # Check if the next node is blocked (dynamic obstacle spawned)
        if next_node.is_barrier():
            return False, False # Path blocked, need replan
            
        # Move
        self.current_node = next_node
        self.path_index += 1
        return True, False # Moved successfully, not done yet
        
    def draw(self, win):
        """Draw a distinct circle to represent the agent"""
        x = self.current_node.x + self.current_node.width // 2
        y = self.current_node.y + self.current_node.width // 2
        radius = self.current_node.width // 2 - 2
        
        pygame.draw.circle(win, CYAN, (x, y), radius)
        # Outline
        pygame.draw.circle(win, BLACK, (x, y), radius, 1)
