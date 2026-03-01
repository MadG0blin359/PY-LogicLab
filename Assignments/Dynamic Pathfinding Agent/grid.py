import pygame
import random
from settings import *

class Node:
    def __init__(self, row, col, width, total_rows, total_cols):
        self.row = row
        self.col = col
        self.x = col * width
        self.y = row * width
        self.color = WHITE
        self.width = width
        self.total_rows = total_rows
        self.total_cols = total_cols
        
        # Pathfinding tracking
        self.g = float('inf')
        self.h = 0
        self.f = float('inf')
        self.parent = None
        
        self.is_start = False
        self.is_goal = False

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == YELLOW

    def is_barrier(self):
        return self.color == BLACK

    def is_start_node(self):
        return self.is_start

    def is_goal_node(self):
        return self.is_goal

    def reset(self):
        self.color = WHITE
        self.g = float('inf')
        self.h = 0
        self.f = float('inf')
        self.parent = None
        self.is_start = False
        self.is_goal = False

    def reset_algo_state(self):
        if not self.is_start_node() and not self.is_goal_node() and not self.is_barrier():
            self.color = WHITE
        self.g = float('inf')
        self.h = 0
        self.f = float('inf')
        self.parent = None

    def make_start(self):
        self.is_start = True
        self.color = ORANGE

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = YELLOW

    def make_barrier(self):
        self.color = BLACK

    def make_goal(self):
        self.is_goal = True
        self.color = PURPLE

    def make_path(self):
        self.color = GREEN
        
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        # DOWN
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row + 1][self.col])
        # UP
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row - 1][self.col])
        # RIGHT
        if self.col < self.total_cols - 1 and not grid[self.row][self.col + 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col + 1])
        # LEFT
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False

class Grid:
    def __init__(self, rows, cols, pixel_width, pixel_height):
        self.rows = rows
        self.cols = cols
        self.width = pixel_width
        self.height = pixel_height
        self.node_width = self.width // self.cols
        self.node_height = self.height // self.rows
        
        # Ensure it's square cells for uniformity based on width mostly
        self.node_size = min(self.node_width, self.node_height)
        
        # Recalculate working grid sizes
        self.working_width = self.cols * self.node_size
        self.working_height = self.rows * self.node_size

        self.grid = []
        self.start_node = None
        self.goal_node = None
        self.build_grid()

    def build_grid(self):
        self.grid = []
        for i in range(self.rows):
            self.grid.append([])
            for j in range(self.cols):
                node = Node(i, j, self.node_size, self.rows, self.cols)
                self.grid[i].append(node)
                
        # Set default start and goal
        self.start_node = self.grid[self.rows//2][1]
        self.start_node.make_start()
        
        self.goal_node = self.grid[self.rows//2][self.cols-2]
        self.goal_node.make_goal()
        
        self.update_all_neighbors()

    def draw_grid_lines(self, win):
        for i in range(self.rows):
            pygame.draw.line(win, GRAY, (0, i * self.node_size), (self.working_width, i * self.node_size))
        for j in range(self.cols):
            pygame.draw.line(win, GRAY, (j * self.node_size, 0), (j * self.node_size, self.working_height))
            
        # Draw borders of the grid area
        pygame.draw.rect(win, BLACK, (0, 0, self.working_width, self.working_height), 2)

    def draw(self, win):
        # Background for the grid area
        pygame.draw.rect(win, WHITE, (0, 0, self.width, self.height))
        for row in self.grid:
            for node in row:
                node.draw(win)
        self.draw_grid_lines(win)

    def get_clicked_pos(self, pos):
        y, x = pos
        if x >= self.working_width or y >= self.working_height:
            return None
        row = y // self.node_size
        col = x // self.node_size
        if row < self.rows and col < self.cols:
            return row, col
        return None

    def reset_all_algo_states(self):
        for row in self.grid:
            for node in row:
                node.reset_algo_state()
                
    def generate_random_maze(self, density=0.3):
        # Clear existing walls but keep start/goal
        for row in self.grid:
            for node in row:
                if not node.is_start_node() and not node.is_goal_node():
                    node.reset()

        # Randomly place walls
        for row in self.grid:
            for node in row:
                if not node.is_start_node() and not node.is_goal_node():
                    if random.random() < density:
                        node.make_barrier()
        self.update_all_neighbors()

    def update_all_neighbors(self):
        for row in self.grid:
            for node in row:
                node.update_neighbors(self.grid)

    def get_node(self, row, col):
        return self.grid[row][col]
