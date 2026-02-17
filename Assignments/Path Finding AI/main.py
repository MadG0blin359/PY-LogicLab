import pygame
import sys
from logic import bfs, dfs

# Constants
WIDTH, HEIGHT = 600, 600
ROWS, COLS = 20, 20
CELL = WIDTH // COLS
DELAY = 10 

# Colors
WHITE  = (245, 245, 245)
BLACK  = (40, 40, 40)      # Static Walls
GREEN  = (39, 174, 96)     # Start (S)
RED    = (192, 57, 43)     # Target (T)
BLUE   = (41, 128, 185)    # Frontier Nodes
GRAY   = (127, 140, 141)   # Explored Nodes
YELLOW = (241, 196, 15)    # Final Path

class PathfinderApp:
    def __init__(self):
        pygame.init()
        # Requirement: Mandatory Title
        pygame.display.set_caption("GOOD PERFORMANCE TIME APP")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        self.start = (2, 2)
        self.target = (ROWS - 5, COLS - 5)
        self.setup_walls()

    def setup_walls(self):
        # Static walls to navigate around
        for i in range(5, 15):
            self.grid[i][10] = -1

    def draw_node(self, pos, color):
        r, c = pos
        pygame.draw.rect(self.screen, color, (c * CELL, r * CELL, CELL - 1, CELL - 1))
        pygame.display.update()
        pygame.time.delay(DELAY)

    def gui_callback(self, pos, node_type):
        """Mandatory GUI visualization update"""
        if pos == self.start or pos == self.target:
            return
        # Visually distinguish frontier and explored
        color = BLUE if node_type == "frontier" else GRAY
        self.draw_node(pos, color)

    def render_base(self):
        self.screen.fill(WHITE)
        for r in range(ROWS):
            for c in range(COLS):
                color = WHITE
                if (r, c) == self.start: color = GREEN
                elif (r, c) == self.target: color = RED
                elif self.grid[r][c] == -1: color = BLACK
                pygame.draw.rect(self.screen, color, (c * CELL, r * CELL, CELL - 1, CELL - 1))
        pygame.display.update()

    def run_algorithm(self, algo_func):
        self.render_base() # Clear previous search visualization
        path = algo_func(self.start, self.target, ROWS, COLS, self.grid, self.gui_callback)
        if path:
            # Highlight the final successful route
            for p in path:
                if p != self.start and p != self.target:
                    self.draw_node(p, YELLOW)

    def main_loop(self):
        self.render_base()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1: # BFS
                        self.run_algorithm(bfs)
                    if event.key == pygame.K_2: # DFS
                        self.run_algorithm(dfs)
                    if event.key == pygame.K_r: # Reset
                        self.render_base()

if __name__ == "__main__":
    app = PathfinderApp()
    app.main_loop()