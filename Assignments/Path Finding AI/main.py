import pygame
import sys
from algorithms import get_neighbors, bfs

# --- Constants & Configuration ---
WIDTH, HEIGHT = 600, 600
# You can adjust ROWS/COLS for best/worst case scenarios later
ROWS, COLS = 20, 20 
CELL_SIZE = WIDTH // COLS

# Mandatory Title 
APP_TITLE = "GOOD PERFORMANCE TIME APP"

# Colors for Visualization [cite: 2346, 2347, 2348]
WHITE = (255, 255, 255)  # Empty
BLACK = (0, 0, 0)        # Wall
GREEN = (0, 255, 0)      # Start (S)
RED = (255, 0, 0)        # Target (T)
BLUE = (0, 0, 255)       # Frontier (Queue/Stack)
GRAY = (128, 128, 128)   # Explored
YELLOW = (255, 255, 0)   # Final Path

class Pathfinder:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(APP_TITLE)
        self.grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        self.start = (2, 2)
        self.target = (ROWS - 3, COLS - 3)
        
    def draw_grid(self):
        for r in range(ROWS):
            for c in range(COLS):
                color = WHITE
                if (r, c) == self.start: color = GREEN
                elif (r, c) == self.target: color = RED
                elif self.grid[r][c] == -1: color = BLACK # Static Wall
                
                pygame.draw.rect(self.screen, color, 
                                 (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1))

    def run(self):
        while True:
            self.screen.fill(BLACK)
            self.draw_grid()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            pygame.display.update()

    def draw_node_callback(self, pos, color):
        r, c = pos
        # Redraw a single rectangle in the grid
        pygame.draw.rect(self.screen, color, 
                        (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1))
        pygame.display.update()

    def trigger_bfs(self):
        final_path = bfs(
            self.start, self.target, ROWS, COLS, self.grid, 
            self.draw_node_callback, self.spawn_dynamic_obstacle
        )
        if final_path:
            self.draw_final_path(final_path)

if __name__ == "__main__":
    app = Pathfinder()
    app.run()