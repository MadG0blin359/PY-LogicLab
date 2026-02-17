import pygame
import sys
from logic import bfs

# Configuration
WIDTH, HEIGHT = 600, 600
ROWS, COLS = 20, 20
CELL = WIDTH // COLS
DELAY = 15  # Animation delay in ms

# Colors [cite: 41, 42, 43]
WHITE = (240, 240, 240)
BLACK = (30, 30, 30)     # Walls
GREEN = (46, 204, 113)   # Start
RED = (231, 76, 60)      # Target
BLUE = (52, 152, 219)    # Frontier
GRAY = (149, 165, 166)   # Explored
YELLOW = (241, 196, 15)  # Final Path

class PathfinderApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("AI Pathfinder")
        self.grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        self.start = (2, 2)
        self.target = (ROWS - 5, COLS - 5)
        # Example wall for testing
        for i in range(5, 15):
            self.grid[i][10] = -1

    def draw_node(self, pos, color):
        r, c = pos
        pygame.draw.rect(self.screen, color, (c * CELL, r * CELL, CELL - 1, CELL - 1))
        pygame.display.update()
        pygame.time.delay(DELAY)

    def update_gui_callback(self, pos, node_type):
        """Callback for logic.py to update visualization step-by-step."""
        if pos == self.start or pos == self.target:
            return
        
        color = BLUE if node_type == "frontier" else GRAY
        self.draw_node(pos, color)

    def render_base_grid(self):
        self.screen.fill(WHITE)
        for r in range(ROWS):
            for c in range(COLS):
                color = WHITE
                if (r, c) == self.start: color = GREEN
                elif (r, c) == self.target: color = RED
                elif self.grid[r][c] == -1: color = BLACK
                pygame.draw.rect(self.screen, color, (c * CELL, r * CELL, CELL - 1, CELL - 1))
        pygame.display.update()

    def run(self):
        self.render_base_grid()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        path = bfs(self.start, self.target, ROWS, COLS, self.grid, self.update_gui_callback)
                        if path:
                            for p in path:
                                if p != self.start and p != self.target:
                                    self.draw_node(p, YELLOW)

if __name__ == "__main__":
    app = PathfinderApp()
    app.run()