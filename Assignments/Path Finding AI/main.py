import pygame
import sys
from logic import bfs, dfs, ucs

# --- High-Contrast Cyber Theme ---
CLR_BG      = (15, 15, 20)     # Darkest Navy
CLR_SIDEBAR = (25, 25, 35)     # Deep Slate
CLR_GRID    = (35, 35, 45)     # Base Cell
CLR_WALL    = (70, 75, 90)     # Concrete Wall
CLR_START   = (0, 255, 127)    # Neon Spring Green
CLR_TARGET  = (255, 46, 99)    # Neon Crimson
CLR_FRONT   = (0, 217, 255)    # Cyan (Frontier)
CLR_VISITED = (60, 60, 80)     # Muted Purple-Gray (Explored)
CLR_PATH    = (255, 211, 0)    # Cyber Yellow (Final Route)
CLR_TEXT    = (230, 230, 240)  # Off-White

WIDTH, HEIGHT = 900, 650
GRID_SIZE = 550
ROWS, COLS = 20, 20
CELL = GRID_SIZE // COLS
OFF_X, OFF_Y = 280, 50

class PathfinderPro:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("AI Search Laboratory")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.font_m = pygame.font.SysFont('Consolas', 24, bold=True)
        self.font_s = pygame.font.SysFont('Consolas', 16)
        self.grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        self.start, self.target = (2, 2), (ROWS-5, COLS-5)
        self.is_running = False 
        self._init_walls()

    def _init_walls(self):
        for i in range(4, 16): self.grid[i][10] = -1

    def draw_cell(self, r, c, color, border=False):
        rect = (c * CELL + OFF_X, r * CELL + OFF_Y, CELL - 2, CELL - 2)
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        if border: pygame.draw.rect(self.screen, CLR_TEXT, rect, 1, border_radius=4)

    def gui_callback(self, pos, node_type):
        if pos == self.start or pos == self.target: return
        color = CLR_FRONT if node_type == "frontier" else CLR_VISITED
        self.draw_cell(pos[0], pos[1], color)
        pygame.display.update()
        pygame.time.delay(10) # Step-by-step animation

    def render_ui(self, status="READY"):
        self.screen.fill(CLR_BG)
        pygame.draw.rect(self.screen, CLR_SIDEBAR, (0, 0, OFF_X - 20, HEIGHT))
        
        # Labels
        self.screen.blit(self.font_m.render("ENGINE CONTROL", True, CLR_TEXT), (30, 50))
        controls = [
            ("[1] BFS (Shortest)", CLR_FRONT),
            ("[2] DFS (Deep)", CLR_TARGET),
            ("[3] UCS (Cost-Based)", CLR_PATH),
            ("[R] Reset Canvas", CLR_TEXT)
        ]
        for i, (t, c) in enumerate(controls):
            self.screen.blit(self.font_s.render(t, True, c), (30, 120 + (i * 45)))
        
        st_text = self.font_s.render(f">> {status}", True, CLR_PATH if "SUCCESS" in status else CLR_TEXT)
        self.screen.blit(st_text, (30, HEIGHT - 60))

        for r in range(ROWS):
            for c in range(COLS):
                color = CLR_GRID
                if (r, c) == self.start: color = CLR_START
                elif (r, c) == self.target: color = CLR_TARGET
                elif self.grid[r][c] == -1: color = CLR_WALL
                self.draw_cell(r, c, color)
        pygame.display.update()

    def start_search(self, name, algo):
        if self.is_running: return
        self.is_running = True
        self.render_ui(f"STATUS: {name}...")
        path = algo(self.start, self.target, ROWS, COLS, self.grid, self.gui_callback)
        
        if path:
            for p in path:
                if p != self.start and p != self.target:
                    self.draw_cell(p[0], p[1], CLR_PATH, border=True)
                    pygame.display.update()
                    pygame.time.delay(25)
            # The path stays visible because we DON'T call render_ui() immediately here
            self.is_running = False 
        else:
            self.render_ui("STATUS: NO PATH FOUND")
            self.is_running = False

    def loop(self):
        self.render_ui()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and not self.is_running:
                    if event.key == pygame.K_1: self.start_search("BFS", bfs)
                    if event.key == pygame.K_2: self.start_search("DFS", dfs)
                    if event.key == pygame.K_3: self.start_search("UCS", ucs)
                    if event.key == pygame.K_r: self.render_ui()

if __name__ == "__main__":
    PathfinderPro().loop()