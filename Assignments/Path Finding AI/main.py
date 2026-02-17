import pygame
import sys
from logic import bfs, dfs, ucs, dls, iddfs, bidirectional_search

# --- Theme & Dimensions ---
CLR_BG, CLR_SIDEBAR = (15, 15, 20), (25, 25, 35)
CLR_GRID, CLR_WALL = (35, 35, 45), (70, 75, 90)
CLR_START, CLR_TARGET = (0, 255, 127), (255, 46, 99)
CLR_FRONT, CLR_VISITED = (0, 217, 255), (60, 60, 80)
CLR_PATH, CLR_TEXT = (255, 211, 0), (230, 230, 240)

WIDTH, HEIGHT = 950, 750 
GRID_SIZE, ROWS, COLS = 600, 20, 20
CELL = GRID_SIZE // COLS
OFF_X, OFF_Y = 320, 50

class PathfinderPro:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("AI Path Finder")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.font_m = pygame.font.SysFont('Consolas', 26, bold=True)
        self.font_s = pygame.font.SysFont('Consolas', 16)
        self.font_n = pygame.font.SysFont('Consolas', 12) 
        self.grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        self.start, self.target = (2, 2), (ROWS-5, COLS-5)
        self.is_running = False 
        self.exploration_count = 0
        self._init_walls()

    def _init_walls(self):
        """Initializes the static walls in the center for testing."""
        for i in range(4, 16): self.grid[i][10] = -1

    def draw_cell(self, r, c, color, number=None, border=False):
        rect = (c * CELL + OFF_X, r * CELL + OFF_Y, CELL - 2, CELL - 2)
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        
        if number is not None:
            num_surface = self.font_n.render(str(number), True, CLR_TEXT)
            num_rect = num_surface.get_rect(center=(c * CELL + OFF_X + CELL//2, r * CELL + OFF_Y + CELL//2))
            self.screen.blit(num_surface, num_rect)
            
        if border: 
            pygame.draw.rect(self.screen, CLR_TEXT, rect, 1, border_radius=4)

    def gui_callback(self, pos, node_type):
        """Mandatory visualization of frontier and explored nodes."""
        if pos == self.start or pos == self.target: return
        
        num = None
        if node_type == "explored":
            self.exploration_count += 1
            num = self.exploration_count
            color = CLR_VISITED
        else:
            color = CLR_FRONT # FIXED: Changed from CL_FRONT to CLR_FRONT
            
        self.draw_cell(pos[0], pos[1], color, number=num)
        pygame.display.update()
        pygame.time.delay(10) # Animation flow

    def render_ui(self, status="READY"):
        self.screen.fill(CLR_BG)
        self.exploration_count = 0 
        pygame.draw.rect(self.screen, CLR_SIDEBAR, (0, 0, OFF_X - 30, HEIGHT))
        
        self.screen.blit(self.font_m.render("AI ENGINE", True, CLR_TEXT), (30, 40))
        controls = [
            ("[1] BFS (Shortest)", CLR_FRONT),
            ("[2] DFS (Deep)", CLR_TARGET),
            ("[3] UCS (Cost-Based)", CLR_PATH),
            ("[4] DLS (Limit=10)", CLR_TARGET),
            ("[5] IDDFS (Optimal DFS)", CLR_START),
            ("[6] Bidirectional", CLR_FRONT),
            ("[R] Reset Canvas", CLR_TEXT)
        ]
        for i, (t, c) in enumerate(controls):
            self.screen.blit(self.font_s.render(t, True, c), (30, 100 + (i * 45)))
        
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

    def start_search(self, name, algo, *args):
        if self.is_running: return
        self.is_running = True
        order_map = {} 
        
        def tracked_callback(pos, node_type):
            self.gui_callback(pos, node_type)
            if node_type == "explored":
                order_map[pos] = self.exploration_count

        self.render_ui(f"STATUS: {name}...")
        path = algo(self.start, self.target, ROWS, COLS, self.grid, tracked_callback, *args)
        
        if path:
            for p in path:
                if p != self.start and p != self.target:
                    num = order_map.get(p) 
                    self.draw_cell(p[0], p[1], CLR_PATH, number=num, border=True) # Maintain numbers
                    pygame.display.update()
                    pygame.time.delay(30)
            self.is_running = False 
        else:
            self.render_ui(f"STATUS: {name} FAILED")
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
                    if event.key == pygame.K_4: self.start_search("DLS", dls, 20) 
                    if event.key == pygame.K_5: self.start_search("IDDFS", iddfs)
                    if event.key == pygame.K_6: self.start_search("BIDIRECTIONAL", bidirectional_search)
                    if event.key == pygame.K_r: self.render_ui()

if __name__ == "__main__":
    PathfinderPro().loop()