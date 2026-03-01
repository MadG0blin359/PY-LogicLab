import pygame

# --- UI and Display Settings ---
WIDTH = 1200
HEIGHT = 800
GRID_WIDTH = 800
UI_WIDTH = WIDTH - GRID_WIDTH
FPS = 60

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
RED = (255, 0, 0)        # Visited
GREEN = (0, 255, 0)      # Path
BLUE = (0, 0, 255)       # Extra
YELLOW = (255, 255, 0)   # Frontier
ORANGE = (255, 165, 0)   # Start
PURPLE = (128, 0, 128)   # Goal
CYAN = (0, 255, 255)     # Agent

# --- Grid Settings ---
DEFAULT_ROWS = 40
DEFAULT_COLS = 40
MIN_ROWS = 10
MAX_ROWS = 100

# --- Pygame Initialization for Fonts ---
pygame.font.init()
FONT_SMALL = pygame.font.SysFont("Arial", 16)
FONT_MEDIUM = pygame.font.SysFont("Arial", 20)
FONT_LARGE = pygame.font.SysFont("Arial", 28)
