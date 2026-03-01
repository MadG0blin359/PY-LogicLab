import pygame
from settings import *

class Button:
    def __init__(self, x, y, width, height, text, font=FONT_MEDIUM, bg_color=LIGHT_GRAY, hover_color=GRAY, text_color=BLACK):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.is_hovered = False

    def draw(self, win):
        color = self.hover_color if self.is_hovered else self.bg_color
        pygame.draw.rect(win, color, self.rect, border_radius=5)
        pygame.draw.rect(win, BLACK, self.rect, 2, border_radius=5)
        
        text_surface = self.font.render(self.text, True, self.text_color)
        win.blit(text_surface, (self.rect.x + (self.rect.width - text_surface.get_width()) // 2, 
                               self.rect.y + (self.rect.height - text_surface.get_height()) // 2))

    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered

    def is_clicked(self, pos, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(pos):
                return True
        return False

class ToggleButton(Button):
    def __init__(self, x, y, width, height, text_options, current_index=0, *args, **kwargs):
        super().__init__(x, y, width, height, text_options[current_index], *args, **kwargs)
        self.text_options = text_options
        self.current_index = current_index

    def toggle(self):
        self.current_index = (self.current_index + 1) % len(self.text_options)
        self.text = self.text_options[self.current_index]

    def get_value(self):
        return self.text

class UI:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.buttons = {}
        
        # Setup specific buttons
        btn_width = width - 40
        btn_x = x + 20
        start_y = y + 20
        spacing = 50

        # Run/Clear
        self.buttons['run'] = Button(btn_x, start_y, btn_width, 40, "Run Search")
        self.buttons['clear_path'] = Button(btn_x, start_y + spacing, btn_width, 40, "Clear Path/State")
        self.buttons['clear_board'] = Button(btn_x, start_y + 2*spacing, btn_width, 40, "Clear All")

        # Configuration Toggles
        self.buttons['algo'] = ToggleButton(btn_x, start_y + 4*spacing, btn_width, 40, ["A* Search", "GBFS"])
        self.buttons['heuristic'] = ToggleButton(btn_x, start_y + 5*spacing, btn_width, 40, ["Manhattan", "Euclidean"])
        self.buttons['dynamic'] = ToggleButton(btn_x, start_y + 6*spacing, btn_width, 40, ["Dynamic: OFF", "Dynamic: ON"])

        # Map generation
        self.buttons['gen_maze'] = Button(btn_x, start_y + 8*spacing, btn_width, 40, "Generate Random Maze")
        
        # Grid Size adjustments
        self.buttons['size_up'] = Button(btn_x, start_y + 9*spacing, btn_width//2 - 5, 40, "Size +")
        self.buttons['size_down'] = Button(btn_x + btn_width//2 + 5, start_y + 9*spacing, btn_width//2 - 5, 40, "Size -")

        # Metrics data
        self.metrics = {
            "nodes_visited": 0,
            "path_cost": 0,
            "exec_time_ms": 0.0,
            "status": "Ready",
            "grid_size": f"{DEFAULT_ROWS}x{DEFAULT_COLS}"
        }

    def update_metrics(self, visited, cost, time_ms, status):
        self.metrics["nodes_visited"] = visited
        self.metrics["path_cost"] = cost
        self.metrics["exec_time_ms"] = time_ms
        self.metrics["status"] = status

    def draw(self, win):
        # Draw Panel Background
        pygame.draw.rect(win, LIGHT_GRAY, self.rect)
        pygame.draw.rect(win, BLACK, self.rect, 2)
        
        # Draw Buttons
        for btn in self.buttons.values():
            btn.draw(win)

        # Draw Metrics
        start_y = self.buttons['size_down'].rect.bottom + 30
        x_pos = self.rect.x + 20
        
        texts = [
            f"Status: {self.metrics['status']}",
            f"Grid Size: {self.metrics['grid_size']}",
            f"Nodes Expanded: {self.metrics['nodes_visited']}",
            f"Path Cost: {self.metrics['path_cost']}",
            f"Time (ms): {self.metrics['exec_time_ms']:.2f}"
        ]
        
        for i, text in enumerate(texts):
            text_surf = FONT_MEDIUM.render(text, True, BLACK)
            win.blit(text_surf, (x_pos, start_y + i * 30))

    def handle_event(self, pos, event):
        clicked_btn = None
        for key, btn in self.buttons.items():
            btn.check_hover(pos)
            if btn.is_clicked(pos, event):
                clicked_btn = key
                if isinstance(btn, ToggleButton):
                    btn.toggle()
        return clicked_btn
