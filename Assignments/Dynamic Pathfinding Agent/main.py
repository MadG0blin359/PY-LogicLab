import pygame
import time
from settings import *
from grid import Grid
from ui import UI
from algorithms import astar_search, gbfs_search, h_manhattan, h_euclidean
from agent import Agent

def main():
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Dynamic Pathfinding Agent")

    grid = Grid(DEFAULT_ROWS, DEFAULT_COLS, GRID_WIDTH, HEIGHT)
    ui = UI(GRID_WIDTH, 0, UI_WIDTH, HEIGHT)

    run = True
    agent = None
    path_found = False
    
    clock = pygame.time.Clock()
    
    # Track drawing state to allow continuous drawing
    drawing_barrier = False
    erasing = False

    while run:
        clock.tick(FPS)
        grid.draw(win)
        
        # If agent is active, draw it and handle movement
        if agent:
            # Step the agent slowly
            pygame.time.delay(100)
            
            # Dynamic Obstacle Spawning
            dynamic_mode_on = "ON" in ui.buttons['dynamic'].get_value()
            if dynamic_mode_on:
                agent.spawn_dynamic_obstacle(grid, spawn_prob=0.03)

            moved, finished = agent.move_step()
            
            if finished:
                ui.metrics["status"] = "Agent Reached Target!"
                agent = None
            elif not moved:
                # Agent is blocked! Re-plan
                ui.metrics["status"] = "Path Blocked! Replanning..."
                ui.draw(win)
                pygame.display.update()
                
                # Reset grid path markings but keep barriers
                grid.reset_all_algo_states()
                
                # New Start is agent's current position
                # Need to run search from agent's current node
                start_node = agent.current_node
                goal_node = grid.goal_node
                start_node.make_start()
                
                # Run search again
                algo_name = ui.buttons['algo'].get_value()
                heuristic_name = ui.buttons['heuristic'].get_value()
                h_func = h_manhattan if heuristic_name == "Manhattan" else h_euclidean
                
                draw_func = lambda: (grid.draw(win), ui.draw(win), agent.draw(win) if agent else None, pygame.display.update())

                if "A*" in algo_name:
                    success, exp, cost, exec_time, path = astar_search(draw_func, grid, start_node, goal_node, h_func)
                else:
                    success, exp, cost, exec_time, path = gbfs_search(draw_func, grid, start_node, goal_node, h_func)

                if success:
                    path_found = True
                    agent.path = path
                    agent.path_index = 0
                    if len(agent.path) > 0 and agent.path[0] == start_node:
                        agent.path_index = 1
                    
                    ui.update_metrics(exp, cost, exec_time, "Agent Moving")
                else:
                    ui.update_metrics(exp, 0, exec_time, "No Path Found")
                    agent = None # Stuck
            
            if agent:
                agent.draw(win)
                
        ui.draw(win)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            
            pos = pygame.mouse.get_pos()
            
            # UI Interaction
            if pos[0] > GRID_WIDTH:
                clicked_btn = ui.handle_event(pos, event)
                if clicked_btn:
                    if clicked_btn == 'clear_path':
                        grid.reset_all_algo_states()
                        agent = None
                        path_found = False
                        ui.metrics["status"] = "Path Cleared"
                    
                    elif clicked_btn == 'clear_board':
                        grid.build_grid()
                        agent = None
                        path_found = False
                        ui.metrics["status"] = "Grid Cleared"
                        
                    elif clicked_btn == 'gen_maze':
                        grid.generate_random_maze(density=0.3)
                        agent = None
                        path_found = False
                        ui.metrics["status"] = "Maze Generated"
                        
                    elif clicked_btn == 'size_up':
                        if grid.rows < MAX_ROWS:
                            new_size = grid.rows + 10
                            grid = Grid(new_size, new_size, GRID_WIDTH, HEIGHT)
                            agent = None
                            path_found = False
                            ui.metrics["grid_size"] = f"{new_size}x{new_size}"
                            ui.metrics["status"] = "Size Increased"

                    elif clicked_btn == 'size_down':
                        if grid.rows > MIN_ROWS:
                            new_size = grid.rows - 10
                            grid = Grid(new_size, new_size, GRID_WIDTH, HEIGHT)
                            agent = None
                            path_found = False
                            ui.metrics["grid_size"] = f"{new_size}x{new_size}"
                            ui.metrics["status"] = "Size Decreased"

                    elif clicked_btn == 'run':
                        # Stop agent if running
                        agent = None
                        
                        grid.reset_all_algo_states()
                        
                        algo_name = ui.buttons['algo'].get_value()
                        heuristic_name = ui.buttons['heuristic'].get_value()
                        h_func = h_manhattan if heuristic_name == "Manhattan" else h_euclidean
                        
                        start_node = grid.start_node
                        goal_node = grid.goal_node
                        
                        ui.metrics["status"] = f"Running {algo_name}"
                        ui.draw(win)
                        pygame.display.update()
                        
                        draw_func = lambda: (grid.draw(win), ui.draw(win), pygame.display.update())

                        if "A*" in algo_name:
                            success, exp, cost, exec_time, path = astar_search(draw_func, grid, start_node, goal_node, h_func)
                        else:
                            success, exp, cost, exec_time, path = gbfs_search(draw_func, grid, start_node, goal_node, h_func)

                        if success:
                            path_found = True
                            dynamic_mode_on = "ON" in ui.buttons['dynamic'].get_value()
                            status_msg = "Path Found"
                            
                            ui.update_metrics(exp, cost, exec_time, status_msg)
                            
                            # Start agent moving immediately if returning true
                            agent = Agent(start_node, path)
                            ui.metrics["status"] = "Agent Moving"
                        else:
                            ui.update_metrics(exp, 0, exec_time, "No Path Found")

            # Grid Interaction
            elif pos[0] < GRID_WIDTH:
                row_col = grid.get_clicked_pos(pos)
                if row_col:
                    row, col = row_col
                    node = grid.get_node(row, col)

                    # Toggle drawing walls
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1: # Left Click
                            if node != grid.start_node and node != grid.goal_node:
                                drawing_barrier = True
                                node.make_barrier()
                        elif event.button == 3: # Right Click
                            if node != grid.start_node and node != grid.goal_node:
                                erasing = True
                                node.reset()

                    elif event.type == pygame.MOUSEBUTTONUP:
                        drawing_barrier = False
                        erasing = False
                        grid.update_all_neighbors()

                    elif event.type == pygame.MOUSEMOTION:
                        if drawing_barrier:
                            if node != grid.start_node and node != grid.goal_node:
                                node.make_barrier()
                        elif erasing:
                            if node != grid.start_node and node != grid.goal_node:
                                node.reset()

    pygame.quit()

if __name__ == "__main__":
    main()
