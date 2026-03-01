import pygame
import math
import time
from queue import PriorityQueue

def h_manhattan(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def h_euclidean(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def reconstruct_path(current, draw):
    path_cost = 0
    path = []
    
    # Trace back
    while current.parent:
        path.append(current)
        current = current.parent
        path_cost += 1
        
    # Mark the path
    for node in reversed(path):
        if not node.is_start_node() and not node.is_goal_node():
            node.make_path()
            draw()
            pygame.time.delay(10) # Small delay for visualization
            
    return path_cost, list(reversed(path))

def astar_search(draw, grid_obj, start, goal, heuristic_func):
    start_time = time.perf_counter()
    nodes_expanded = 0
    
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    
    # Initialize all nodes to infinity except start
    for row in grid_obj.grid:
        for node in row:
            node.g = float('inf')
            node.f = float('inf')
            node.parent = None
            
    start.g = 0
    start.f = heuristic_func(start.get_pos(), goal.get_pos())
    
    open_set_hash = {start}
    
    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        current = open_set.get()[2]
        open_set_hash.remove(current)
        
        if current == goal:
            # Reconstruct path
            cost, path_nodes = reconstruct_path(goal, draw)
            goal.make_goal()
            exec_time = (time.perf_counter() - start_time) * 1000
            return True, nodes_expanded, cost, exec_time, path_nodes

        nodes_expanded += 1
        for neighbor in current.neighbors:
            temp_g_score = current.g + 1
            
            if temp_g_score < neighbor.g:
                neighbor.parent = current
                neighbor.g = temp_g_score
                neighbor.h = heuristic_func(neighbor.get_pos(), goal.get_pos())
                neighbor.f = neighbor.g + neighbor.h
                
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((neighbor.f, count, neighbor))
                    open_set_hash.add(neighbor)
                    if not neighbor.is_goal_node():
                        neighbor.make_open()

        draw()
        
        if current != start:
            current.make_closed()
            
    exec_time = (time.perf_counter() - start_time) * 1000
    return False, nodes_expanded, 0, exec_time, []

def gbfs_search(draw, grid_obj, start, goal, heuristic_func):
    start_time = time.perf_counter()
    nodes_expanded = 0
    
    count = 0
    open_set = PriorityQueue()
    # Priority is just h(n)
    h_start = heuristic_func(start.get_pos(), goal.get_pos())
    open_set.put((h_start, count, start))
    
    for row in grid_obj.grid:
        for node in row:
            node.parent = None
            node.h = 0
            
    start.h = h_start
    open_set_hash = {start}
    closed_set = set()
    
    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        current = open_set.get()[2]
        open_set_hash.remove(current)
        closed_set.add(current)
        
        if current == goal:
            # Reconstruct path
            cost, path_nodes = reconstruct_path(goal, draw)
            goal.make_goal()
            exec_time = (time.perf_counter() - start_time) * 1000
            return True, nodes_expanded, cost, exec_time, path_nodes

        nodes_expanded += 1
        for neighbor in current.neighbors:
            if neighbor in closed_set:
                continue
                
            if neighbor not in open_set_hash:
                neighbor.parent = current
                neighbor.h = heuristic_func(neighbor.get_pos(), goal.get_pos())
                
                count += 1
                open_set.put((neighbor.h, count, neighbor))
                open_set_hash.add(neighbor)
                
                if not neighbor.is_goal_node():
                    neighbor.make_open()

        draw()
        
        if current != start:
            current.make_closed()
            
    exec_time = (time.perf_counter() - start_time) * 1000
    return False, nodes_expanded, 0, exec_time, []
