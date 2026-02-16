import collections
import time

def get_neighbors(node, rows, cols, grid):
    """
    Returns neighbors in the mandatory clockwise order:
    Up, Top-Right, Right, Bottom-Right, Bottom, Bottom-Left, Left, Top-Left
    """
    r, c = node
    # Mandatory Clockwise Directions 
    directions = [
        (-1, 0),  # Up
        (-1, 1),  # Top-Right
        (0, 1),   # Right
        (1, 1),   # Bottom-Right
        (1, 0),   # Bottom
        (1, -1),  # Bottom-Left
        (0, -1),  # Left
        (-1, -1)  # Top-Left
    ]
    
    neighbors = []
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        # Boundary and Wall Check [cite: 1131]
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != -1:
            neighbors.append((nr, nc))
    return neighbors

def bfs(start, target, rows, cols, grid, draw_callback, spawn_callback):
    """
    Breadth-First Search: Level-by-level exploration (FIFO Queue).
    """
    queue = collections.deque([start])
    # visited stores: {current_node: parent_node} to reconstruct path later
    visited = {start: None}
    
    while queue:
        current = queue.popleft()
        
        # 1. Check if we reached the target
        if current == target:
            return reconstruct_path(visited, target)

        # 2. Handle Dynamic Hurdles (Spawn random obstacles)
        spawn_callback() # [cite: 1156, 1157]

        # 3. Explore Neighbors in mandatory Clockwise order
        for neighbor in get_neighbors(current, rows, cols, grid):
            if neighbor not in visited:
                # Requirement: Re-plan if a neighbor becomes a dynamic obstacle
                if grid[neighbor[0]][neighbor[1]] == -1: 
                    continue # Ignore it, treat it as a wall [cite: 1158]
                
                visited[neighbor] = current
                queue.append(neighbor)
                
                # Visual Distinction: Mark Frontier Node (Blue)
                # Small delay so user can watch the "flood" [cite: 1167]
                draw_callback(neighbor, (0, 0, 255)) 
                time.sleep(0.01)

        # Visual Distinction: Mark Explored Node (Gray)
        if current != start:
            draw_callback(current, (128, 128, 128)) 
            
    return None # No path found

def reconstruct_path(visited, target):
    """Backtracks from target to start to find the final path."""
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = visited[current]
    return path[::-1] # Reverse it