import collections
import time

def get_neighbors(node, rows, cols, grid):
    r, c = node
    # Strict Clockwise Order: Up, Right, Bottom, Bottom-Right, Left, Top-Left
    # Note: Top-Right and Bottom-Left are explicitly excluded.
    directions = [
        (-1, 0),  # 1. Up
        (0, 1),   # 2. Right
        (1, 0),   # 3. Bottom
        (1, 1),   # 4. Bottom-Right (Main Diagonal)
        (0, -1),  # 5. Left
        (-1, -1)  # 6. Top-Left (Main Diagonal)
    ]
    
    neighbors = []
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != -1:
            neighbors.append((nr, nc))
    return neighbors

def bfs(start, target, rows, cols, grid, update_gui):
    queue = collections.deque([start])
    visited = {start: None}
    
    while queue:
        current = queue.popleft()
        
        if current == target:
            return reconstruct_path(visited, target)

        for neighbor in get_neighbors(current, rows, cols, grid):
            if neighbor not in visited:
                visited[neighbor] = current
                queue.append(neighbor)
                update_gui(neighbor, "frontier")

        if current != start:
            update_gui(current, "explored")
            
    return None

def reconstruct_path(visited, target):
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = visited[current]
    return path[::-1]