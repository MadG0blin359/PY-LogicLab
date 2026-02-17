import collections
import heapq

def get_neighbors(node, rows, cols, grid):
    r, c = node
    # Strict 6-direction clockwise order: Up, Right, Bottom, Bottom-Right, Left, Top-Left [cite: 29-37]
    directions = [(-1, 0), (0, 1), (1, 0), (1, 1), (0, -1), (-1, -1)]
    neighbors = []
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != -1:
            neighbors.append((nr, nc))
    return neighbors

def reconstruct_path(visited, target):
    """Unified path reconstruction for all search strategies."""
    path = []
    current = target
    while current is not None:
        path.append(current)
        parent_info = visited[current]
        # Handle UCS (tuple) vs BFS/DFS (direct node) storage
        current = parent_info[1] if isinstance(parent_info, tuple) else parent_info
    return path[::-1]

def ucs(start, target, rows, cols, grid, update_gui):
    """Uniform-Cost Search: Expands lowest-cost node first using a Priority Queue."""
    pq = [(0, start)]
    visited = {start: (0, None)} # {node: (cost, parent)}
    
    while pq:
        cost, current = heapq.heappop(pq)
        
        if current == target:
            return reconstruct_path(visited, target)

        for neighbor in get_neighbors(current, rows, cols, grid):
            new_cost = cost + 1 # Assignment assumes uniform step cost
            if neighbor not in visited or new_cost < visited[neighbor][0]:
                visited[neighbor] = (new_cost, current)
                heapq.heappush(pq, (new_cost, neighbor))
                update_gui(neighbor, "frontier")

        if current != start:
            update_gui(current, "explored")
    return None

def bfs(start, target, rows, cols, grid, update_gui):
    """Breadth-First Search: Level-by-level exploration (FIFO)."""
    queue = collections.deque([start])
    visited = {start: None}
    
    while queue:
        current = queue.popleft()
        if current == target:
            return reconstruct_path(visited, target)

        for n in get_neighbors(current, rows, cols, grid):
            if n not in visited:
                visited[n] = current
                queue.append(n)
                update_gui(n, "frontier")

        if current != start:
            update_gui(current, "explored")
    return None

def dfs(start, target, rows, cols, grid, update_gui):
    """Depth-First Search: Branch-first exploration (LIFO)."""
    stack = [start]
    visited = {start: None}
    
    while stack:
        current = stack.pop()
        if current == target:
            return reconstruct_path(visited, target)

        # Reverse neighbors so the first clockwise direction is on top of stack
        for n in reversed(get_neighbors(current, rows, cols, grid)):
            if n not in visited:
                visited[n] = current
                stack.append(n)
                update_gui(n, "frontier")

        if current != start:
            update_gui(current, "explored")
    return None