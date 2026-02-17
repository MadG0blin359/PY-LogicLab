import collections
import heapq

def get_neighbors(node, rows, cols, grid):
    r, c = node
    # Strict 6-direction clockwise: Up, Right, Bottom, Bottom-Right, Left, Top-Left
    directions = [(-1, 0), (0, 1), (1, 0), (1, 1), (0, -1), (-1, -1)]
    neighbors = []
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != -1:
            neighbors.append((nr, nc))
    return neighbors

def reconstruct_path(visited, target):
    """Safe path reconstruction for all search strategies."""
    path = []
    current = target
    if target not in visited: return None
    
    while current is not None:
        path.append(current)
        # Always extract the parent from the second index
        current = visited[current][1] 
    return path[::-1]

def bfs(start, target, rows, cols, grid, update_gui):
    queue = collections.deque([start])
    visited = {start: (0, None)} # Format: (dist, parent)
    
    while queue:
        current = queue.popleft()
        if current == target: return reconstruct_path(visited, target)

        for n in get_neighbors(current, rows, cols, grid):
            if n not in visited:
                visited[n] = (0, current)
                queue.append(n)
                update_gui(n, "frontier")

        if current != start: update_gui(current, "explored")
    return None

def dfs(start, target, rows, cols, grid, update_gui):
    stack = [start]
    visited = {start: (0, None)}
    
    while stack:
        current = stack.pop()
        if current == target: return reconstruct_path(visited, target)

        for n in reversed(get_neighbors(current, rows, cols, grid)):
            if n not in visited:
                visited[n] = (0, current)
                stack.append(n)
                update_gui(n, "frontier")

        if current != start: update_gui(current, "explored")
    return None

def ucs(start, target, rows, cols, grid, update_gui):
    pq = [(0, start)]
    visited = {start: (0, None)}
    
    while pq:
        cost, current = heapq.heappop(pq)
        if current == target: return reconstruct_path(visited, target)

        for n in get_neighbors(current, rows, cols, grid):
            new_cost = cost + 1
            if n not in visited or new_cost < visited[n][0]:
                visited[n] = (new_cost, current)
                heapq.heappush(pq, (new_cost, n))
                update_gui(n, "frontier")

        if current != start: update_gui(current, "explored")
    return None

def dls(start, target, rows, cols, grid, update_gui, limit):
    # Stack stores: (current_node, current_depth)
    stack = [(start, 0)]
    visited = {start: (0, None)} # {node: (depth, parent)}
    
    while stack:
        current, depth = stack.pop()
        
        if current == target:
            return reconstruct_path(visited, target)
            
        if depth < limit:
            # Reversing neighbors to maintain strict clockwise priority in a stack
            for neighbor in reversed(get_neighbors(current, rows, cols, grid)):
                # If not visited OR we found a shorter path to this node at a shallower depth
                if neighbor not in visited or depth + 1 < visited[neighbor][0]:
                    visited[neighbor] = (depth + 1, current)
                    stack.append((neighbor, depth + 1))
                    update_gui(neighbor, "frontier")

        if current != start:
            update_gui(current, "explored")
            
    return None

def iddfs(start, target, rows, cols, grid, update_gui):
    for depth in range(rows * cols):
        result = dls(start, target, rows, cols, grid, update_gui, depth)
        
        if result:
            return result
            
    return None

def bidirectional_search(start, target, rows, cols, grid, update_gui):
    # Forward search structures
    f_queue = collections.deque([start])
    f_visited = {start: None}
    
    # Backward search structures
    b_queue = collections.deque([target])
    b_visited = {target: None}
    
    while f_queue and b_queue:
        # Expand Forward Frontier
        path = expand_frontier(f_queue, f_visited, b_visited, rows, cols, grid, update_gui, "frontier")
        if path: return path
        
        # Expand Backward Frontier
        path = expand_frontier(b_queue, b_visited, f_visited, rows, cols, grid, update_gui, "frontier")
        if path: return path[::-1] # Reverse because we found it from the back

    return None

def expand_frontier(queue, visited, other_visited, rows, cols, grid, update_gui, node_type):
    current = queue.popleft()
    
    for neighbor in get_neighbors(current, rows, cols, grid):
        if neighbor not in visited:
            visited[neighbor] = current
            queue.append(neighbor)
            update_gui(neighbor, node_type)
            
            # If the frontiers meet, reconstruct the full path
            if neighbor in other_visited:
                return join_paths(visited, other_visited, neighbor)
    
    update_gui(current, "explored")
    return None

def join_paths(f_visited, b_visited, meeting_node):
    # Path from start to meeting node
    path_start = []
    curr = meeting_node
    while curr is not None:
        path_start.append(curr)
        curr = f_visited[curr]
    path_start.reverse()
    
    # Path from meeting node to target
    path_end = []
    curr = b_visited[meeting_node]
    while curr is not None:
        path_end.append(curr)
        curr = b_visited[curr]
        
    return path_start + path_end