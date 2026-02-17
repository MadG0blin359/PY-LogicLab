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
    f_q, b_q = collections.deque([start]), collections.deque([target])
    # Format: {node: (metadata, parent)}
    f_vis, b_vis = {start: (0, None)}, {target: (0, None)}
    
    while f_q and b_q:
        # Step Forward
        curr_f = f_q.popleft()
        for n in get_neighbors(curr_f, rows, cols, grid):
            if n not in f_vis:
                f_vis[n] = (0, curr_f)
                f_q.append(n)
                update_gui(n, "frontier")
                if n in b_vis: return _join_paths(f_vis, b_vis, n)
        update_gui(curr_f, "explored")

        # Step Backward
        curr_b = b_q.popleft()
        for n in get_neighbors(curr_b, rows, cols, grid):
            if n not in b_vis:
                b_vis[n] = (0, curr_b)
                b_q.append(n)
                update_gui(n, "frontier")
                if n in f_vis: return _join_paths(f_vis, b_vis, n)
        update_gui(curr_b, "explored")
    return None

def _join_paths(f_vis, b_vis, meeting_node):
    path_start = []
    curr = meeting_node
    # Traverse forward visited back to start
    while curr is not None:
        path_start.append(curr)
        curr = f_vis[curr][1] # Extract parent from tuple
    path_start.reverse()
    
    path_end = []
    # Traverse backward visited from meeting node back to target
    curr = b_vis[meeting_node][1] # Start from parent of meeting node in backward search
    while curr is not None:
        path_end.append(curr)
        curr = b_vis[curr][1]
        
    return path_start + path_end