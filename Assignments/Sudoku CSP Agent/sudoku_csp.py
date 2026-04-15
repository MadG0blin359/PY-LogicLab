import sys
import copy

class SudokuCSP:
    def __init__(self, filename):
        self.board = self.read_board(filename)
        # Generate 81 variables as (row, col) tuples
        self.variables = [(r, c) for r in range(9) for c in range(9)]
        self.domains = self.get_initial_domains()
        self.neighbors = self.get_neighbors()
        
        # Tracking metrics for Deliverable 3
        self.backtrack_calls = 0
        self.backtrack_failures = 0

    def read_board(self, filename):
        """Reads a 9x9 Sudoku board from a text file."""
        board = []
        with open(filename, 'r') as f:
            for line in f:
                clean_line = line.strip()
                if clean_line:
                    board.append([int(char) for char in clean_line])
        return board

    def get_initial_domains(self):
        """Assigns domains: [1-9] for empty cells (0), or [val] for filled cells."""
        domains = {}
        for r in range(9):
            for c in range(9):
                if self.board[r][c] == 0:
                    domains[(r, c)] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
                else:
                    domains[(r, c)] = [self.board[r][c]]
        return domains

    def get_neighbors(self):
        """Builds a dictionary mapping each cell to its row, col, and box neighbors."""
        neighbors = {var: set() for var in self.variables}
        for r in range(9):
            for c in range(9):
                # 1. Row and Column constraints
                for i in range(9):
                    if i != c: neighbors[(r, c)].add((r, i))
                    if i != r: neighbors[(r, c)].add((i, c))
                
                # 2. 3x3 Subgrid constraints
                box_r, box_c = (r // 3) * 3, (c // 3) * 3
                for i in range(box_r, box_r + 3):
                    for j in range(box_c, box_c + 3):
                        if (i, j) != (r, c):
                            neighbors[(r, c)].add((i, j))
        return neighbors

    def ac3(self, domains):
        """Enforces arc consistency across the entire board."""
        # Initialize queue with all constraint arcs
        queue = [(xi, xj) for xi in self.variables for xj in self.neighbors[xi]]
        
        while queue:
            xi, xj = queue.pop(0)
            if self.revise(domains, xi, xj):
                # If a domain becomes empty, the puzzle is currently unsolvable
                if len(domains[xi]) == 0:
                    return False 
                # Re-evaluate neighbors since xi's domain was restricted
                for xk in self.neighbors[xi]:
                    if xk != xj:
                        queue.append((xk, xi))
        return True

    def revise(self, domains, xi, xj):
        """Removes values from xi's domain if they violate constraints with xj."""
        revised = False
        # If the neighbor xj is definitively solved (only 1 option left)
        if len(domains[xj]) == 1:
            val = domains[xj][0]
            # Remove that value from xi's domain to prevent a duplicate
            if val in domains[xi]:
                domains[xi].remove(val)
                revised = True
        return revised

    def is_complete(self, domains):
        """Checks if every variable has exactly one assigned value."""
        return all(len(domains[var]) == 1 for var in self.variables)

    def select_unassigned_variable(self, domains):
        """MRV Heuristic: Selects the unassigned cell with the fewest remaining choices."""
        unassigned = [var for var in self.variables if len(domains[var]) > 1]
        return min(unassigned, key=lambda var: len(domains[var]))

    def backtrack(self, domains):
        """Recursive backtracking search combined with AC-3."""
        self.backtrack_calls += 1

        if self.is_complete(domains):
            return domains

        var = self.select_unassigned_variable(domains)

        for value in domains[var]:
            # Create an independent copy of domains for this branch
            new_domains = copy.deepcopy(domains)
            new_domains[var] = [value]

            # Forward checking: Propagate the new assignment using AC-3
            if self.ac3(new_domains):
                result = self.backtrack(new_domains)
                if result is not False:
                    return result

        # Branch failed, log it and backtrack
        self.backtrack_failures += 1
        return False

    def solve(self):
        """Executes the CSP solving process."""
        # Step 1: Initial constraint propagation
        if not self.ac3(self.domains):
            print("Unsolvable puzzle detected during initial AC-3.")
            return False

        # Step 2: Begin Backtracking Search
        result = self.backtrack(self.domains)

        if result:
            self.print_board(result)
            print(f"\nBACKTRACK calls: {self.backtrack_calls}")
            print(f"BACKTRACK failures: {self.backtrack_failures}")
            return True
        else:
            print("Failed to solve.")
            return False

    def print_board(self, domains):
        """Outputs the solved board in the requested 9x9 format."""
        for r in range(9):
            row_str = ""
            for c in range(9):
                row_str += str(domains[(r, c)][0])
            print(row_str)

solver = SudokuCSP("easy.txt")
solver.solve()