# AI Path Finder

An interactive pathfinding visualization tool built with Python and Pygame. This application demonstrates various AI search algorithms with real-time visual feedback.

![AI Path Finder](https://img.shields.io/badge/Python-Pygame-blue.svg)
![Version](https://img.shields.io/badge/Version-1.0-green.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Algorithms](#algorithms)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Controls](#controls)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)

## Overview

AI Path Finder is an educational tool that visualizes how different pathfinding algorithms work in real-time. It provides a 20x20 grid-based maze with a start point, target point, and walls. Users can select from multiple search algorithms and watch as the algorithm explores the grid to find a path.

## Features

- **Interactive GUI**: Real-time visualization of the search process
- **Multiple Algorithms**: 6 different pathfinding algorithms
- **Animation**: Watch the algorithm explore step by step
- **Path Highlighting**: Final path displayed in yellow with exploration order
- **Statistics**: Shows exploration count for each node

## Algorithms

### 1. BFS (Breadth-First Search)

- **Key**: `1`
- **Type**: Uninformed Search
- **Guarantee**: Shortest path in unweighted graphs
- **Time Complexity**: O(V + E)

### 2. DFS (Depth-First Search)

- **Key**: `2`
- **Type**: Uninformed Search
- **Behavior**: Explores as deep as possible before backtracking
- **Time Complexity**: O(V + E)

### 3. UCS (Uniform Cost Search)

- **Key**: `3`
- **Type**: Informed Search
- **Guarantee**: Shortest path with lowest cumulative cost
- **Time Complexity**: O(V + E)

### 4. DLS (Depth-Limited Search)

- **Key**: `4`
- **Type**: Uninformed Search
- **Depth Limit**: 20
- **Behavior**: DFS with a depth cutoff

### 5. IDDFS (Iterative Deepening DFS)

- **Key**: `5`
- **Type**: Uninformed Search
- **Guarantee**: Combines DFS space efficiency with BFS completeness
- **Behavior**: Repeatedly increases depth limit until target is found

### 6. Bidirectional Search

- **Key**: `6`
- **Type**: Uninformed Search
- **Behavior**: Searches simultaneously from start and target
- **Efficiency**: Significantly reduces search space

## Installation

### Prerequisites

- Python 3.7 or higher
- Pygame library

### Install Dependencies

```
bash
pip install pygame
```

### Clone/Download

1. Ensure you have the project files:
   - `main.py` - Main application file
   - `logic.py` - Algorithm implementations

## How to Run

```
bash
python main.py
```

Or simply run the `main.py` file in your IDE.

## Controls

| Key   | Action                                   |
| ----- | ---------------------------------------- |
| `1`   | Run BFS (Breadth-First Search)           |
| `2`   | Run DFS (Depth-First Search)             |
| `3`   | Run UCS (Uniform Cost Search)            |
| `4`   | Run DLS (Depth-Limited Search, limit=20) |
| `5`   | Run IDDFS (Iterative Deepening DFS)      |
| `6`   | Run Bidirectional Search                 |
| `R`   | Reset the visualization                  |
| `ESC` | Exit the application                     |

## Project Structure

```
Path Finding AI/
├── main.py          # Pygame GUI application
├── logic.py         # Pathfinding algorithm implementations
└── README.md       # This file
```

## Technical Details

### Grid Configuration

- **Grid Size**: 20 x 20 cells
- **Cell Size**: 30 pixels
- **Window Size**: 950 x 750 pixels
- **Offset**: (320, 50) for grid positioning

### Movement Directions

The algorithm supports 6-direction movement (including diagonals):

1. Up (-1, 0)
2. Right (0, 1)
3. Bottom (1, 0)
4. Bottom-Right (1, 1)
5. Left (0, -1)
6. Top-Left (-1, -1)

### Color Scheme

| Element     | Color (RGB)                   |
| ----------- | ----------------------------- |
| Background  | (15, 15, 20)                  |
| Sidebar     | (25, 25, 35)                  |
| Grid        | (35, 35, 45)                  |
| Walls       | (70, 75, 90)                  |
| Start Node  | (0, 255, 127) - Green         |
| Target Node | (255, 46, 99) - Pink          |
| Frontier    | (0, 217, 255) - Cyan          |
| Explored    | (60, 60, 80) - Dark Blue-Gray |
| Path        | (255, 211, 0) - Yellow        |

### Default Configuration

- **Start Position**: (2, 2)
- **Target Position**: (15, 15) - Row 15, Column 15
- **Walls**: Vertical wall in the center (columns 10, rows 4-15)

## Educational Use

This project is excellent for learning:

- How different search algorithms work
- The difference between informed and uninformed search
- Time and space complexity of algorithms
- Visual debugging of pathfinding logic

## License

This project is for educational purposes.

---

_Built with ❤️ using Python and Pygame_
