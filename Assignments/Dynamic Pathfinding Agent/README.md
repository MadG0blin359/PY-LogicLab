# Dynamic Pathfinding Agent

This project implements a Dynamic Pathfinding Agent that navigates a grid-based environment. It features Informed Search Algorithms (A* and Greedy Best-First Search) and dynamically replans its path if new obstacles spawn while moving.

## Features
- **Algorithms Interface**: Toggle between A* Search and Greedy Best-First Search
- **Heuristics Selection**: Toggle between Manhattan Distance and Euclidean Distance
- **Dynamic Obstacle Spawning**: Real-time obstacle generation that triggers re-planning
- **Metrics Dashboard**: Live tracking of nodes expanded, path cost, and execution time
- **Interactive UI**: Draw walls, clear board, resize grid, and randomly generate mazes

## Requirements
- Python 3.8+
- Pygame

## Installation
1. Clone the repository
2. Install the necessary dependencies:
```cmd
pip install pygame
```

## How to Run
Navigate to the project directory and run:
```cmd
python main.py
```

## Controls
- **Left Click**: Draw wall obstacles
- **Right Click**: Erase wall obstacles
- Use the side control panel to select settings and run the search.
- When **Dynamic** mode is set to ON, obstacles have a chance of spawning while the agent is moving.
