# Pacman AI — Search & Multi-Agent Algorithms  
*A focused Artificial Intelligence project implementing core search and adversarial algorithms.*

### Search Algorithms (`search.py`)
- Depth-First Search (DFS)
-python3 pacman.py -p SearchAgent -a fn=dfs -l tinyMaze 
- Breadth-First Search (BFS)
-python3 pacman.py -p SearchAgent -a fn=bfs -l mediumMaze
- Uniform-Cost Search (UCS)
-python3 pacman.py -p SearchAgent -a fn=ucs -l mediumMaze 
- A* Search (with custom heuristics)
-python3 pacman.py -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic -l bigMaze -z 0.5

Each algorithm is implemented using graph-search principles and returns a valid path from the start state to the goal.

---

### Food Search & Heuristics (`searchAgents.py`)
- A* for the **FoodSearchProblem**
-python3 pacman.py -p AStarFoodSearchAgent -l trickySearch
- Custom **MST-based heuristic** improving efficiency in multi-food mazes  
  - Uses Minimum Spanning Tree estimation  
  - Admissible & consistent  
  - Significantly reduces node expansions  
  - Includes caching for performance optimization  

---

### Multi-Agent Algorithms (`multiAgents.py`)
Adversarial agents implemented:

- **Minimax Agent**
-python3 pacman.py -p MinimaxAgent -a depth=3 -l minimaxClassic
- **Alpha-Beta Pruning Agent**
-python3 pacman.py -p AlphaBetaAgent -a depth=3 -l trappedClassic
- **Expectimax Agent**
-python3 pacman.py -p ExpectimaxAgent -a depth=3 -l minimaxClassic

Includes a custom evaluation function considering:

- Food distance  
- Ghost distance  
- Capsules  
- Game score  
- Escape heuristics  

---
