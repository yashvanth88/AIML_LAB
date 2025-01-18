def print_state(state):
    for row in state:
        print(" ".join(map(str, row)))

def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

def move(state, di, dj):
    i, j = find_blank(state)
    ni, nj = i + di, j + dj
    if 0 <= ni < 3 and 0 <= nj < 3:
        state[i][j], state[ni][nj] = state[ni][nj], state[i][j]
        return state
    return None

def calculate_heuristic(state, goal):
    return sum(state[i][j] != goal[i][j] for i in range(3) for j in range(3))

def a_star(initial, goal):
    open_list = [(calculate_heuristic(initial, goal), 0, initial)]
    visited = set()

    while open_list:
        _, steps, current = min(open_list)
        open_list.remove((_, steps, current))
        visited.add(tuple(map(tuple, current)))

        print_state(current)
        print()

        if current == goal:
            print(f"Solution found in {steps} steps!")
            return

        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_state = move([row[:] for row in current], di, dj)
            if next_state and tuple(map(tuple, next_state)) not in visited:
                h = calculate_heuristic(next_state, goal)
                open_list.append((steps + h + 1, steps + 1, next_state))

    print("No solution found.")

print("Enter initial state (3x3 matrix):")
initial_state = [list(map(int, input().split())) for _ in range(3)]
print("Enter goal state (3x3 matrix):")
goal_state = [list(map(int, input().split())) for _ in range(3)]

a_star(initial_state, goal_state)