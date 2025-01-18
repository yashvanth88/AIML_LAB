def initialize_board():
    return [[' '] * 3 for _ in range(3)]

def print_board(board):
    print("-------------")
    for row in board:
        print("|", " | ".join(row), "|")
        print("-------------")

def player_move(board, player):
    while True:
        try:
            row, col = map(int, input("Enter row and column (0-2): ").split())
            if 0 <= row < 3 and 0 <= col < 3 and board[row][col] == ' ':
                board[row][col] = player
                break
            print("Invalid move. Try again.")
        except:
            print("Invalid input. Use two numbers separated by space.")

def is_game_over(board):
    if check_winner(board, 'X'):
        print("AI Player (X) wins!")
        return True
    if check_winner(board, 'O'):
        print("Player O wins!")
        return True
    if is_board_full(board):
        print("It's a draw!")
        return True
    return False

def is_board_full(board):
    return all(cell != ' ' for row in board for cell in row)

def check_winner(board, player):
    return any(all(board[i][j] == player for j in range(3)) for i in range(3)) or \
           any(all(board[j][i] == player for j in range(3)) for i in range(3)) or \
           all(board[i][i] == player for i in range(3)) or \
           all(board[i][2 - i] == player for i in range(3))

def ai_move(board):
    row, col = find_best_move(board)
    board[row][col] = 'X'

def find_best_move(board):
    best_score, best_move = float('-inf'), (-1, -1)
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'X'
                score = minimax(board, 0, False)
                board[i][j] = ' '
                if score > best_score:
                    best_score, best_move = score, (i, j)
    return best_move

def minimax(board, depth, is_maximizing):
    if check_winner(board, 'X'):
        return 1
    if check_winner(board, 'O'):
        return -1
    if is_board_full(board):
        return 0

    scores = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'X' if is_maximizing else 'O'
                scores.append(minimax(board, depth + 1, not is_maximizing))
                board[i][j] = ' '
    return max(scores) if is_maximizing else min(scores)

def main():
    board = initialize_board()
    print_board(board)

    while True:
        player_move(board, 'O')
        print_board(board)
        if is_game_over(board):
            break

        ai_move(board)
        print_board(board)
        if is_game_over(board):
            break

if __name__ == "__main__":
    main()
