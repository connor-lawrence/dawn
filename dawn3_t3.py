from neuralnets import Dawn3
import numpy as np

dawn = Dawn3([9,128,128,64,32,9], "relu", "tanh")
dawn.init_parameters()

epochs = 15000
learn_rate = 0.0001
decay = 1
checks = epochs

board = [0]*9
current_player = np.random.choice([-1, 1])

x = []
y = []

def check_win(board):
    lines = [
        [0,1,2],[3,4,5],[6,7,8],
        [0,3,6],[1,4,7],[2,5,8],
        [0,4,8],[2,4,6]
    ]

    for a,b,c in lines:
        s = board[a] + board[b] + board[c]
        if s == 3: return 1
        if s == -3: return -1

    if 0 not in board:
        return 0

    return None

def train_dawn():
    x.clear()
    y.clear()

    seen = {}

    def generate(board, player):
        winner = check_win(board)
        if winner is not None:
            return [winner * player]

        key = tuple(board)

        if key in seen:
            return seen[key]

        if abs(board.count(1) - board.count(-1)) > 1:
            return []

        outcomes = []
        targets = [0]*9
        move_idx = 0

        for i in range(9):
            if board[i] != 0:
                continue

            board[i] = player
            result = generate(board, -player)
            board[i] = 0

            if not result:
                result = [0]

            outcomes.append(result)

        for i in range(9):
            if board[i] != 0:
                continue

            outcome = outcomes[move_idx]

            if any(o == player for o in outcome):
                targets[i] = 1.0
            elif any(o == -player for o in outcome):
                targets[i] = 0.9
            else:
                targets[i] = 0.1

            move_idx += 1

        x.append(board.copy())
        y.append(targets)

        seen[key] = targets
        return targets

    generate([0]*9, -1)

    x_arr = np.array(x).T
    y_arr = np.array(y).T

    dawn.teach("bp", x_arr, y_arr, learn_rate, epochs, decay, checks)

def print_board(board):
    symbols = {1:"[X]", -1:"[O]"}

    for i in range(3):
        row = []
        for j in range(3):
            k = i*3 + j
            row.append(symbols.get(board[k], f" {k} "))
        print("|".join(row))
        if i < 2:
            print("---+---+---")


def human_move(board):
    move = -1
    while move not in range(9) or board[move] != 0:
        move = int(input("Move (0-8): "))
    return move


def dawn_move(board):
    output = dawn.think(np.array(board).reshape(9,1)).flatten()

    masked = np.array([
        o if board[i] == 0 else -1e9
        for i, o in enumerate(output)
    ])

    return int(np.argmax(masked))

train_dawn()

while True:

    if current_player == 1:
        print("")
        print_board(board)
        print("")
        move = human_move(board)
    else:
        move = dawn_move(board)

    board[move] = current_player

    winner = check_win(board)

    if winner is not None:
        print("")
        print_board(board)
        print("")

        if winner == 1:
            print("[dawn] You Win!")
        elif winner == -1:
            print("[dawn] Dawn Wins!")
        else:
            print("[dawn] Draw!")

        board = [0]*9
        current_player = -1

    else:
        current_player *= -1