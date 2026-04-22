from neuralnets import byte
import numpy as np
import itertools

epochs = 15000
learn_rate = 0.003 #003
checks = 100
board = [0,0,0,0,0,0,0,0,0]
current_player = -1
x = []
y = []
byte = byte([9,54,54,36,9]) #9,36,27,18,9: 15k, 499

def check_win(board):
    lines = [
        [0,1,2], [3,4,5], [6,7,8],
        [0,3,6], [1,4,7], [2,5,8],
        [0,4,8], [2,4,6]
    ]
    for line in lines:
        total = sum(board[i] for i in line)
        if total  == 3:
            return 1
        elif total == -3:
            return -1
    if 0 not in board:
        return 0
    return None

def train_byte():
    all_boards = itertools.product([-1, 0, 1], repeat=9)
    for board in all_boards:
        board = list(board)
        x_count = board.count(1)
        o_count = board.count(-1)
        if abs(x_count - o_count) > 1:
            continue
        if check_win(board) is not None:
            continue
        target = [0,0,0,0,0,0,0,0,0]
        for i in range(9):
            if board[i] != 0:
                continue
            board[i] = -1
            if check_win(board) == -1:
                target[i] = 1.0
            board[i] = 1
            if check_win(board) == 1:
                target[i] = 0.9
            board[i] = 0
        x.append(board.copy())
        y.append(target)
    x_arr = (np.array(x) + 1) / 2
    y_arr = np.array(y)
    byte.train(x_arr, y_arr, epochs, learn_rate, checks)

def print_board(board):
    symbols = {1:"[X]", -1:"[O]"}
    for i in range(3):
        row = []
        for j in range(3):
            k = j + i*3
            if board[k] == 0:
                row.append(" " + str(k) + " ")
            else:
                row.append(symbols[board[k]])
        print("|".join(row))
        if i < 2:
            print("---+---+---")

def human_move(board):
    move = -1
    while move not in range(9) or board[move] != 0:
        move = int(input(f"[byte_t3] Your Turn! (0-8): "))
    return move

def byte_move(board):
    output = byte.push(np.array(board).reshape(1,9))[0]
    masked = [o if board[i] == 0 else -1e9 for i, o in enumerate(output)]
    move = masked.index(max(masked))
    return move

train_byte()

while True:
    if current_player == 1:
        print("")
        print_board(board)
        print("")
        move = human_move(board)
    else:
        move = byte_move(board)
    board[move] = current_player if current_player == 1 else -1
    winner = check_win(board)
    if winner is not None:
        print("")
        print_board(board)
        print("")
        if winner == 1:
            print("[byte_t3] You Win! (^C To Exit Game)")
        elif winner == -1:
            print("[byte_t3] Byte Wins! (^C To Exit Game)")
        else:
            print("[byte_t3] Draw! (^C To Exit Game)")
        board = [0,0,0,0,0,0,0,0,0]
        current_player = -1
    else:
        current_player *= -1