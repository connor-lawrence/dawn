import numpy as np
import itertools

board = [0,0,0,0,0,0,0,0,0]
layer_sizes = [9,32,32,9]
learn_rate = 0.01
epochs = 5000

def print_board(board):
    symbols = {0:" ", 1:"X", -1:"O"}
    print("")
    for i in range(3):
        row = " | ".join(symbols[board[j + i*3]] for j in range(3))
        print(" " + row + " ")
        if i < 2:
            print("---+---+---")
    print("")

def check_win(board):
    lines = [
        [0,1,2], [3,4,5], [6,7,8],
        [0,3,6], [1,4,7], [2,5,8],
        [0,4,8], [2,4,6]
    ]
    for line in lines:
        total = sum(board[i] for i in line)
        if total == 3:
            return 1
        elif total == -3:
            return -1
    if 0 not in board:
        return 0
    return None

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    return x * (1 - x)

def is_legal(b):
    x_count = b.count(1)
    o_count = b.count(-1)
    if abs(x_count - o_count) > 1:
        return False
    winner = check_win(b)
    if winner == 1 and x_count <= o_count:
        return False
    if winner == -1 and o_count < x_count:
        return False
    return True

def best_moves(b):
    x_count = b.count(1)
    o_count = b.count(-1)
    current = 1 if x_count == o_count else -1
    opponent = -current
    moves = []
    for i in range(9):
        if b[i] == 0:
            b[i] = current
            if check_win(b) == current:
                moves.append(i)
            b[i] = 0
    if moves:
        return moves
    for i in range(9):
        if b[i] == 0:
            b[i] = opponent
            if check_win(b) == opponent:
                moves.append(i)
            b[i] = 0
    if moves:
        return moves
    return [i for i in range(9) if b[i] == 0]

print("")
print("[node1] Beginning Board Check...")
print("")

all_boards = list(itertools.product([0,1,-1], repeat=9))
valid_boards = [list(b) for b in all_boards if is_legal(list(b)) and check_win(list(b)) is None]

print("[node1] Valid Boards: " + str(len(valid_boards)))
print("")

X = []
Y = []

for b in valid_boards:
    X.append(b)
    moves = best_moves(b)
    y = [0] * 9
    for m in moves:
        y[m] = 1
    Y.append(y)

X = np.array(X)
Y = np.array(Y)*0.8 + 0.1

np.random.seed(1)
weights = []

print("[node1] Beginning training...")
print("")

for i in range(len(layer_sizes) - 1):
    w = 2 * np.random.random((layer_sizes[i], layer_sizes[i + 1])) * 0.1
    weights.append(w)


for _ in range(epochs):
    layers = [X]
    for w in weights:
        layers.append(sigmoid(np.dot(layers[-1], w)))
    deltas = [(Y - layers[-1]) * sigmoid_der(layers[-1])]
    for i in reversed(range(len(weights) - 1)):
        delta = deltas[0].dot(weights[i + 1].T) * sigmoid_der(layers[i + 1])
        deltas.insert(0, delta)
    for i in range(len(weights)):
        weights[i] += layers[i].T.dot(deltas[i]) * learn_rate
    if _ % (epochs // 50) == 0: 
        mse = np.mean((Y - layers[-1])**2)
        print(f"[node1] Epoch {_}, MSE: {mse:.6f}")

def node_move(board):
    inp = np.array(board).reshape(1,-1)
    layer = [inp]
    for w in weights:
        layer.append(sigmoid(np.dot(layer[-1], w)))
    output = layer[-1][0]
    move = np.argmax(output)
    if board[move] != 0:
        move = [i for i in range(9) if board[i] == 0][0]
    return move

current_player = 1

while True:
    print_board(board)
    if current_player == 1:
        move = int(input(f"[node1] Your turn (0-8): "))
        while move not in range(9) or board[move] != 0:
            move = int(input(f"[node1] Your turn (0-8): "))
    else:
        move = node_move(board)
        print(f"[node1] Node plays: {move}")
    board[move] = current_player if current_player == 1 else -1
    winner = check_win(board)
    if winner is not None:
        print_board(board)
        if winner == 1:
            print("[node1] You win!")
        elif winner == -1:
            print("[node1] Node wins!")
        else:
            print("Draw!")
        break
    current_player = 2 if current_player == 1 else 1