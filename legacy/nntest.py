import numpy as np
print("")

# X is 2 bit adder input, Y is 2 bit adder output
x = np.array([[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],[1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1]])
y = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[0,1,1],[1,0,0],[1,0,1],[1,1,0]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

np.random.seed(1)

weights0 = 2 * np.random.random((4, 8)) - 1
weights1 = 2 * np.random.random((8, 8)) - 1
weights2 = 2 * np.random.random((8, 6)) - 1
weights3 = 2 * np.random.random((6, 3)) - 1

lr = 0.1

for epoch in range(50000):
    layer0 = x
    layer1 = sigmoid(np.dot(layer0, weights0))
    layer2 = sigmoid(np.dot(layer1, weights1))
    layer3 = sigmoid(np.dot(layer2, weights2))
    layer4 = sigmoid(np.dot(layer3, weights3))

    layer4_error = y - layer4
    layer4_delta = layer4_error * sigmoid_derivative(layer4)

    layer3_error = layer4_delta.dot(weights3.T)
    layer3_delta = layer3_error * sigmoid_derivative(layer3)

    layer2_error = layer3_delta.dot(weights2.T)
    layer2_delta = layer2_error * sigmoid_derivative(layer2)

    layer1_error = layer2_delta.dot(weights1.T)
    layer1_delta = layer1_error * sigmoid_derivative(layer1)

    weights3 += layer3.T.dot(layer4_delta) * lr
    weights2 += layer2.T.dot(layer3_delta) * lr
    weights1 += layer1.T.dot(layer2_delta) * lr
    weights0 += layer0.T.dot(layer1_delta) * lr

print("[nntest] Predictions after training:")
print(layer4)
print("")
predicted_bits = (layer4 > 0.5).astype(int)
print("[nntest] Simplified Predictions:")
print(predicted_bits)
print("")
print("[nntest] Correct Answers:")
print(y)
print("")