import numpy as np

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    return x * (1 - x)

layer_sizes = [2,3,2,1]
learn_rate = 0.1
epochs = 500000

np.random.seed(1)
weights = []

print("")
print("[betternn] Beginning training...")
print("")

for i in range(len(layer_sizes) - 1):
    w = 2 * np.random.random((layer_sizes[i], layer_sizes[i+1])) - 1
    weights.append(w)

for _ in range(epochs):
    layers = [x]
    for w in weights:
        layers.append(sigmoid(np.dot(layers[-1], w)))
    deltas = [(y - layers[-1]) * sigmoid_der(layers[-1])]
    for i in reversed(range(len(weights)-1)):
        delta = deltas[0].dot(weights[i+1].T) * sigmoid_der(layers[i+1])
        deltas.insert(0, delta)
    for i in range(len(weights)):
        weights[i] += layers[i].T.dot(deltas[i]) * learn_rate
    if _ % (epochs // 50) == 0:
        mse = np.mean((y -  layers[-1])**2)
        print(f"[betternn] Epoch {_}, MSE: {mse:.6f}")

final_output = layers[-1]
predicted_bits = (final_output > 0.5).astype(int)

print("")
print("[betternn] Predictions after training:")
print(final_output)
print("")
print("[betternn] Simplified Predictions:")
print(predicted_bits)
print("")
print("[betternn] Correct Bits:")
print(y)
print("")