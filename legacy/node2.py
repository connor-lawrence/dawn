import numpy as np

# Setup

layer_sizes = [2,3,2,1]
learn_rate = 0.05
epochs = 100000

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

def sigmoid(x):
  x = np.clip(x, -500, 500)
  return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
  return x * (1 - x)

np.random.seed(1)
weights = []

# Training

print("")
print("[node2] Beginning training...")
print("")

for i in range(len(layer_sizes) - 1):
    limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
    w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
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
    if _ % (epochs // 10) == 0:
        mse = np.mean((y -  layers[-1])**2)
        print(f"[node2] Epoch {_}, MSE: {mse:.6f}")

# Finalizing

final_output = layers[-1]
predicted_bits = (final_output > 0.5).astype(int)

print("")
print("[node2] Predictions after training:")
print(final_output)
print("")
print("[node2] Simplified Predictions:")
print(predicted_bits)
print("")
print("[node2] Correct Bits:")
print(y)
print("")