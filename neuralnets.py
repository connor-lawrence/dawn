import numpy as np                                                                                                                                                          #
class Dawn3:
    def __init__(self, net, hidden_type, output_type): ##### Initialize Dawn3 #####
        np.random.seed(1) # Set random seed
        self.net_dimensions = net; self.layer_count = len(net) - 1 # Bring in network dimensions and saves number of layers
        self.weights = []; self.biases = [] # Make lists for weights and biases
        self.activated_layer_outputs = []; self.raw_layer_outputs = [] # Make lists for raw and activated layer outputs for training
        self.hidden_type = hidden_type; self.output_type = output_type # Bring in activation functions for hidden and output layers
        self.hidden_function = {"sigmoid": self.sigmoid, "tanh": self.tanh, "relu": self.relu}.get(hidden_type, self.sigmoid) # Hidden layer activation function
        self.hidden_derivative = {"sigmoid": self.sigmoid_der, "tanh": self.tanh_der, "relu": self.relu_der}.get(hidden_type, self.sigmoid_der) # Hidden layer derivative
        self.output_function = {"sigmoid": self.sigmoid, "tanh": self.tanh, "relu": self.relu}.get(output_type, self.sigmoid) # Output layer activation function
        self.output_derivative = {"sigmoid": self.sigmoid_der, "tanh": self.tanh_der, "relu": self.relu_der}.get(output_type, self.sigmoid_der) # Output layer derivative
        self.layer_outputs = {"sigmoid": self.activated_layer_outputs, "tanh": self.activated_layer_outputs, "relu": self.raw_layer_outputs}.get(hidden_type) # Layer outputs #CHANGE: hidden
    def init_parameters(self, nu="n"): ##### Initialize Parameters #####
        self.weights = []; self.biases = [] # Bring in lists for weights and biases
        for i in range(self.layer_count): # For each layer:
            nodes_in = self.net_dimensions[i]; nodes_out = self.net_dimensions[i+1] # Store the number of neurons in this and next layer (to size arrays)
            if i < self.layer_count - 1: act = self.hidden_type # If in a hidden layer, use the hidden layer activation function refrence
            else: act = self.output_type # If in an output layer, use the output layer activation function refrence
            if act in ["sigmoid", "tanh"]: # If Sigmoid or TanH (Xavier):
                if nu == "n": weight_matrix = np.random.randn(nodes_out, nodes_in) * np.sqrt(2 / (nodes_in + nodes_out)) # Xavier Normal
                else: weight_matrix = np.random.uniform((-1 * (np.sqrt(6 / (nodes_in + nodes_out)))), np.sqrt(6 / (nodes_in + nodes_out)), (nodes_out, nodes_in)) # Xvr Unfrm
            elif act == "relu": # If ReLU (He):
                if nu == "n": weight_matrix = np.random.randn(nodes_out, nodes_in) * np.sqrt(2 / nodes_in) # He Normal
                else: weight_matrix = np.random.uniform((-1 * (np.sqrt(6 / nodes_in))), np.sqrt(6 / nodes_in), (nodes_out, nodes_in)) # He Uniform
            else: weight_matrix = np.random.randn(nodes_out, nodes_in) * 0.1 # Use small random numbers as default
            self.weights.append(weight_matrix); self.biases.append(np.zeros((nodes_out, 1))) # Activate weights, set biases to 0
    def think(self, inputs): ##### Forward pass #####
        self.activated_layer_outputs = [inputs]; self.raw_layer_outputs = [inputs] # Bring in lists for raw and activated layer outputs
        current_input = inputs # Start with the input layer
        for i in range(self.layer_count): # For each layer:
            current_weights = self.weights[i]; current_biases = self.biases[i] # Bring in the current layer's weights and biases
            raw_output = current_weights @ current_input + current_biases; self.raw_layer_outputs.append(raw_output) # Compute raw layer output and store
            if i == self.layer_count - 1: layer_output = self.output_function(raw_output) # If in the output layer, compute it's activated output and store
            else: layer_output = self.hidden_function(raw_output) # If in a hidden layer, compute it's activated output and store
            self.activated_layer_outputs.append(layer_output); current_input = layer_output # Bring the output from the current layer into the next
        if self.output_type in ["sigmoid", "tanh"]: self.layer_outputs = self.activated_layer_outputs # Refresh layer outputs for sigmoid and TanH (activated)
        elif self.output_type == "relu": self.layer_outputs = self.raw_layer_outputs # Refresh layer outputs for ReLU (raw)
        return current_input # Return output
    def teach(self, method, inputs, signals, lr, epochs=1, decay=1, logs=1): ##### Train Network #####
        inputs = np.asarray(inputs); signals = np.asarray(signals); deltas = [None] * self.layer_count # Convert inputs and targets into NumPy arrays and initialize deltas
        interval = max(1, epochs // logs) if logs != 0 else None; epoch_digits = len(str(epochs)); current_lr = lr # Set log interval and starting learn rate
        for epoch in range(epochs): # For each epoch:
            self.think(inputs) # Forward pass
            if method == "bp": deltas[-1] = (self.activated_layer_outputs[-1] - signals) * self.output_derivative(self.activated_layer_outputs[-1]) # Backprop output layer deltas #CHANGE: activated
            elif method == "rl": deltas[-1] = (signals - self.activated_layer_outputs[-1]) * self.output_derivative(self.activated_layer_outputs[-1]) # Reinforcement output layer deltas #CHANGE: activated
            for i in reversed(range(self.layer_count - 1)): deltas[i] = (self.weights[i+1].T @ deltas[i+1]) * self.hidden_derivative(self.layer_outputs[i+1]) # Hdn lyr dltas
            for i in range(self.layer_count): # For each layer:
                previous_output = self.activated_layer_outputs[i] # Set the previous output correctly
                self.weights[i] -= current_lr * (deltas[i] @ previous_output.T); self.biases[i] -= current_lr * np.sum(deltas[i], axis=1, keepdims=True) # Update wts and bs #CHANGE: sum
            current_lr *= decay # Decay learn rate
            if logs != 0 and epoch % interval == 0: # If a log is expected:
                mse = np.mean((self.activated_layer_outputs[-1] - signals) ** 2) # Compute MSE
                print(f"[Dawn3] {int(100*(epoch/epochs)):02}% E:{epoch:0{epoch_digits}d} LR:{current_lr:.4f} MSE:{mse:.6f}") # Log data
        return np.mean((self.activated_layer_outputs[-1] - signals) ** 2) # Returns MSE
    def sigmoid(self, x): return 1 / (1 + np.exp(-x)) # Sigmoid
    def sigmoid_der(self, y): return y * (1 - y) # Sigmoid derivative
    def tanh(self, x): return np.tanh(x) # TanH
    def tanh_der(self, y): return 1 - (y ** 2) # TanH derivative
    def relu(self, x): return np.maximum(0.01 * x, x) # ReLU
    def relu_der(self, x): return (x > 0).astype(float) + 0.01 * (x <= 0) # ReLU derivative


class Dawn4:
    def __init__(self, net, hidden_type, output_type): ##### Initialize Dawn3 #####
        np.random.seed(1) # Set random seed
        self.net_dimensions = net; self.layer_count = len(net) - 1 # Bring in network dimensions and saves number of layers
        self.weights = []; self.biases = [] # Make lists for weights and biases
        self.activated_layer_outputs = []; self.raw_layer_outputs = [] # Make lists for raw and activated layer outputs for training
        self.hidden_type = hidden_type; self.output_type = output_type # Bring in activation functions for hidden and output layers
        self.hidden_function = {"sigmoid": self.sigmoid, "tanh": self.tanh, "relu": self.relu}.get(hidden_type, self.sigmoid) # Hidden layer activation function
        self.hidden_derivative = {"sigmoid": self.sigmoid_der, "tanh": self.tanh_der, "relu": self.relu_der}.get(hidden_type, self.sigmoid_der) # Hidden layer derivative
        self.output_function = {"sigmoid": self.sigmoid, "tanh": self.tanh, "relu": self.relu}.get(output_type, self.sigmoid) # Output layer activation function
        self.output_derivative = {"sigmoid": self.sigmoid_der, "tanh": self.tanh_der, "relu": self.relu_der}.get(output_type, self.sigmoid_der) # Output layer derivative
        self.layer_outputs = {"sigmoid": self.activated_layer_outputs, "tanh": self.activated_layer_outputs, "relu": self.raw_layer_outputs}.get(output_type) # Layer outputs
    def init_parameters(self, nu="n"): ##### Initialize Parameters #####
        self.weights = []; self.biases = [] # Bring in lists for weights and biases
        for i in range(self.layer_count): # For each layer:
            nodes_in = self.net_dimensions[i]; nodes_out = self.net_dimensions[i+1] # Store the number of neurons in this and next layer (to size arrays)
            if i < self.layer_count - 1: act = self.hidden_type # If in a hidden layer, use the hidden layer activation function refrence
            else: act = self.output_type # If in an output layer, use the output layer activation function refrence
            if act in ["sigmoid", "tanh"]: # If Sigmoid or TanH (Xavier):
                if nu == "n": weight_matrix = np.random.randn(nodes_out, nodes_in) * np.sqrt(2 / (nodes_in + nodes_out)) # Xavier Normal
                else: weight_matrix = np.random.uniform((-1 * (np.sqrt(6 / (nodes_in + nodes_out)))), np.sqrt(6 / (nodes_in + nodes_out)), (nodes_out, nodes_in)) # Xvr Unfrm
            elif act == "relu": # If ReLU (He):
                if nu == "n": weight_matrix = np.random.randn(nodes_out, nodes_in) * np.sqrt(2 / nodes_in) # He Normal
                else: weight_matrix = np.random.uniform((-1 * (np.sqrt(6 / nodes_in))), np.sqrt(6 / nodes_in), (nodes_out, nodes_in)) # He Uniform
            else: weight_matrix = np.random.randn(nodes_out, nodes_in) * 0.1 # Use small random numbers as default
            self.weights.append(weight_matrix); self.biases.append(np.zeros((nodes_out, 1))) # Activate weights, set biases to 0
    def think(self, inputs): ##### Forward pass #####
        self.activated_layer_outputs = [inputs]; self.raw_layer_outputs = [inputs] # Bring in lists for raw and activated layer outputs
        current_input = inputs # Start with the input layer
        for i in range(self.layer_count): # For each layer:
            current_weights = self.weights[i]; current_biases = self.biases[i] # Bring in the current layer's weights and biases
            raw_output = np.clip(current_weights @ current_input + current_biases, 0, 10); self.raw_layer_outputs.append(raw_output) # Compute raw layer output and store
            if i == self.layer_count - 1: layer_output = self.output_function(raw_output) # If in the output layer, compute it's activated output
            else: layer_output = self.hidden_function(raw_output) # If in a hidden layer, compute it's activated output
            self.activated_layer_outputs.append(layer_output) # Store the layer's activated output
            current_input = layer_output # Bring the output from the current layer into the next
        if self.output_type in ["sigmoid", "tanh"]: self.layer_outputs = self.activated_layer_outputs # Refresh layer outputs for sigmoid and TanH (activated)
        elif self.output_type == "relu": self.layer_outputs = self.raw_layer_outputs # Refresh layer outputs for ReLU (raw)
        return current_input # Return output
    def teach(self, method, inputs, signals, lr, epochs=1, decay=1, logs=1): ##### Train Network #####
        inputs = np.asarray(inputs); signals = np.asarray(signals); deltas = [None] * self.layer_count # Convert inputs and targets into NumPy arrays and initialize deltas
        interval = max(1, epochs // logs) if logs != 0 else None; epoch_digits = len(str(epochs)); current_lr = lr # Set log interval and starting learn rate
        for epoch in range(epochs): # For each epoch:
            self.think(inputs) # Forward pass
            if method == "bp": deltas[-1] = (self.activated_layer_outputs[-1] - signals) * self.output_derivative(self.layer_outputs[-1]) # Backprop output layer deltas
            elif method == "rl": deltas[-1] = (signals - self.activated_layer_outputs[-1]) * self.output_derivative(self.layer_outputs[-1]) # Reinforcement output layer deltas
            for i in reversed(range(self.layer_count - 1)): deltas[i] = (self.weights[i+1].T @ deltas[i+1]) * self.hidden_derivative(self.layer_outputs[i+1]) # Hdn lyr dltas
            for i in range(self.layer_count): # For each layer:
                previous_output = self.activated_layer_outputs[i] # Set the previous output correctly
                self.weights[i] -= current_lr * (deltas[i] @ previous_output.T); self.biases[i] -= current_lr * np.mean(deltas[i], axis=1, keepdims=True) # Update wts and bs
            current_lr *= decay # Decay learn rate
            if logs != 0 and epoch % interval == 0: # If a log is expected:
                mse = np.mean((self.activated_layer_outputs[-1] - signals) ** 2) # Compute MSE
                print(f"[Dawn3] {int(100*(epoch/epochs)):02}% E:{epoch:0{epoch_digits}d} LR:{current_lr:.4f} MSE:{mse:.6f}") # Log data
        return np.mean((self.activated_layer_outputs[-1] - signals) ** 2) # Returns MSE
    def sigmoid(self, x): return 1 / (1 + np.exp(-1 * (np.clip(x, -500, 500)))) # Sigmoid
    def sigmoid_der(self, y): return y * (1 - y) # Sigmoid derivative
    def tanh(self, x): return np.tanh(x) # TanH
    def tanh_der(self, y): return 1 - np.clip(y, -0.999, 0.999) ** 2 # TanH derivative
    def relu(self, x): return np.maximum(0, x) # ReLU
    def relu_der(self, x): return (x > 0).astype(float) # ReLU derivative

##########################################################################################################################################################################

class Dawn2: # Broken somehow...
    # Initialize Dawn2
    def __init__(self, net, hidden_act, output_act):
        np.random.seed(1) # Sets random number seed
        self.net = net # Brings in network dimensions
        self.layers = len(net) - 1 # Number of layers (not inputs)
        self.weights = []; self.biases = [] # Creates lists for weights and biases
        self.raw_layer_outputs = []; self.layer_outputs = [] # Creates lists for layer outputs (before and after activation, used in training for deltas)
        self.hidden_act = hidden_act; self.output_act = output_act # Brings in activation functions
        self.hidden_function = {"sigmoid": self.sigmoid, "tanh": self.tanh, "relu": self.relu}.get(hidden_act, self.sigmoid) # Sets hidden activation
        self.hidden_derivative = {"sigmoid": self.sigmoid_der, "tanh": self.tanh_der, "relu": self.relu_der}.get(hidden_act, self.sigmoid_der) # Sets hidden derivative
        self.output_function = {"sigmoid": self.sigmoid, "tanh": self.tanh, "relu": self.relu}.get(output_act, self.sigmoid) # Sets output activation
        self.output_derivative = {"sigmoid": self.sigmoid_der, "tanh": self.tanh_der, "relu": self.relu_der}.get(output_act, self.sigmoid_der) # Sets output derivative
    # Initialize Weights
    def init_parameters(self, type):
        self.weights = []; self.biases = [] # Brings values in
        for i in range(self.layers): # Between each layer:
            nodes_in = self.net[i]; nodes_out = self.net[i + 1] #  Stores number of neurons (this and next layer)
            if type == "xn": weight_matrix = np.random.randn(nodes_out, nodes_in) * np.sqrt(2 / (nodes_in + nodes_out)) # Xavier Normal
            elif type == "xu": limit = np.sqrt(6 / (nodes_in + nodes_out)); weight_matrix = np.random.uniform(-limit, limit, (nodes_out, nodes_in)) # Xavier Uniform
            elif type == "hn": weight_matrix = np.random.randn(nodes_out, nodes_in) * np.sqrt(2 / nodes_in) # He Normal
            elif type == "hu": limit = np.sqrt(6 / nodes_in); weight_matrix = np.random.uniform(-limit, limit, (nodes_out, nodes_in)) # He Uniform
            else: weight_matrix = np.random.randn(nodes_out, nodes_in) * 0.1 # Small random numbers
            self.weights.append(weight_matrix); self.biases.append(np.zeros((nodes_out, 1))) # Activate weights, set biases to zero
    # Forward Pass
    def think(self, inputs):
        self.layer_outputs = [] # Brings values in
        current_input = inputs # Start with the input layer
        for i in range(self.layers): # For each layer:
            current_weights = self.weights[i]; current_biases = self.biases[i] # Brings in current layer's weights and biases
            raw_layer = current_weights @ current_input + current_biases # Computes raw layer (no squish)
            if i == self.layers - 1: current_input = self.output_function(raw_layer) # Output layer activation
            else: current_input = self.hidden_function(raw_layer) # Hidden layer activation
            self.layer_outputs.append(current_input) # Store layer output
        return current_input # Return output
    # Train Network
    def teach(self, method, input, target, epochs, lr, decay, logs):
            print(f'[Dawn2] Teach "{method}", {epochs}, {lr}, {decay}, {logs}')
            current_lr = lr # Sets Current Learn Rate to Starting Learn Rate ("lr")
            input = np.asarray(input); target = np.asarray(target) # Turn inputs and outputs into arrays
            deltas = [None] * self.layers # Makes list of deltas for each layer
            samples = input.shape[1] # Number of samples in training data
            for epoch in range(epochs): # For each epoch:
                self.think(input) # Forward pass
                if method == "bp": # If backprop:
                    deltas[-1] = (self.layer_outputs[-1] - target) * self.output_derivative(self.layer_outputs[-1]) # Delta for output layer using
                elif method == "rl": #HERE
                    gamma = 0.9  # discount factor, tweak as needed
                    discounted_rewards = np.zeros_like(target)
                    cumulative = 0
                    # Compute discounted rewards backwards
                    for t in reversed(range(target.shape[1])):
                        cumulative = target[:, t:t+1] + gamma * cumulative
                        discounted_rewards[:, t:t+1] = cumulative
                    # Compute delta using discounted rewards instead of target
                    deltas[-1] = (self.layer_outputs[-1] - discounted_rewards) * self.output_derivative(self.layer_outputs[-1])
                for i in reversed(range(self.layers - 1)): deltas[i] = (self.weights[i + 1].T @ deltas[i + 1]) * self.hidden_derivative(self.layer_outputs[i]) # HLyrDt
                for i in range(self.layers): # For each layer:
                    if i == 0: previous_output = input # At first, set the previous layer's output to the input
                    else: previous_output = self.layer_outputs[i - 1] # After that, set it as normal
                    self.weights[i] -= current_lr * (deltas[i] @ previous_output.T) # Update weights
                    self.biases[i] -= current_lr * deltas[i] # Update biases
                current_lr = lr * (decay ** epoch) # Decay learning rate
                if epoch % (epochs // logs) == 0: # If a log is expected:
                    mse = np.mean((self.layer_outputs[-1] - target) ** 2) # Find MSE
                    print(f"[Dawn2] {round(100*(epoch/epochs)):02}%, EP:{epoch:0{len(str(epochs))}d}, LR:{current_lr:.3f}, MSE:{mse:.6f}") # Print log
            print(f"[Dawn2] Training Score: {(epochs * mse):.1f} ({epochs} * {mse:.{(len(str(epochs)))-2}f})") # Log training score
    # Math Functions
    def sigmoid(self, x): return 1 / (1 + np.exp(-1 * (np.clip(x, -500, 500)))) # Sigmoid
    def sigmoid_der(self, y): return y * (1 - y) # Sigmoid derivative
    def tanh(self, x): return np.tanh(x) # TanH
    def tanh_der(self, y): return 1 - y ** 2 # TanH derivative
    def relu(self, x): return np.maximum(0, x) # ReLU
    def relu_der(self, x): return (x > 0).astype(float) # ReLu derivative

##########################################################################################################################################################################

class dawn:

    # Init

    def __init__(self, net, hidden_act, output_act):

        np.random.seed(314159)
        self.net = net
        self.hidden_act = hidden_act
        self.output_act = output_act
        self.weights = []
        self.biases = []
        self.layer_inputs = []
        self.layer_outputs = []
        self.stop = False
        self.pause = False

    # Init Weights

    def init_weights(self, type):

        self.weights = []
        self.biases = []

        for i in range(len(self.net) - 1):

            in_n = self.net[i]
            out_n = self.net[i+1]

            if type == "xavier_normal":
                new_weights = np.random.randn(out_n, in_n) * np.sqrt(2 / (in_n + out_n))
            elif type == "xavier_uniform":
                limit = np.sqrt(6 / (in_n + out_n))
                new_weights = np.random.uniform(-limit, limit, (out_n, in_n))
            elif type == "he_normal":        
                new_weights = np.random.randn(out_n, in_n) * np.sqrt(2 / in_n)
            elif type == "he_uniform":
                limit = np.sqrt(6 / in_n)
                new_weights = np.random.uniform(-limit, limit, (out_n, in_n))
            else:
                limit = np.sqrt(6 / (in_n + out_n))
                new_weights = np.random.uniform(-limit, limit, (out_n, in_n))

            self.weights.append(new_weights)
            self.biases.append(np.zeros((out_n, 1)))

    # Forward Pass

    def think(self, input_data):

        self.layer_inputs = []
        self.layer_outputs = []
        current_layer = input_data
        
        for i in range(len(self.weights)):
            
            weight_set = self.weights[i]
            bias_set = self.biases[i]
            result = np.dot(weight_set, current_layer) + bias_set
            self.layer_inputs.append(result)

            if i == len(self.weights) - 1:
                current_layer = self.activate(result, "output")
            else:
                current_layer = self.activate(result, "hidden")

            self.layer_outputs.append(current_layer)

        return current_layer

    # Train Network (RIGHT NOW ONLY ONE AT A TIME! FIX!)

    def teach(self, input_data, target, learn_rate, method, epochs, checks, decay):
        
        if method == "backprop":

            print(f"[dawn] LR: {learn_rate}, {method}, EP:{epochs}, DC:{decay}")
            current_rate = learn_rate

            for epoch in range(epochs):

                self.think(input_data)
                deltas = [None] * len(self.weights)
                delta_output = (self.layer_outputs[-1] - target) * self.derivative(self.layer_inputs[-1], "output")
                deltas[-1] = delta_output

                for i in reversed(range(len(self.weights) - 1)):
                    deltas[i] = np.dot(self.weights[i+1].T, deltas[i+1]) * self.derivative(self.layer_inputs[i], "hidden")

                for i in range(len(self.weights)):
                    if i == 0:
                        previous_output = np.array(input_data)
                    else:
                        previous_output = self.layer_outputs[i-1]

                    self.weights[i] -= current_rate * np.dot(deltas[i], previous_output.T) / input_data.shape[1]
                    self.biases[i] -= current_rate * np.mean(deltas[i], axis=1, keepdims=True)

                mse = np.mean((self.layer_outputs[-1] - target) ** 2)
                current_rate *= decay

                if epoch % (epochs // checks) == 0:
                    print(f"[dawn] {round(100*(epoch/epochs)):02}%, Ep: {epoch:0{len(str(epochs))}d}, LR: {current_rate:.3f}, MSE: {mse:.6f}")

        elif method == "reinforce":
            pass # ADD REINFORCEMENT HERE!

    # Math Functions

    def activate(self, x, layer_type):
        if layer_type == "output":
            if self.output_act == "sigmoid":
                return self.sigmoid(x)
            elif self.output_act == "tanh":
                return self.tanh(x)
            elif self.output_act == "relu":
                return self.relu(x)
            else:
                return self.sigmoid(x)
        else:
            if self.hidden_act == "sigmoid":
                return self.sigmoid(x)
            elif self.hidden_act == "tanh":
                return self.tanh(x)
            elif self.hidden_act == "relu":
                return self.relu(x)
            else:
                return self.sigmoid(x)

    def derivative(self, x, layer_type):
        if layer_type == "output":
            if self.output_act == "sigmoid":
                return self.sigmoid_der(x)
            elif self.output_act == "tanh":
                return self.tanh_der(x)
            elif self.output_act == "relu":
                return self.relu_der(x)
            else:
                return self.sigmoid_der(x)
        else:
            if self.hidden_act == "sigmoid":
                return self.sigmoid_der(x)
            elif self.hidden_act == "tanh":
                return self.tanh_der(x)
            elif self.hidden_act == "relu":
                return self.relu_der(x)
            else:
                return self.sigmoid_der(x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * np.clip(x, -500, 500)))

    def sigmoid_der(self, x):
        return np.clip(x, 0, 1) * (1 - np.clip(x, 0, 1))

    def tanh(self, x):
        return np.tanh(x)

    def tanh_der(self, x):
        return 1 - np.tanh(x) ** 2

    def relu(self, x):
        return np.maximum(0, x)

    def relu_der(self, x):
        return (x > 0).astype(float)

##########################################################################################################################################################################

class byte:

    def __init__(self, layer_sizes):
        print("")
        self.layer_sizes = layer_sizes
        self.weights = []
        np.random.seed(1)
        for i in range(len(layer_sizes) - 1):
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
            w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
            self.weights.append(w)
        print("[byte] Byte Setup Complete!")

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def sigmoid_der(self, x):
        return x * (1 - x)

    def push(self, x):
        layer = x
        for w in self.weights:
            layer = self.sigmoid(np.dot(layer, w))
        return layer

    def train(self, x, y, epochs, learn_rate, checks):
        print("[byte] Byte Training underway...")
        for epoch in range(epochs):
            layers = [x]
            for w in self.weights:
                layers.append(self.sigmoid(np.dot(layers[-1], w)))
            error = y - layers[-1]
            delta = error * self.sigmoid_der(layers[-1])
            deltas = [delta]
            for i in reversed(range(len(self.weights) - 1)):
                delta = deltas[-1].dot(self.weights[i+1].T) * self.sigmoid_der(layers[i+1])
                deltas.append(delta)
            deltas.reverse()
            for i in range(len(self.weights)):
                self.weights[i] += layers[i].T.dot(deltas[i]) * learn_rate
            if epoch % (epochs // checks) == 0:
                mse = np.mean((y - layers[-1])**2)
                print(f"[byte] {round(100 * (epoch / epochs))}%, Epoch: {epoch} , MSE: {mse:.6f}")
        print("[byte] Byte Training Complete!")
        print("")

##########################################################################################################################################################################

class node2: # Slightly edited for class integration, functionalty identical

    def __init__(self):
        np.random.seed(1)
        self.weights = []

    x = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])

    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def sigmoid_der(x):
        return x * (1 - x)

    def train(self, layer_sizes, epochs, x, y, learn_rate):
        print("")
        print("[node2] Beginning training...")
        print("")

        for i in range(len(layer_sizes) - 1):
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
            w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
            self.weights.append(w)

        for _ in range(epochs):
            layers = [x]
            for w in self.weights:
                layers.append(self.sigmoid(np.dot(layers[-1], w)))
            deltas = [(y - layers[-1]) * self.sigmoid_der(layers[-1])]
            for i in reversed(range(len(self.weights)-1)):
                delta = deltas[0].dot(self.weights[i+1].T) * self.sigmoid_der(layers[i+1])
                deltas.insert(0, delta)
            for i in range(len(self.weights)):
                self.weights[i] += layers[i].T.dot(deltas[i]) * learn_rate
            if _ % (epochs // 10) == 0:
                mse = np.mean((y -  layers[-1])**2)
                print(f"[node2] Epoch {_}, MSE: {mse:.6f}")

        final_output = layers[-1]
        return final_output

##########################################################################################################################################################################

class node1: # Slightly edited for class integration, functionalty identical

    def __init__(self):
        np.random.seed(1)
        self.weights = []

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_der(x):
        return x * (1 - x)

    def train(self, layer_sizes, epochs, X, Y, learn_rate):

        print("[node1] Beginning training...")
        print("")

        for i in range(len(layer_sizes) - 1):
            w = 2 * np.random.random((layer_sizes[i], layer_sizes[i + 1])) * 0.1
            self.weights.append(w)

        for _ in range(epochs):
            layers = [X]
            for w in self.weights:
                layers.append(self.sigmoid(np.dot(layers[-1], w)))
            deltas = [(Y - layers[-1]) * self.sigmoid_der(layers[-1])]
            for i in reversed(range(len(self.weights) - 1)):
                delta = deltas[0].dot(self.weights[i + 1].T) * self.sigmoid_der(layers[i + 1])
                deltas.insert(0, delta)
            for i in range(len(self.weights)):
                self.weights[i] += layers[i].T.dot(deltas[i]) * learn_rate
            if _ % (epochs // 50) == 0: 
                mse = np.mean((Y - layers[-1])**2)
                print(f"[node1] Epoch {_}, MSE: {mse:.6f}")
    
    def think(self, input):
        inp = np.array(input).reshape(1,-1)
        layer = [inp]
        for w in self.weights:
            layer.append(self.sigmoid(np.dot(layer[-1], w)))
        output = layer[-1][0]
        return output

##########################################################################################################################################################################

class betternn: # Slightly edited for class integration, functionalty identical

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_der(x):
        return x * (1 - x)

    def train(self, layer_sizes, learn_rate, epochs, x, y):

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
                layers.append(self.sigmoid(np.dot(layers[-1], w)))
            deltas = [(y - layers[-1]) * self.sigmoid_der(layers[-1])]
            for i in reversed(range(len(weights)-1)):
                delta = deltas[0].dot(weights[i+1].T) * self.sigmoid_der(layers[i+1])
                deltas.insert(0, delta)
            for i in range(len(weights)):
                weights[i] += layers[i].T.dot(deltas[i]) * learn_rate
            if _ % (epochs // 50) == 0:
                mse = np.mean((y -  layers[-1])**2)
                print(f"[betternn] Epoch {_}, MSE: {mse:.6f}")

        final_output = layers[-1]

        return final_output

##########################################################################################################################################################################

class nntest: # Slightly edited for class integration, functionalty identical

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(x):
        return x * (1 - x)

    def train(self, x, y, epochs, lr):

        np.random.seed(1)

        weights0 = 2 * np.random.random((4, 8)) - 1
        weights1 = 2 * np.random.random((8, 8)) - 1
        weights2 = 2 * np.random.random((8, 6)) - 1
        weights3 = 2 * np.random.random((6, 3)) - 1

        for epoch in range(epochs):
            layer0 = x
            layer1 = self.sigmoid(np.dot(layer0, weights0))
            layer2 = self.sigmoid(np.dot(layer1, weights1))
            layer3 = self.sigmoid(np.dot(layer2, weights2))
            layer4 = self.sigmoid(np.dot(layer3, weights3))

            layer4_error = y - layer4
            layer4_delta = layer4_error * self.sigmoid_derivative(layer4)

            layer3_error = layer4_delta.dot(weights3.T)
            layer3_delta = layer3_error * self.sigmoid_derivative(layer3)

            layer2_error = layer3_delta.dot(weights2.T)
            layer2_delta = layer2_error * self.sigmoid_derivative(layer2)

            layer1_error = layer2_delta.dot(weights1.T)
            layer1_delta = layer1_error * self.sigmoid_derivative(layer1)

            weights3 += layer3.T.dot(layer4_delta) * lr
            weights2 += layer2.T.dot(layer3_delta) * lr
            weights1 += layer1.T.dot(layer2_delta) * lr
            weights0 += layer0.T.dot(layer1_delta) * lr

        return layer4