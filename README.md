# Hello, World!
This is where I keep all of my neural network experiments, from my very first neural net “nntest.py” that could only learn XOR, all the way to DAWN, a fully-fledged, dynamic, and adaptable program that can play games against you.

## Dawn3 (Dynamically Adaptive Weighted Network v3) - Capable
My best neural network so far, currently under development. She is also my first completely handwritten network program (all other programs I used help for). Dawn can be called as a class and given specifications, such as:

* **Network Dimensions** - Hidden Layers, and Nodes per Layer (which can be unique for each layer)
    * Given as an array, such as `[9,128,128,64,32,9]` (can be any length, width and shape)
* **Activation Functions** - Hidden Layers and the Output Layer (which can each be unique per layer type)
    * Choice between `sigmoid`, `tanh`, and `relu` for each layer type (hidden or output)
* **Weight Initiation Algorithm** - Function type used to initiate random weights on `init_weights()`
    * Choice between `xavier_normal`, `he_normal`, `xavier_uniform`, and `he_uniform`
* **Training Configuration** - Many useful variables used when training the network depending on the task
    * Examples include: `epochs`, `learn_rate`, `decay`, and progress `checks` that print as you train
* **Training Data** - Because, y'know, it's sort of essential... it can't really teach itself something it doesn't know!

Some other neat features include training updates, with info such as the Epoch, dynamic Learn Rate, and MSE. DAWN can also be put into RL mode, where she can learn through reinforcement!*

## Node - User-Friendly
My first network to communicate with the user! Node was also my first poke at trying to write a network program myself, although I still ended up needing some help when it came to writing the code itself.

## Byte - Implementable
Byte was my first network to be used for a task, as I used it in a program designed to play Tic-Tac-Toe against the user! It got rid of many shortfalls and inefficiencies that kept the first two networks from tackling anything past XOR. 

## "`nntest`" and "`betternn`" - Functional
These two programs represent the first time I dipped my toes into neural networks! I needed to self-teach Python and the concept of neural networks themselves to even understand their code (mostly written by AI), which ended up helping me grow into a significantly more talented programmer. They were my first steps into Python, helping me to grow into the teenage software developer I am today!

*(Yes, I know that there are many bugs, but DAWN is only periodically worked on, so she isn't always perfect. And neither is this README... just check the code itself. Because come on- I'm 15, guys. Deal with it)