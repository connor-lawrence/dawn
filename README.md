# Dawn - Dynamically Adaptive Weighted Neural-Network
This repository contains the code for my flagship neural network, `Dawn3`, along with her earlier versions and younger siblings!

From `nntest` learning XOR to `Dawn3` playing Tic-Tac-Toe against you (and losing), join me on my neural network journey!

## From `nntest` to `Dawn3`
This repository represents something that I take a lot of pride in, both as a software developer and as a person. All of the highs, all of the lows, the late nights and the early mornings went into bringing `Dawn` to life.

She represents many of the ways I now approach programming differently, from project structure to my coding style. I self-taught Python over several weeks just to build her, utilizing the same strategy I now use wnen learning new programming languages. That's why I named her `Dawn`... she's just the beginning.

And, most importantly... I’ve always struggled with staying focused on single long-term projects. `Dawn` is my first completed software project.

## Models

* `nntest` and `betternn` represent my first steps into neural networks. I self-taught both Python and the core ideas behind neural networks just to understand and build them (written almost completely by AI), which ended up helping me grow significantly as a programmer. They were my introduction into Python, a major step in helping me to become the teenage software developer I am today!

* `Node` was my first network designed to communicate with the user! It was also my first attempt at writing a neural network program on my own, although I did end up needing some help when it came to writing the code itself.

* `Byte` was my first network used in a real task, implemented in a program designed to play Tic-Tac-Toe against the user! It rid of many shortfalls and inefficiencies that kept the first two networks from tackling anything past XOR. 

* `Dawn3` is my finished flagship Python class, designed for use anywhere. She is also my first completely handwritten network program (all other programs I used help for). She is called as a class, and given arguments such as:

    * `Network Dimensions`, given as an array (such as `[9,128,128,64,32,9]`) with any length, width and shape
    * `Activation Functions`, given for both hidden and output layers (which may be separate if desired), as well as `Normal` vs `Uniform` for weight/bias initialization
    * `Teaching Method`, such as Backprop (BP) and Reinforcement Learning (RL).

    Some other capabilities she has includes `Decay`, automatic weight/bias initialization algorithm selection according to the `Activation Functions`, NumPy integration for significantly faster training, among others. Search around `Dawn3` in `neuralnets.py` for more if you're curious!