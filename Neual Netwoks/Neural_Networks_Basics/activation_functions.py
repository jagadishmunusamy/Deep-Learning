"""
🔥 1. Sigmoid (Logistic Function)
        σ(x)= 1/(1 + e^(-x1))
    ✅ Best For
            Binary classification (output layer)
    👍 Advantages
            Smooth probability output (0 to 1)

    👎 Disadvantages
            Vanishing gradients
            Not zero-centered
            Slows training
"""
import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-2, -1, 0, 1, 2])
print(sigmoid(x))
""" Pytorch Implementation"""

import torch
import torch.nn as nn
sig = nn.Sigmoid()
x = torch.tensor([-2.,-1.,0,1.,2.])
print(sig(x))


"""
🔥 2. Tanh (Hyperbolic Tangent)
        Formula, tanh(x) = (e^x - e^-x) / e^x + e^-x)

    ✅ Best For
            Hidden layers (earlier deep networks)
    👍 Advantages
            Zero-centered (better than sigmoid)
    👎 Disadvantages
    Still suffers from vanishing gradients
"""
def tanh(x):
    return np.tanh(x)
print("Tanh: ",tanh(x))


print("git_commit")