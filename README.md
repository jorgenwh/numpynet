### Disclaimer
This project only contains simple implementations for very limited functionality and should not really be used for anything.

## Installation

To install all necessary dependencies to run the example.py file
```bash
pip install -r requirements.txt
```

To install numpynet using pip
```bash
pip install .
```

## Example usage
A very simple mnist example can be found in example.py, but to summarize the VERY limited current functionality
```python
import numpy as np
from numpynet.model import Model
from numpynet.layers import Linear
from numpynet.activations import ReLU
from numpynet.losses import MSE

# The model object takes a loss function object, so we create a MSE loss object
# to pass to the model constructor
loss_function = MSE()
model = Model(loss=loss_function)

# Add some linear layers and ReLU activation functions.
# Activation functions are added in the same way as layers
model.add_layer(Linear(4, 10))
model.add_layer(ReLU())
model.add_layer(Linear(10, 2)) # Linear activations for the output layer

X = np.random.uniform(-1, 1, (1, 4))
t = np.random.uniform(-1, 1, (1, 2))

# Forward pass
y = model(X)
# Compute loss
loss = model.loss(y, t)
# Backward pass
model.backward()
# Update parameters
model.update_step(lr=0.01)
# Set all grads to zero
model.zero_grad()
```