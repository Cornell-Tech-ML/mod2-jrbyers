"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch
from minitorch.tensor import Tensor
import numpy as np
import random

# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)

# TODO: Implement for Task 2.5.
class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()

        self.hidden_layers = hidden_layers
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        middle = self.layer1.forward(x).relu()
        end = self.layer2.forward(middle).relu()
        output = self.layer3.forward(end)
        return output.sigmoid()

class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        
        """
        xavier_weights = self.get_xavier_weights(in_size, out_size)

        # Create weights and biases as tensors
        self.weights = minitorch.tensor(xavier_weights.tolist(), requires_grad=True)
        self.bias = minitorch.tensor([2 * (random.random() - 0.5) for _ in range(out_size)], requires_grad=True)
        
        self.bias = Tensor.make(
            [2 * (random.random() - 0.5) for _ in range(out_size)], 
            (out_size,)
        )"""


    def forward(self, inputs):
        # Ensure inputs are in the correct shape
        assert inputs.shape[1] == self.weights.value.shape[0], "Input size must match weights size."

        batch_size = inputs.shape[0]
        num_features = inputs.shape[1]
        hidden_size = self.weights.value.shape[1]

        inputs = inputs.view(batch_size, num_features, 1)

        mul_weight = inputs * self.weights.value

        reduced_X = mul_weight.sum(1)

        output = reduced_X + self.bias.value
        new_tensor = output.view(batch_size, hidden_size)
        return new_tensor
        
    
    @staticmethod
    def get_xavier_weights(fan_in: int, fan_out: int):
        """Function for producing Xavier initialization as outlined in Ed post 179."""
        n = fan_in * fan_out
        random_weights = np.random.uniform(low=-1.0, high=1.0, size=n)

        # Adjust the mean to be exactly 0
        actual_mean = np.mean(random_weights)
        xavier_weights = random_weights - actual_mean

        # Calculate desired variance
        desired_variance = 2/ (fan_in + fan_out)

        # Adjust the variance to be the desired variance
        actual_variance = np.var(xavier_weights)
        scaling_factor = np.sqrt(desired_variance / actual_variance)
        xavier_weights = xavier_weights * scaling_factor

        return xavier_weights


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
