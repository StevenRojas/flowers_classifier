import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


class Network(nn.Module):
    """
    Class for a feed forward network created dynamically for a given architecture
    @author: Steven Rojas <steven.rojas@gmail.com>
    """

    ACTIVATION_RELU = 'relu'
    ACTIVATION_SOFTMAX = 'softmax'
    ACTIVATION_NONE = 'none'

    def __init__(self, arch):
        """
        Creates a feed forward model for a given architecture
        :param arch: Dictionary of network architecture with the following fields:
            id: model identifier
            input: number of inputs
            hidden: array with number of nodes per hidden layer
            output: number of outputs
            h_act: activation function for hidden layers (default ReLU)
            o_act: activation function for output layer (default log_softmax)
            drop_p: Dropout probability (if not set it is not applicated)

            example: {"id": "1", "input": 784, "hidden": [256, 128, 64], "output": 10,
                      "h_act": "relu", "o_act": "softmax", "drop_p": 0.3},
        """
        super().__init__()
        #  TODO: Validate architecture
        self.layers = nn.ModuleList([nn.Linear(arch['input'], arch['hidden'][0])])
        self.layers.extend([nn.Linear(h1, h2)
                            for h1, h2 in zip(arch['hidden'][:-1], arch['hidden'][1:])])
        self.output = nn.Linear(arch['hidden'][-1], arch['output'])

        if 'drop_p' in arch:
            self.dropout = nn.Dropout(p=arch['drop_p'])
        else:
            self.dropout = None

        self.h_act = arch['h_act']
        self.o_act = arch['o_act']
        self.id = arch['id']

        self.lr = None
        self.architecture = []
        self.architecture.append(arch['input'])
        for h in arch['hidden']:
            self.architecture.append(h)
        self.architecture.append(arch['output'])
        self.arch = arch

    def forward(self, x):
        """
        Forward the input through the model and return the logits applying the function defined in arch['o_act']
        :param x: The input array
        :return: logits
        """

        for layer in self.layers:
            x = self.__apply_hidden_activation(layer(x))
            if self.dropout is not None:
                x = self.dropout(x)
        return self.__apply_output_activation(self.output(x))

    def set_lr(self, lr):
        self.lr = lr

    def get_lr(self):
        return self.lr

    def get_id(self):
        return self.id

    def get_description(self):
        return "Network {}: {} lr={}".format(self.id, self.architecture, self.lr)

    def get_architecture(self):
        return self.arch

    def __apply_hidden_activation(self, x):
        if self.h_act == self.ACTIVATION_RELU:
            return F.relu(x)
        # TODO: Add more activation functions
        return F.relu(x)  # Default

    def __apply_output_activation(self, x):
        if self.h_act == self.ACTIVATION_SOFTMAX:
            return F.log_softmax(x, dim=1)
        if self.h_act == self.ACTIVATION_NONE:
            return x
        # TODO: Add more activation functions
        return F.log_softmax(x, dim=1)  # Default
