import logging

import numpy as np
import torch

from pytorch_mimo.Enums import *
from pytorch_mimo.Signal import Signal

logger = logging.getLogger('mimo')


class Feedforward(torch.nn.Module):
    required_domain = Domain.TIME

    def __init__(self, n_out_streams, linear_bypass=1, n_mem=1, n_ues=1, n_hidden_neurons=80, n_layers=1, **kwargs):
        super(Feedforward, self).__init__()
        # TODO. Gerneralize to multiple layers
        logging.info(f'Creating a NN with {n_hidden_neurons} for {n_ues} user.')  # will print a message to the console
        self.n_ues = n_ues
        self.n_mem = n_mem
        self.n_out_streams = n_out_streams
        self.linear_bypass = linear_bypass
        self.n_input_neurons = n_mem * 2 * n_ues
        self.n_hidden_neurons = n_hidden_neurons
        self.fc1 = torch.nn.Linear(self.n_input_neurons, self.n_hidden_neurons)
        self.linear_bypass_layer = torch.nn.Linear(self.n_input_neurons, n_out_streams * 2, )
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.n_hidden_neurons, n_out_streams * 2)

    def use(self, x_sig: Signal) -> Signal:
        x_sig.match_this(self.required_domain)
        x = x_sig.tensor[:, :self.n_ues]  # Need to prune in case of vues
        real_part = x.real
        imag_part = x.imag
        x_in = torch.cat((real_part, imag_part), 1)
        for i_mem in np.arange(1, self.n_mem - 1):
            delayed_basis = torch.roll(x_in, i_mem)
            delayed_basis[:i_mem] = 0
            x_in = torch.cat((x_in, delayed_basis), 1)

        hidden = self.fc1(x_in)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        bypass = self.linear_bypass_layer(x_in)
        output = output + bypass

        # Convert to complex
        real_part = output[:, 0:self.n_out_streams]
        imag_part = output[:, self.n_out_streams:]
        full = torch.complex(real_part, imag_part)

        input_shape = x.shape
        full = torch.reshape(full, [input_shape[0], self.n_out_streams])

        out = Signal(full, 'NN out', x_sig.sample_rate, self.required_domain, x_sig.n_symbols,
                     x_sig.fft_size, self.n_out_streams, x_sig.rrc_taps)
        return out

    def backprop_update(self, loss):
        with torch.no_grad():
            self.linear_bypass_mat -= 0.0001 * self.linear_bypass_mat.grad
            self.linear_bypass_mat.grad = None
