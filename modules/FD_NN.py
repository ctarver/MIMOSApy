import logging

import torch

from pytorch_mimo.Signal import Signal

logger = logging.getLogger('mimo')


class Feedforward(torch.nn.Module):
    required_domain = 'frequency'

    def __init__(self, n_mem=1, n_ues=1, n_hidden_neurons=80, n_layers=1, n_subcarriers=1200, n_out_scs=4096, **kwargs):
        super(Feedforward, self).__init__()
        # TODO. Gerneralize to multiple layers
        logging.info(f'Creating a NN with {n_hidden_neurons} for {n_ues} user.')  # will print a message to the console
        self.n_ues = n_ues
        self.n_input_neurons = 2 * n_ues * n_subcarriers
        self.n_hidden_neurons = self.n_input_neurons
        self.n_output_neurons = 2 * n_out_scs
        self.fc1 = torch.nn.Linear(self.n_input_neurons, self.n_hidden_neurons)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.n_hidden_neurons, 2 * self.n_output_neurons)

    def use(self, x_sig: Signal) -> Signal:
        x_sig.match_this(self.required_domain)
        x = x_sig.tensor
        real_part = x.real
        imag_part = x.imag
        x_in = torch.cat((real_part, imag_part), 1)
        hidden = self.fc1(x_in)
        relu = self.relu(hidden)
        output = self.fc2(relu)

        # Convert to complex
        real_part = output[:, 0:self.n_ues]
        imag_part = output[:, self.n_ues:]
        full = torch.complex(real_part, imag_part)
        full = torch.reshape(full, x.shape) + x
        out = Signal(full, 'NN out', x_sig.sample_rate, self.required_domain, x_sig.n_symbols,
                     x_sig.fft_size, x_sig.n_streams)
        return out
