import numpy as np
import torch

from pytorch_mimo.Signal import Signal, Domain


class GMP:
    def __init__(self, order: int = 7, memory: int = 4, lag: int = 0, step: int = 2):
        self.order = order  # Polynomial order to model
        self.memory = memory  # Number of memory taps
        self.lag = lag  # Number of lag/lead taps
        self.step = step

        # Setup the coeffs to default, linear, state
        self.coeffs = torch.zeros((self.n_coeffs, 1))
        self.coeffs[0] = 1
        self.rms_learned = []  # The rms of the input signal that we learned at.

    def use(self, gmp_in):
        """ This method will choose the correct strategy based on incoming type."""
        if type(gmp_in) is Signal:
            use_function = self.signal_use
        elif type(gmp_in) is torch.Tensor:
            use_function = self.tensor_use
        else:
            Exception('Wrong type incoming to GMP')
        return use_function(gmp_in)

    def signal_use(self, gmp_in: Signal) -> Signal:
        gmp_in.match_this(Domain.TIME)
        x = gmp_in.tensor
        # Rescale to desired rms
        y = self.tensor_use(x)
        gmp_out = Signal(y, name='GMP Out', sample_rate=gmp_in.sample_rate, domain=Domain.TIME,
                         n_symbols=gmp_in.n_symbols, fft_size=gmp_in.fft_size,
                         n_streams=gmp_in.n_streams)
        return gmp_out

    def tensor_use(self, x: torch.Tensor) -> torch.Tensor:
        """Main method that operated on a tensor."""
        X = self.setup_basis_matrix(x)
        return torch.matmul(X, self.coeffs)

    def learn(self, x, y):
        """learns a GMP model."""
        # Divide out gain of each PA? TODO
        X = self.setup_basis_matrix(x)
        self.coeffs = self.least_squares(X, y)

    def setup_basis_matrix(self, x):
        n_samples = len(x)
        X = torch.zeros((n_samples, self.n_coeffs), dtype=torch.cfloat)
        column_index = 0
        for i_order in np.arange(1, self.order + 1, self.step):
            basis = x * torch.pow(torch.abs(x), i_order - 1)
            for i_mem in np.arange(self.memory):
                delayed_basis = torch.roll(basis, i_mem)
                delayed_basis[:i_mem] = 0
                X[:, column_index] = delayed_basis
                column_index = column_index + 1
        return X

    def calculate_model_error(self, x, y):
        X = self.setup_basis_matrix(x)
        model_output = torch.matmul(X, self.coeffs)
        squared_error = torch.pow(torch.abs(model_output - y), 2)
        mean_squared_error = torch.mean(squared_error)
        expected_mean_output_power = torch.mean(torch.pow(torch.abs(y), 2))
        nmse = mean_squared_error / expected_mean_output_power
        return nmse

    @property
    def n_coeffs(self):
        if self.step == 1:
            n_order_terms = self.order
        else:
            n_order_terms = (self.order + 1) / 2
        n_coeffs = n_order_terms * self.memory
        return int(n_coeffs)

    @property
    def n_mults(self):
        pass

    @staticmethod
    def least_squares(X, y, regularizer: float = 0.001):
        X_hermatian = torch.conj(torch.transpose(X, 0, 1))
        gram = torch.matmul(X_hermatian, X)
        left_inverse = torch.inverse(gram + regularizer * torch.eye(len(gram)))
        right_side = torch.matmul(X_hermatian, y)
        b = torch.matmul(left_inverse, right_side)
        return b
