import abc

import numpy as np
import torch

from pytorch_mimo.Enums import *
from pytorch_mimo.GMP import GMP
from pytorch_mimo.Signal import Signal


@dataclass
class PAs(abc.ABC):
    """This class is an interface. Subclass will store and invoke a bunch of PAs."""
    pa_required_domain: Domain

    @abc.abstractmethod
    def use(self, x: Signal) -> Signal:
        pass


@dataclass
class Linear(PAs):
    """Basic PA Bypass class to debug."""
    n_streams: int

    def use(self, x: Signal) -> Signal:
        x.normalize_to_this_rms(1)
        return x


@dataclass
class MP(PAs):
    """Main subclass that will implement an array of GMPs"""
    n_streams: int
    order: int
    step: int  #
    memory: int
    lag: int
    variance: float

    def __post_init__(self):
        # Setup a list of GMPs.
        self.models = [GMP(order=self.order, memory=self.memory, lag=self.lag, step=self.step)
                       for i in range(self.n_streams)]
        self.make_coeffs()
        self.scale_factor = 0

    def make_coeffs(self):
        # Make a big set of default coeffs for 7th order, 4 taps.
        default_coeffs = torch.tensor(
            [[0.9295 - 0.0001j, 0.2939 + 0.0005j, -0.1270 + 0.0034j, 0.0741 - 0.0018j],  # 1st order
             [0.01 - 0.008j, -0.001 - 0.0008j, -0.0535 + 0.0004j, 0.0908 - 0.0473j],  # 3rd oder
             [-0.005 - 0.0169j, -0.0005 + 0.000169j, -0.3011 - 0.1403j, -0.0623 - 0.0269j],  # 5th order
             [0.01774 + 0.0265j, 0.0848 + 0.0613j, -0.0362 - 0.0307j, 0.0415 + 0.0429j]])  # 7th order
        n_poly_terms = int((self.order + 1) / self.step)
        pruned_coeffs = default_coeffs[:n_poly_terms, :self.memory]
        default_coeffs = torch.flatten(pruned_coeffs)

        mag_of_each = torch.abs(default_coeffs)

        # Add variance and set to each
        for gmp in self.models:
            Z = self.variance * (torch.rand(default_coeffs.size()) + torch.rand(default_coeffs.size()) * 1j)
            gmp.coeffs = default_coeffs + np.multiply(Z, mag_of_each)

    def use(self, x: Signal) -> Signal:
        """Need to take the input signal and feed it into all the GMPs"""
        x.match_this(Domain.TIME)

        if not self.scale_factor:
            self.scale_factor = x.normalize_to_this_rms(1)
        else:
            x.tensor = self.scale_factor * x.tensor

        gmp_out = torch.zeros(x.tensor.size(), dtype=torch.cfloat)
        for index, gmp in enumerate(self.models):
            gmp_out[:, index] = gmp.use(x.tensor[:, index])
        return Signal(gmp_out, x.name, x.sample_rate, Domain.TIME, x.n_symbols, x.fft_size, x.n_streams, x.rrc_taps,
                      x.cp_length, x.use_windows, x.window_length, x.device)


@dataclass
class MP_Same(PAs):
    def __post_init__(self):
        self.scale_factor = 0
        self.gain = 0

    def set_scale_factors(self, x_in: Signal):
        # Take IFFT of each...
        x = x_in.copy()
        x.match_this(self.pa_required_domain)
        self.scale_factor = x.normalize_to_this_amp(0.9)

    def calculate_gain(self, x_in: Signal):
        x = x_in.copy()
        x.match_this(self.pa_required_domain)
        x.tensor = x.tensor * self.scale_factor
        x_mp = self.mp(x.tensor)
        x_out = Signal(x_mp, name='PA Out', sample_rate=x.sample_rate, domain=self.pa_required_domain,
                       n_symbols=x.n_symbols, fft_size=x.fft_size, n_streams=x.n_streams, rrc_taps=x.rrc_taps,
                       device=x.device)
        input_rms = x.calculate_rms()
        output_rms = x_out.calculate_rms()
        self.gain = output_rms/input_rms


    def use(self, x: Signal) -> Signal:
        # Take IFFT of each...
        x.match_this(self.pa_required_domain)
        x.tensor = x.tensor * self.scale_factor
        x_mp = self.mp(x.tensor)
        x_out = Signal(x_mp, name='PA Out', sample_rate=x.sample_rate, domain=self.pa_required_domain,
                       n_symbols=x.n_symbols, fft_size=x.fft_size, n_streams=x.n_streams, rrc_taps=x.rrc_taps,
                       device=x.device)
        return x_out

    def mp(self, x_time):
        #x_mp = (1.0441 - 0.1768j)*x_time + (0.4991 + 0.7764j) * x_time * (abs(x_time) ** 2)  + (-0.7265 - 0.5481j) * x_time * (abs(x_time) ** 4)
        x_mp = (1.0427 + 0.1791j)*x_time + (-0.2449 - 0.9761j) * x_time * (abs(x_time) ** 2)  #+ (0.2166 + 0.7790j) * x_time * (abs(x_time) ** 4)
        #x_mp = (1.0222 + 0.1053j)*x_time + (-0.0833 - 0.3952j) * x_time * (abs(x_time) ** 2)
        #x_mp = x_time + (-0.0833 - 0.3952j) * x_time * (abs(x_time) ** 2)
        #x_mp = x_time  + 0.4 * x_time * (abs(x_time) ** 2)
        return x_mp


if __name__ == "__main__":
    my_pa = PAs.create(pa_type=PaTypes.GMP, pa_required_domain=Domain.TIME, n_streams=1, order=7, step=2, memory=7,
                       lag=0, variance=0.01)
    1 + 1
