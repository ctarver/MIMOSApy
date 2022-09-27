import abc

from pytorch_mimo.Channel import Channel
from pytorch_mimo.Signal import *
from pytorch_mimo.Signal import Signal


@dataclass
class Precoder():
    precoder_required_domain: Domain
    device: torch.device

    @property
    @abc.abstractmethod
    def P(self):
        """Abstract property for precoder matrix."""

    @abc.abstractmethod
    def calculate_precoder(self, channel: Channel):
        pass

    def use(self, s: Signal, index=-1) -> Signal:
        """Use the precoder. If "index" is nonnegative, we only apply the precoder on those index"""
        s.match_this(desired_domain=self.precoder_required_domain)
        if index >= 0:
            self.P[:, :, index] = torch.zeros(self.P[:, :, index].size())
        x = torch.matmul(self.P, s.tensor)
        _, n_ants, _ = x.size()
        x_out = Signal(x, name='Precoded', sample_rate=s.sample_rate, domain=self.precoder_required_domain,
                       n_symbols=s.n_symbols, fft_size=s.fft_size, n_streams=n_ants, rrc_taps=s.rrc_taps,
                       device=s.device)
        return x_out

    def use_with_this(self, s: Signal, P) -> Signal:
        """Use the precoder. If "index" is nonnegative, we only apply the precoder on those index"""
        s.match_this(desired_domain=self.precoder_required_domain)
        x = torch.matmul(P, s.tensor)
        _, n_ants, _ = x.size()
        x_out = Signal(x, name='Precoded', sample_rate=s.sample_rate, domain=self.precoder_required_domain,
                       n_symbols=s.n_symbols, fft_size=s.fft_size, n_streams=n_ants, rrc_taps=s.rrc_taps,
                       device=s.device)
        return x_out


class ZF(Precoder):
    def calculate_precoder(self, channel):
        logger.debug(f'Setting up ZF')
        transpose = channel.H.transpose(1, 2)
        hermitian = transpose.conj()
        gram_matrix = torch.matmul(channel.H, hermitian)
        gram_inverse = torch.linalg.inv(gram_matrix)
        _, n_ants, n_ues = hermitian.size()
        self.P = torch.matmul(hermitian, gram_inverse)

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, value):
        self._P = value


class ZF_TIME(Precoder):
    def calculate_precoder(self, channel):
        logger.info(f'Setting up ZF')
        transpose = channel.H.transpose(1, 2)
        hermitian = transpose.conj()
        gram_matrix = torch.matmul(channel.H, hermitian)
        gram_inverse = torch.linalg.inv(gram_matrix)
        _, n_ants, n_ues = hermitian.size()
        P_fd = torch.matmul(hermitian, gram_inverse)

        # Take the IFFT for each stream...

    def use(self):
        """Overload the use"""
        # MAke time domain

        # Convolve with the precoders.

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, value):
        self._P = value


class MRT(Precoder):
    def calculate_precoder(self, channel):
        transpose = channel.H.transpose(1, 2)
        self.P = transpose.conj()
        self.P = torch.exp(self.P.angle()*1j)

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, value):
        self._P = value


class Linear(Precoder):
    def __init__(self, n_total_subcarriers=4096, n_ants=64, n_ues=2, device=[], learning_rate=0.5):
        self.P = torch.randn((n_total_subcarriers, n_ants, n_ues), device=device, dtype=torch.cfloat,
                             requires_grad=True)  # Starting guess for antennas
        self.learning_rate = learning_rate

    def backprop_update(self, loss):
        loss.backward()
        with torch.no_grad():
            self.P -= self.learning_rate * self.P.grad
            self.P.grad = None
