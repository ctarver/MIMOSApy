import abc
import copy
from dataclasses import dataclass

import numpy as np
import torch
from scipy.constants import speed_of_light as c
from scipy.io import loadmat

from pytorch_mimo.Signal import Signal, Domain


@dataclass
class Channel(abc.ABC):
    """Channel superclasss"""
    channel_required_domain: Domain
    f_c: int
    noise_std: float
    device: torch.device

    @property
    @abc.abstractmethod
    def H(self):
        """Abstract property for channel matrix."""

    def use(self, x_in: Signal) -> Signal:
        x = copy.copy(x_in)
        x.match_this(self.channel_required_domain)
        hx = torch.matmul(self.H, x.tensor)
        _, n_streams, _ = hx.size()
        # n = torch.normal(0, std=self.noise_std, size=hx.size(), device=self.device)
        y = hx# + n
        s_out = Signal(y, name='Channel Out', sample_rate=x.sample_rate, domain=self.channel_required_domain,
                       n_symbols=x.n_symbols, fft_size=x.fft_size, n_streams=n_streams, rrc_taps=x.rrc_taps,
                       device=self.device)
        return s_out

    @property
    def wavelength(self):
        return c / self.f_c


@dataclass
class Random(Channel):
    n_scs: int
    n_ues: int
    n_ants: int

    def __post_init__(self):
        self.H = torch.randn((self.n_scs, self.n_ues, self.n_ants), dtype=torch.cfloat)


@dataclass
class LOS_2D(Channel):
    n_ants: int
    ue_azimuth: list[int]
    sc_spacing: int
    n_total_scs: int

    @property
    def H(self):
        return self._H

    @H.setter
    def H(self, val):
        self._H = val

    def __post_init__(self):
        self.H = self.create_los(np.asarray(zip(self.ue_azimuth, self.ue_elevation)))

    def create_los(self, angle):
        lambda_center = self.wavelength
        H = np.zeros((self.n_total_scs, angle.size, self.n_ants))
        for i_sc in range(self.n_total_scs):
            n_scs_from_center = i_sc - self.n_total_scs / 2
            this_frequency = self.f_c + (n_scs_from_center * self.sc_spacing)
            this_wavelength = c / this_frequency
            # Each row needs a cos theta
            # Each column gets 0_n_ants-1
            ant_index = np.asmatrix(np.arange(0, self.n_ants))
            ue_angle = np.asmatrix(np.cos(angle * np.pi / 180))
            H[i_sc, :, :] = ue_angle.T * ant_index
        final_H = np.exp(-1j * np.pi * lambda_center / this_wavelength * H)
        return torch.from_numpy(np.complex64(final_H))

@dataclass
class Quadriga(Channel):
    n_ants: int
    ue_azimuth: list[int]
    sc_spacing: int
    n_total_scs: int
    device: torch.device
    channel: str

    @property
    def H(self):
        return self._H

    @H.setter
    def H(self, val):
        self._H = val

    def __post_init__(self):
        if self.channel == 'los':
            H_mat = loadmat('../pytorch_mimo/quadriga_pregenerated/quadriga_los.mat')
        elif self.channel == '70_100':
            H_mat = loadmat('../pytorch_mimo/quadriga_pregenerated/rma_70_100.mat')
        elif self.channel == '71_100':
            H_mat = loadmat('../pytorch_mimo/quadriga_pregenerated/rma_71_100.mat')
        elif self.channel == '75_100':
            H_mat = loadmat('../pytorch_mimo/quadriga_pregenerated/rma_75_100.mat')
        elif self.channel == '90_100':
            H_mat = loadmat('../pytorch_mimo/quadriga_pregenerated/rma_90_100.mat')
        elif self.channel == 'rma':
            H_mat = loadmat('../pytorch_mimo/quadriga_pregenerated/quadriga_rma.mat')
        elif self.channel == 'nlos':
            H_mat = loadmat('../pytorch_mimo/quadriga_pregenerated/quadriga_umi_nlos.mat')
        elif self.channel == 'argos':
            H_mat = loadmat('../pytorch_mimo/argos.mat')

        full_H = H_mat['H']
        max_H = np.max(full_H)
        full_H = 1 * full_H / max_H
        full_H = np.complex64(full_H)

        if self.channel == '70_100' or '71_100' or '90_100':
            self.H = torch.from_numpy(full_H).to(self.device)
        else:
            self.H = torch.from_numpy(full_H[:, self.ue_azimuth, :]).to(self.device)


@dataclass
class RENEW(Channel):

    @property
    def H(self):
        return self._H

    @H.setter
    def H(self, val):
        self._H = val

    def __post_init__(self):
        H_mat = loadmat('../pytorch_mimo/matlab_channel.mat')

        full_H = H_mat['channel']
        full_H = 1 * full_H
        full_H = np.complex64(full_H)
        full_H = np.moveaxis(full_H, [0, 1, 2], [1, 2, 0])
        self.H = torch.from_numpy(full_H[:, :, :]).to(self.device)

@dataclass
class RayTrace(Channel):
    n_ants: int
    ue_index: list[int]
    sc_spacing: int
    n_total_scs: int
    device: torch.device
    channel: str

    @property
    def H(self):
        return self._H

    @H.setter
    def H(self, val):
        self._H = val

    def __post_init__(self):
        #H_mat = loadmat('../pytorch_mimo/ray_tracing45_easy90.mat')
        #full_H = H_mat['new_H']
        H_mat = loadmat('../pytorch_mimo/ray_tracing45.mat')
        full_H = H_mat['H']
        full_H = np.complex64(full_H)
        this_H = full_H[:, self.ue_index, :]
        max_H = np.max(this_H)
        this_H = 1 * this_H / max_H
        self.H = torch.from_numpy(this_H).to(self.device)

@dataclass
class LOS(Channel):
    n_ants: int
    ue_azimuth: list[int]
    sc_spacing: int
    n_total_scs: int
    device: torch.device

    @property
    def H(self):
        return self._H

    @H.setter
    def H(self, val):
        self._H = val

    def __post_init__(self):
        self.H = self.create_los(np.asarray(self.ue_azimuth))

    def create_los(self, angle):
        lambda_center = self.wavelength
        H = np.zeros((self.n_total_scs, angle.size, self.n_ants))
        for i_sc in range(self.n_total_scs):
            n_scs_from_center = i_sc - self.n_total_scs / 2
            this_frequency = self.f_c + (n_scs_from_center * self.sc_spacing)
            this_wavelength = c / this_frequency
            # Each row needs a cos theta
            # Each column gets 0_n_ants-1
            ant_index = np.asmatrix(np.arange(0, self.n_ants))
            ue_angle = np.asmatrix(np.cos(angle * np.pi / 180))
            H[i_sc, :, :] = ue_angle.T * ant_index
        final_H = np.exp(-1j * np.pi * lambda_center / this_wavelength * H)
        return torch.from_numpy(np.complex64(final_H)).to(self.device)


if __name__ == "__main__":
    my_channel = LOS(n_ants=64, ue_azimuth=[20], sc_spacing=15e3, n_total_scs=1200)
    1 + 1
