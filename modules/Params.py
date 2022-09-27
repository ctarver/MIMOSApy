import itertools as it
from datetime import datetime

import numpy as np
import torch

from pytorch_mimo.Enums import *


# To see the options for any of the enumerated classes, see the Enums.py file


@dataclass
class Params:
    # Standard params.
    n_data_scs: int = 1200
    n_total_scs: int = 4096
    sc_spacing: int = 15e3
    n_symbols: int = 2
    n_ues: int = 2
    use_vues: bool = True

    n_ants: int = 64
    n_rows: int = 1  # ULA if 1.
    f_c: int = 3.5e9
    ue_azimuth: np.ndarray = np.array([70, 85, 90, 124, 134, 90, 145])
    ue_elevation: np.ndarray = np.array([0, 10, -10, 0, 0, 0, 0, 0])

    # Channel Settings
    channel_type: ChannelTypes = ChannelTypes.LOS
    noise_std: float = 0.001
    channel_required_domain: Domain = Domain.FREQ

    # Precoder Settings
    precoder_type: PrecoderTypes = PrecoderTypes.MRT
    precoder_required_domain: Domain = Domain.FREQ

    # PA Settings. DOES NOTHING NOW
    pa_type: PaTypes = PaTypes.OLD
    pa_required_domain: Domain = Domain.TIME
    n_streams: int = n_ants
    order: int = 5
    memory: int = 2
    step: int = 2
    lag: int = 0
    variance: float = 0.0001

    # GMP DPD Settings
    # This conflicts with the name of data for PA. Prob need to make a struct that each expects.

    # NN for NN Tests
    n_layers: int = 1  # Does nothing right now...
    n_hidden_neurons: int = 80
    n_memory: int = 1  # DONT TOUCH
    lr: float = 0.0000001  # learning rate
    n_epochs: int = 1000

    use_gpu: bool = True  # Use GPU if available

    def __post_init__(self):
        # Sanitize data
        self.ue_azimuth = self.ue_azimuth[0:self.n_ues]
        self.ue_elevation = self.ue_elevation[0:self.n_ues]

        if self.use_vues:
            self.calculate_vue_angles()
        else:
            self.n_vues = 0

            # Other setups
        self.time = datetime.now().strftime("%Y%m%d-%H%M%S")
        # Pytorch
        if self.use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

    def calculate_vue_angles(self):
        # Generating the iterator to calculate number of elements
        combinations = it.product(self.ue_azimuth, self.ue_azimuth, self.ue_azimuth)
        n_vue_angles = len(list(combinations))

        # Need to regenerate teh iterator so we can loop over it
        combinations = it.product(self.ue_azimuth, self.ue_azimuth, self.ue_azimuth)
        v_ue_angles = np.zeros(n_vue_angles)
        for i, combination in enumerate(combinations):
            combination = np.deg2rad(combination)
            theta1, theta2, theta3 = combination
            vue_angle_radians = np.arccos(np.cos(theta1) + np.cos(theta2) - np.cos(theta3))
            v_ue_angles[i] = np.rad2deg(vue_angle_radians)

        unique_vue_angles = np.setdiff1d(v_ue_angles, self.ue_azimuth)
        self.n_vues = len(unique_vue_angles)
        self.ue_azimuth = np.append(self.ue_azimuth, unique_vue_angles)


if __name__ == "__main__":
    p = Params()
    total_angle = zip(p.ue_azimuth, p.ue_elevation)
    test = p.__dict__
    1 + 1
