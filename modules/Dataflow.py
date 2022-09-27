import abc
import csv
import logging
import os
import pickle
from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np
import torch

from pytorch_mimo.Channel import Channel
from pytorch_mimo.Enums import *
from pytorch_mimo.Signal import Signal


class Dataflow(metaclass=abc.ABCMeta):
    """Dataflow class is meant to be a super class"""

    def __init__(self):
        os.mkdir(self.results_dir)
        self.setup_logger()
        logging.info(pformat(vars(self)))

    @abc.abstractmethod
    def train(self):
        """Run the dataflow"""

    def calculate_evm(self, y):
        """Calculates evm from user data, y, to intended data, s"""
        self.s.match_this(Domain.FREQ)
        y.match_this(Domain.FREQ)
        y_agc = self.ue_agc(y.tensor)
        y = y_agc[:, :self.n_ues, :]
        s = self.s.tensor[:, :self.n_ues, :]
        error = s - y
        error_mag = torch.sum(torch.abs(error))
        signal_mag = torch.sum(torch.abs(s))
        evm_db = 20 * torch.log10(error_mag / signal_mag)
        evm_percent = 100 * error_mag / signal_mag
        return evm_db, evm_percent

    def ue_agc(self, y):
        y_abs = torch.max(torch.abs(y))
        s_abs = torch.max(torch.abs(self.s.tensor))
        scale_factor = s_abs / y_abs
        y = y * scale_factor
        return y

    def plot_learning(self, **kwargs):
        plt.semilogy(abs(np.asarray(self.loss_values)))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training')
        plt.grid(True)
        if 'results_dir' in kwargs:
            results_dir = kwargs['results_dir']
            plt.savefig(f'{results_dir}/training.png')
        plt.show()

    @staticmethod
    def make_angle_grid(n_angular_bins: int = 300):
        angles = np.linspace(0, 180, n_angular_bins)
        return angles

    def calculate_farfield(self, angles):
        """
        Calculates the farfield azimuth response for the dataflow for a given np array of azimuth angle
            Inputs:
                - Angles is a np array from make_angle_grid()

            Returns:
                - Signal output of the dataflow with the high resolution angle passed in.

         See also:
                - make_angle_grid() can be used to make the angles input
                - the output can be passed to the plot_beamplot() or plot_beamgrid()
        """
        channel_settings = self.__dict__
        channel_settings['channel_type'] = ChannelTypes.LOS
        channel_settings['ue_azimuth'] = angles
        H = Channel.create(**channel_settings)
        return self.use(channel=H)

    def calculate_trp(self, farfield_channel_out: Signal):
        """The goal is to sum the channel output to get the TRP"""
        farfield_channel_out.match_this(Domain.TIME)
        test = farfield_channel_out.calculate_rms()
        n_points = np.size(test.numpy())
        delta_theta = 2 * np.pi / n_points
        return delta_theta * np.sum(test.numpy())

    def plot_beamplot(self, angles, channel_out, **kwargs):
        channel_out.match_this(Domain.FREQ)
        aclr_vs_azimuth = channel_out.calculate_aclr()
        l1 = aclr_vs_azimuth[:, 0].max()
        u1 = aclr_vs_azimuth[:, 2].max()
        worst = max(l1, u1)
        plt.plot(angles, aclr_vs_azimuth[:, 1])  # Plot inband main power in band.
        plt.plot(angles, aclr_vs_azimuth[:, 0])  # Plot L1 Power
        plt.plot(angles, aclr_vs_azimuth[:, 2])  # Plot U1 Power
        plt.ylim((-80, 1))
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Amplitude (dB)')
        plt.grid(True)
        if 'results_dir' in kwargs:
            results_dir = kwargs['results_dir']
            plt.savefig(f'{results_dir}/beamplot.png')
        plt.show()
        return worst

    def plot_beamgrid(self, angles, channel_out):
        channel_out.match_this(Domain.FREQ)
        subcarriers = np.arange(0, self.x.fft_size)
        angle_np = channel_out.tensor.detach().numpy()
        angle_np = angle_np.squeeze()
        angle_np = np.fft.fftshift(np.absolute(angle_np), axes=0)
        angle_np_1_sym = np.squeeze(angle_np[:, :, 0])
        fig1, ax2 = plt.subplots(constrained_layout=True)
        CS = ax2.contourf(angles, subcarriers, 20 * np.log10(angle_np_1_sym), cmap='jet', levels=200, vmax=50, vmin=-70)
        cbar = fig1.colorbar(CS)
        cbar.ax.set_ylabel('Magnitude (dB)')
        cbar.vmin = -70
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Subcarrier')
        plt.show()

    def add_vues(self, s: Signal) -> Signal:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        user_tensor = s.tensor
        v_ue_tensor = torch.zeros((self.n_total_scs, self.n_vues + self.n_ues, self.n_symbols), dtype=torch.complex64,
                                  device=device)
        v_ue_tensor[:, :self.n_ues, :] = user_tensor

        return Signal(v_ue_tensor, 'v_ue_input', self.s_original.sample_rate, self.s_original.domain,
                      self.n_symbols, self.n_total_scs, self.n_vues + self.n_ues, self.s_original.rrc_taps,
                      self.s_original.cp_length, self.s_original.use_windows, self.s_original.window_length)

    def post_process(self):
        self.y.plot_psd(results_dir=self.results_dir)
        self.y.plot_constellation(results_dir=self.results_dir)
        try:
            self.plot_learning(results_dir=self.results_dir)
        except:
            print('No NN Training Data')

        evm_db, evm_percent = self.calculate_evm(self.y)
        logging.info(f'EVM = {evm_db} dB, {evm_percent}%')

        this_aclr = self.y.calculate_aclr()
        logging.info(f'ACLR: user, band\n {this_aclr}')

        angles = self.make_angle_grid(n_angular_bins=400)
        channel_out = self.calculate_farfield(angles)
        worst_aclr = self.plot_beamplot(angles, channel_out, results_dir=self.results_dir)
        # test.plot_beamgrid(angles, channel_out, results_dir)

        # Time, Name, n_users, Loss function, Size, EVM, User ACLR, Worst ACLR
        self.results = [str(self.time), str(type(self).__name__), self.n_ues, self.loss_function, self.best_loss,
                        self.n_hidden_neurons,
                        evm_percent.cpu().detach().numpy(), max(this_aclr[:, 0]), worst_aclr]
        self.save_workspace()
        self.update_csv()

    def setup_logger(self):
        logging.basicConfig(filename=f'{self.results_dir}/test.log', level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler())  # Also take console output

    def update_csv(self):
        with open('../results/table.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(self.results)

    def save_workspace(self):
        pickle.dump(self, open(f'{self.results_dir}/save.p', "wb"))

    @staticmethod
    def load_workspace():
        self = pickle.load(open("save.p", "rb"))
        return self
