import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal
import torch

from pytorch_mimo.Enums import *

logger = logging.getLogger('mimo')
import copy


@dataclass
class Signal:
    tensor: torch.Tensor
    name: str
    sample_rate: float
    domain: Domain
    n_symbols: int
    fft_size: int
    n_streams: int
    rrc_taps: torch.Tensor
    cp_length: int = 36 # 144 * 2  # For a 4.7 us CP at 4096 FFT. 7% overhead
    use_windows: bool = True
    window_length: int = 32
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @classmethod
    def create(cls, dataflow, name, device):
        "Creates OFDM data"
        rrc_taps = cls.generate_rrc(cls.window_length)
        np.random.seed(1)
        alphabet = np.array([1 + 1j, 1 - 1j, -1 - 1j, -1 + 1j])
        s = torch.zeros((dataflow.n_total_scs, dataflow.n_ues, dataflow.n_symbols), dtype=torch.cfloat, device=device)
        index = np.random.randint(0, 4, [int(dataflow.n_data_scs / 2), dataflow.n_ues, dataflow.n_symbols])
        s[1:int(dataflow.n_data_scs / 2) + 1, :, :] = torch.from_numpy(alphabet[index])
        index = np.random.randint(0, 4, [int(dataflow.n_data_scs / 2), dataflow.n_ues, dataflow.n_symbols])
        s[-int(dataflow.n_data_scs / 2):, :, :] = torch.from_numpy(alphabet[index])
        fs = dataflow.n_total_scs * dataflow.sc_spacing
        return cls(tensor=s, name=name, sample_rate=fs, domain=Domain.FREQ, n_symbols=dataflow.n_symbols,
                   fft_size=dataflow.n_total_scs, n_streams=dataflow.n_ues, rrc_taps=rrc_taps, device=device)

    @classmethod
    def create_without_df(cls, name, n_data_scs, n_total_scs, sc_spacing, n_ues, n_symbols, device, my_seed=1):
        "Creates OFDM data"
        rrc_taps = cls.generate_rrc(cls.window_length)
        np.random.seed(my_seed)
        alphabet = np.array([1 + 1j, 1 - 1j, -1 - 1j, -1 + 1j])
        s = torch.zeros((n_total_scs, n_ues, n_symbols), dtype=torch.cfloat, device=device)
        index = np.random.randint(0, 4, [int(n_data_scs / 2), n_ues, n_symbols])
        s[1:int(n_data_scs / 2) + 1, :, :] = torch.from_numpy(alphabet[index])
        index = np.random.randint(0, 4, [int(n_data_scs / 2), n_ues, n_symbols])
        s[-int(n_data_scs / 2):, :, :] = torch.from_numpy(alphabet[index])
        fs = n_total_scs * sc_spacing
        return cls(tensor=s, name=name, sample_rate=fs, domain=Domain.FREQ, n_symbols=n_symbols,
                   fft_size=n_total_scs, n_streams=n_ues, rrc_taps=rrc_taps, device=device)

    def match_this(self, desired_domain: Domain):
        if self.domain == desired_domain:
            return
        elif self.domain == Domain.FREQ and desired_domain == Domain.TIME:
            self.convert_fd_to_td()
        elif self.domain == Domain.TIME and desired_domain == Domain.FREQ:
            self.convert_td_to_fd()
        else:
            raise Exception(f'{desired_domain} was not a valid desired_domain')

    def copy(self):
        """ Makes a new copy of the signal."""
        return copy.copy(self)

    def normalize_to_this_amp(self, amp):
        # What is the current max?
        the_max = torch.max(torch.abs(self.tensor))
        scale_factor = amp / the_max
        self.tensor = scale_factor * self.tensor
        return scale_factor

    def normalize_to_this_rms(self, this_rms=1):
        rms = self.calculate_rms()

        # Calculate scale factor between max rms and desired
        max_rms = torch.max(rms)
        n_samples = 512
        scale_factor = 0.65#torch.sqrt(50*n_samples * torch.pow(10, (this_rms-30)/10)) / torch.norm(self.tensor)
        self.tensor = self.tensor * scale_factor
        return scale_factor

    def extract_stream(self, index: int):
        desired_data = self.tensor[:, index]
        return Signal(desired_data, self.name, self.sample_rate, self.domain, self.n_symbols, self.fft_size,
                      self.n_streams, self.rrc_taps, self.cp_length, self.use_windows, self.window_length)

    def lpf(self):
        self.match_this(Domain.TIME)
        x = self.tensor
        input_length, _ = x.size()
        n_taps = 101
        b = scipy.signal.firls(n_taps, (0, 18e6, 20e6, self.sample_rate), (1, 1, 0, 0), fs=2 * self.sample_rate)
        npconv_result = np.array([np.convolve(xi, b, mode='full') for xi in np.transpose(x)])
        npconv_result = npconv_result.transpose()
        npconv_result = npconv_result[int((n_taps - 1) / 2):, :]
        npconv_result = npconv_result[:input_length]
        self.tensor = torch.tensor(npconv_result, dtype=torch.cfloat)

    def calculate_rms(self):
        squared = torch.pow(torch.abs(self.tensor), 2)
        mean_squared = torch.mean(squared, dim=0)
        rms_volt = torch.sqrt(mean_squared)
        # rms_power = torch.pow(rms_volt, 2) / 50
        # rms_dbm = 10 * torch.log10(rms_power) + 30
        return rms_volt

    def calculate_total_power(self):
        rms = self.calculate_rms()
        return torch.sum(rms)

    def calculate_papr(self, i_stream=0):
        self.match_this(Domain.TIME)

        # What is the peak power?
        powers = torch.pow(torch.abs(self.tensor), 2)
        peak_power = torch.max(powers[:, i_stream])
        avg_power = torch.mean(powers[:, i_stream])
        papr = 10 * torch.log10(peak_power / avg_power)
        print(f'PAPR of {papr} for {self.name} stream {i_stream}')
        return papr

    def calculate_amp(self):
        # What is the peak power?
        powers = torch.pow(torch.abs(self.tensor), 2)
        peak_powers = torch.max(powers, dim = 0)
        print(f'Peak Amp of {peak_powers}')
        return peak_powers

    def calculate_aclr(self, mode='max', trp=[]):
        """Calculates the ACLR for all streams in the signal...."""
        ibw = 18e6
        offset = 20e6
        bandpower = np.zeros((self.n_streams, 3))
        for j_stream in np.arange(self.n_streams):
            start = -offset - ibw / 2
            for i in np.arange(3):
                end = start + ibw
                bandpower[j_stream, i] = self.bandpower(start, end, index=j_stream)
                start = start + offset

        if mode == 'max':
            # What is the max power in all streams?
            mostest_power = np.max(bandpower)

            # Convert to dBc dB dBc
            bandpower = bandpower - mostest_power

        elif mode == 'trp':
            max_trp = 64
            powerdb = 10 * np.log10(max_trp)
            bandpower = bandpower - powerdb
        else:
            bandpower[:, 0] = bandpower[:, 0] - bandpower[:, 1]
            bandpower[:, 2] = bandpower[:, 2] - bandpower[:, 1]

        return bandpower

    def calculate_evm(self, ref_signal, i_stream=0):
        self.match_this(Domain.FREQ)
        ref_signal.match_this(Domain.FREQ)
        # TODO. Make data subcarriers part of class?
        n_data = 300
        y_agc = self.ue_agc(self.tensor)
        error = self.s.tensor - y_agc
        error_mag = torch.sum(torch.abs(error[-150:150]))
        signal_mag = torch.sum(torch.abs(self.s.tensor))
        evm_db = 20 * torch.log10(error_mag / signal_mag)
        evm_percent = 100 * error_mag / signal_mag
        return evm_db, evm_percent
        pass

    def plot_psd(self, **kwargs):
        """" Plots the power spectral density of the full signal"""
        self.match_this(Domain.TIME)
        np_array = self.tensor.cpu().detach().numpy()
        if 'index' in kwargs:
            index = kwargs['index']
            this_stream = np_array[:, index]
        else:
            this_stream = np_array[:, 0]

        plt.psd(this_stream, Fs=self.sample_rate, NFFT=1024)
        plt.ylim(-180, -70)  # set the ylim to bottom, top
        plt.title(self.name)

        if 'results_dir' in kwargs:
            results_dir = kwargs['results_dir']
            plt.savefig(f'{results_dir}/psd_{self.name}.png')
        plt.show()

    def plot_fft(self, **kwargs):
        """ Plots the raw FD Domain data for 1 symbol """
        self.match_this(Domain.FREQ)
        np_array = self.tensor.cpu().detach().numpy()

        data = np.fft.fftshift(np.abs(np_array[:, 0, 0]))
        if 'index' in kwargs:
            stream = kwargs['index']
            data = np.fft.fftshift(np.abs(np_array[:, stream, 0]))

        plt.plot(20 * np.log10(data))
        # for i_stream, this_data_stream in enumerate(test):
        #    plt.subplot(1, self.n_streams, i_stream+1)
        #    plt.plot(np.fft.fftshift(np.abs(this_data_stream)))
        plt.xlabel('Subcarrier')
        plt.ylabel('Magnitude (dB)')
        plt.title(self.name)
        plt.grid(True)
        if 'results_dir' in kwargs:
            results_dir = kwargs['results_dir']
            plt.savefig(f'{results_dir}/fft_{self.name}.png')
        plt.show()

    def plot_constellation(self, i_stream=0, i_sym=0, **kwargs):
        self.match_this(Domain.FREQ)
        scale = self.normalize_to_this_rms(0.635)
        print(f'Scale = {scale}')
        np_array = self.tensor[:, i_stream, i_sym].cpu().detach().numpy()
        plt.plot(np_array.real, np_array.imag, 'o')
        plt.ylim([-170, -60])
        with open('const.dat', 'w') as f:
            nl = '\n'
            for value in np_array:
                f.write(f'{value.real}  {value.imag} {nl}')
        plt.grid(True)
        plt.title(self.name)
        if 'results_dir' in kwargs:
            results_dir = kwargs['results_dir']
            plt.savefig(f'{results_dir}/constellation_{self.name}.png')
        plt.show()

    def plot_iq(self, **kwargs):
        self.match_this(Domain.TIME)
        np_array = self.tensor.cpu().detach().numpy()
        n_samples, _ = np_array.shape
        period = 1 / self.sample_rate
        time = period * np.arange(n_samples) / 1e-6
        for i_stream, this_data_stream in enumerate(np_array.transpose()):
            plt.subplot(1, self.n_streams, i_stream + 1)
            plt.plot(time, this_data_stream.real)
            plt.plot(time, this_data_stream.imag)
            plt.grid(True)
            plt.xlabel('time (us)')

        if 'results_dir' in kwargs:
            results_dir = kwargs['results_dir']
            plt.savefig(f'{results_dir}/iq_{self.name}.png')
        plt.show()

    def convert_td_to_fd(self):
        logger.debug('Converting from TD to FD')
        # FD should be bins, users, symbols
        # Take each symbol for each user and place it..

        sym_length = self.fft_size + self.cp_length
        g = torch.reshape(self.tensor, (self.n_symbols, sym_length, self.n_streams))
        h = torch.swapaxes(g, 0, 1)
        a = torch.swapaxes(h, 1, 2)

        b = self.remove_cp(a)

        self.tensor = torch.fft.fft(b, dim=0, norm='ortho')
        self.domain = Domain.FREQ

    def convert_fd_to_td(self):
        logger.debug('Converting from FD to TD')
        x = torch.fft.ifft(self.tensor, dim=0, norm='ortho')
        x_cp = self.add_cyclic_prefix(x)
        x_win = self.add_windows(x_cp)
        # FD Data is subcarriers, users, symbols.
        self.domain = Domain.TIME

        sym_length = self.fft_size + self.cp_length
        self.tensor = torch.zeros((sym_length * self.n_symbols, self.n_streams), dtype=torch.cfloat, device=self.device)

        for i_sym in np.arange(self.n_symbols - 1):
            current_index = (i_sym) * sym_length
            end_index = current_index + sym_length + self.window_length
            self.tensor[current_index:end_index, :] = self.tensor[current_index:end_index, :] + x_win[:, :, i_sym]

        if self.n_symbols == 1:
            end_index = self.window_length

        # Last symbol is special. Doesn't include final suffix window
        self.tensor[end_index - self.window_length:, :] = self.tensor[end_index - self.window_length:, :] + x_win[
                                                                                                            :sym_length,
                                                                                                            :,
                                                                                                            self.n_symbols - 1]

    def bandpower(self, fmin, fmax, index=0):
        self.match_this(Domain.TIME)
        x = self.tensor[:, index].cpu().detach().numpy()
        f, Pxx = scipy.signal.periodogram(x, fs=self.sample_rate, window='hanning', nfft=1024)
        f = np.fft.fftshift(f)
        Pxx = np.fft.fftshift(Pxx)
        ind_min = scipy.argmax(f > fmin) - 1
        ind_max = scipy.argmax(f > fmax) - 1
        power = scipy.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])
        powerdb = 10 * np.log10(power)
        return powerdb

    @staticmethod
    def generate_rrc(n_taps, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        rrc_taps = torch.zeros((n_taps, 1), device=device)
        for i, _ in enumerate(rrc_taps):
            rrc_taps[i] = torch.tensor(0.5 * (1 - np.sin(np.pi * (n_taps - 1 - 2 * i) / (2 * n_taps))), device=device)
        return rrc_taps

    def add_windows(self, x_cp):
        # Assumes we just converted to TD and are still in 3d.
        # For each symbol. add window to 1st n samples and last n
        tap_matrix = torch.zeros(x_cp[:self.window_length, :, :].size(), device=self.device)
        opposite_tap_matrix = torch.zeros(x_cp[:self.window_length, :, :].size(), device=self.device)
        for i, _ in enumerate(tap_matrix):
            tap_matrix[i, :, :] = self.rrc_taps[i] * torch.ones(tap_matrix[i, :, :].size(), device=self.device)
            opposite_tap_matrix[i, :, :] = self.rrc_taps[self.window_length - i - 1] * torch.ones(
                tap_matrix[i, :, :].size(), device=self.device)

        x_win = torch.clone(x_cp)
        x_win[:self.window_length, :, :] = x_cp[:self.window_length, :, :] * tap_matrix
        x_win[-self.window_length:, :, :] = x_cp[-self.window_length:, :, :] * opposite_tap_matrix
        return x_win

    def add_cyclic_prefix(self, x):
        total_added_length = self.cp_length + self.window_length
        total_td_length = self.fft_size + total_added_length

        out = torch.zeros((total_td_length, self.n_streams, self.n_symbols), dtype=torch.cfloat, device=self.device)

        # Add in the main data
        out[self.cp_length:self.cp_length + self.fft_size, :, :] = x

        # Add in the CP
        out[:self.cp_length, :, :] = x[-self.cp_length:, :, :]

        # Add in the cyclic suffix
        out[-self.window_length:, :, :] = x[:self.window_length, :, :]
        return out

    def remove_cp(self, x):
        out = torch.zeros((self.fft_size, self.n_streams, self.n_symbols), dtype=torch.cfloat, device=self.device)
        out = x[self.cp_length:self.cp_length + self.fft_size, :, :]
        return out

    def add_vues(self, n_vues):
        user_tensor = self.tensor
        n_ues = self.n_streams
        v_ue_tensor = torch.zeros((self.fft_size, n_vues + n_ues, self.n_symbols),
                                  dtype=torch.complex64, device=self.device)
        v_ue_tensor[:, :n_ues, :] = user_tensor

        return Signal(v_ue_tensor, 'v_ue_input', self.sample_rate, self.domain,
                      self.n_symbols, self.fft_size, n_vues + n_ues, self.rrc_taps,
                      self.cp_length, self.use_windows, self.window_length)

    def __sub__(self, other):
        """Method to subtract two signals"""
        # Are they the same domain?
        # Do they have same number of strems?
        # Do subtraction.
        pass


if __name__ == "__main__":
    random.seed(0)
    my_signal = Signal.create(name='Test')
    my_signal.plot_fft()
    1 + 1
