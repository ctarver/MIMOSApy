import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
from scipy.constants import speed_of_light as c


class FD_DPD(object):
    def __init__(self, n_total_subcarriers=600, n_data_subcarriers=100,
                 learning_rate=1, n_epochs=600):
        self.n_symbols = 10
        self.n_total_subcarriers = n_total_subcarriers
        self.n_data_subcarriers = n_data_subcarriers
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.loss_values = []
        self.device = torch.device("cpu")
        self.s = self.create_data()
        self.x = []
        self.P = []

    def learn_P(self):
        # Can we learn a linear operation from each subcarrier to each subcarrier to do DPD?
        s = self.s  # Copy of the ideal data
        P = torch.randn((self.n_total_subcarriers, self.n_total_subcarriers), device=self.device,
                        dtype=torch.cfloat, requires_grad=True)
        for t in range(self.n_epochs):
            L = P * 0.01
            x = torch.matmul(L, s)
            loss = self.calculate_loss(x)
            if t % 100 == 99:
                print(t, loss.item())
            loss.backward()
            with torch.no_grad():
                P -= self.learning_rate * P.grad
                P.grad = None
        self.x = x  # Save final value in object
        self.P = L

    def learn_x(self):
        s = self.s  # Copy of the ideal data
        x = torch.randn((self.n_total_subcarriers, 1, 1), device=self.device, dtype=torch.cfloat,
                        requires_grad=True)  # Starting guess for antennas
        for t in range(self.n_epochs):
            loss = self.calculate_loss(x)
            if t % 100 == 99:
                print(t, loss.item())
            loss.backward()
            with torch.no_grad():
                x -= self.learning_rate * x.grad
                x.grad = None
        self.x = x  # Save final value in object

    def calculate_loss(self, x):
        x_hat = self.use_pas(x)
        ue_loss = abs(x_hat - self.s).pow(2).sum()
        power_constraint = abs(x_hat).pow(2).sum()
        # peak_constraint = torch.max(abs(x))
        # antenna_aclr_constraint = torch.max(20*torch.log10(abs(x_hat[301:-301, :, :]))) + 1000
        # loss = torch.add(ue_loss, power_constraint)
        loss = torch.add(torch.add(torch.add(ue_loss, 0), 0), 0)
        self.loss_values.append(loss.item())
        return loss

    def plot_learning(self):
        plt.semilogy(abs(np.asarray(self.loss_values)))
        plt.show()

    def plot_antenna_data(self, str='Antenna FD Data', i_symbol=0):
        # Send through pas
        x_hat = self.use_pas(self.x)
        antenna_i = x_hat[:, i_symbol]
        plt.plot(20 * numpy.log10(np.fft.fftshift(abs(antenna_i.detach().numpy()))), label=str)
        plt.xlabel('Subcarrier Index')
        plt.ylabel('Amplitude (dB)')
        plt.title('PA Output for 1 Symbol')
        plt.grid(True)
        plt.legend(loc="upper left")

    def plot_user_data(self, i_ue=0):
        s_pred = self.use_mimo(self.x, self.H)
        user_i = s_pred[:, i_ue, 0]
        plt.plot(20 * numpy.log10(abs(user_i.detach().numpy())))
        plt.xlabel('Subcarrier Index')
        plt.ylabel('Amplitude (dB)')
        plt.title('User FD Data')
        plt.grid(True)
        plt.show()

    @property
    def wavelength(self):
        return c / self.f_c

    def use_pas(self, x):
        # Take IFFT of each...
        x_time = torch.fft.ifft(x, n=None, dim=0, norm='ortho')
        # PAss through this MP.
        x_mp = 0.8 * x_time + 0.05 * x_time * (abs(x_time) ** 2) + 0.05 * x_time * (abs(x_time) ** 4)

        # Convert back to Frequency Domain.
        x_out = torch.fft.fft(x_mp, n=None, dim=0, norm='ortho')
        return x_out

    def create_data(self):
        alphabet = np.array([1 + 1j, 1 - 1j, -1 - 1j, -1 + 1j])
        fft_size = self.n_total_subcarriers
        n_sc = self.n_data_subcarriers

        # Create the user data
        s = torch.zeros((fft_size, self.n_symbols), dtype=torch.cfloat)
        index = np.random.randint(0, 4, [int(n_sc / 2), self.n_symbols])
        s[1:int(n_sc / 2) + 1, :] = torch.from_numpy(alphabet[index])
        index = np.random.randint(0, 4, [int(n_sc / 2), self.n_symbols])
        s[-int(n_sc / 2):, :] = torch.from_numpy(alphabet[index])
        return s


if __name__ == "__main__":
    dpd = FD_DPD()
    # Test without DPD
    dpd.x = dpd.s
    dpd.plot_antenna_data('No DPD')

    # dpd.learn_x()
    dpd.learn_P()
    dpd.plot_antenna_data('Training Data')

    # Test with new data.
    dpd.s = dpd.create_data()
    x = torch.matmul(dpd.P, dpd.s)
    loss = dpd.calculate_loss(x)
    dpd.x = x
    dpd.plot_antenna_data('Testing Data')
    plt.show()
    dpd.plot_learning()
