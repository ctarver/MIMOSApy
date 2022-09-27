import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pytorch_mimo.Channel import LOS
from pytorch_mimo.Enums import Domain


class NN_Dataflow():

    def make_beamplot(self, signal, name: str = 'beam'):
        n_bins = 500
        angles = self.make_angle_grid(n_angular_bins=n_bins)
        og_channel = self.channel

        self.channel = LOS(f_c=self.f_c, noise_std=0.0001, n_ants=self.n_ants, ue_azimuth=angles,
                           sc_spacing=self.sc_spacing, n_total_scs=self.n_total_scs,
                           channel_required_domain=Domain.FREQ, device=self.device)
        channel_out_signal = self(signal)
        aclr_vs_azimuth = channel_out_signal.calculate_aclr()
        self.write_to_latex_table(angles, aclr_vs_azimuth)
        l1 = aclr_vs_azimuth[:, 0].max()
        main = aclr_vs_azimuth[:, 1].max()
        u1 = aclr_vs_azimuth[:, 2].max()
        worst = max(l1, u1)
        fig = px.line_polar(r=aclr_vs_azimuth[:, 1], theta=angles,
                            range_theta=[0, 180], start_angle=0, direction="counterclockwise",
                            range_r=[-60, 5])
        fig.add_trace(go.Scatterpolar(r=aclr_vs_azimuth[:, 0], theta=angles, mode='lines'))
        fig.show()
        fig.write_html(f"{name}.html")
        self.channel = og_channel

    def make_ideal_beamplot(self, signal):
        n_bins = 500
        angles = self.make_angle_grid(n_angular_bins=n_bins)
        self.channel = LOS(f_c=self.f_c, noise_std=0.0001, n_ants=self.n_ants, ue_azimuth=angles,
                           sc_spacing=self.sc_spacing, n_total_scs=self.n_total_scs,
                           channel_required_domain=Domain.FREQ, device=self.device)
        channel_out_signal = self.use_no_pa(signal)
        aclr_vs_azimuth = channel_out_signal.calculate_aclr()
        l1 = aclr_vs_azimuth[:, 0].max()
        main = aclr_vs_azimuth[:, 1].max()
        u1 = aclr_vs_azimuth[:, 2].max()
        worst = max(l1, u1)
        fig = px.line_polar(r=aclr_vs_azimuth[:, 1], theta=angles,
                            range_theta=[0, 180], start_angle=0, direction="counterclockwise",
                            range_r=[-60, 5])
        fig.add_trace(go.Scatterpolar(r=aclr_vs_azimuth[:, 0], theta=angles, mode='lines'))
        fig.show()
        fig.write_html("beam.html")

    def make_angle_grid(self, n_angular_bins: int = 300):
        angles = np.linspace(0, 180, n_angular_bins)
        return angles

    def write_to_latex_table(self, angles, aclr):
        backslash_char = '\\\\'
        nl = '\n'
        with open('l1.txt', 'w') as f:
            for angle, power in zip(angles, aclr):
                f.write(f'{angle} {power[0]} {backslash_char} {nl}')
        with open('main.txt', 'w') as f:
            for angle, power in zip(angles, aclr):
                f.write(f'{angle} {power[1]} {backslash_char} {nl}')
        with open('u1.txt', 'w') as f:
            for angle, power in zip(angles, aclr):
                f.write(f'{angle} {power[2]} {backslash_char} {nl}')






