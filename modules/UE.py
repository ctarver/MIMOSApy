from dataclasses import dataclass


@dataclass
class UE:
    index: int  # index of UE in the experiment
    azimuth: int  # degrees
    distance: int  # meters

    def make_data(self):

    # Method for each user to make data.

    def rx(self, channel):
        # Main method
        this_rx = self.agc(channel[:, :, :])  # TODO. ADD INDEX

    def calculate_evm(self):
        # Select the in band subcarriers
        # Calculate error against s.
        pass

    def calculate_ber(self):
        pass
