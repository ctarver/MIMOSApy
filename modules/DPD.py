import abc
from dataclasses import dataclass

from pytorch_mimo.GMP import GMP
from pytorch_mimo.Signal import Signal, Domain


class DPD(abc.ABC):
    @abc.abstractmethod
    def use(self, dpd_input: Signal) -> Signal:
        pass


@dataclass
class ILA_GMP(DPD):
    """Indirect Learning Architecture DPD using GMP models. This class creates and stores an arbitrary number of GMP
    models and feeds data to them for learning."""
    n_streams: int
    order: int
    memory: int
    lag: int
    step: int
    learning_method: str

    def __post_init__(self):
        self.models = [GMP(order=self.order, memory=self.memory, lag=self.lag, step=self.step)
                       for i in range(self.n_streams)]
        self.rms_learned = []

    def use(self, dpd_input: Signal):
        """ Main method for using the GMPs. Calls the use method for all GMPs in this object"""
        dpd_input.match_this(Domain.TIME)

        for gmp in self.models:
            gmp.use(dpd_input)

    def learn(self, pa_input: Signal, pa_output: Signal):
        pa_input.match_this(Domain.TIME)
        pa_output.match_this(Domain.TIME)
        self.rms_learned = pa_input.calculate_rms()

        pa_input_data = pa_input.tensor
        pa_output_data = pa_output.tensor

        for gmp in self.models:
            gmp.learn()


if __name__ == "__main__":
    my_dpd = DPD.create('gmp')
    my_dpd.learn()
    1 + 1
