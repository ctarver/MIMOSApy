from dataclasses import dataclass
from enum import Enum, auto


class PaTypes(Enum):
    LINEAR = auto()
    GMP = auto()
    OLD = auto()


class PrecoderTypes(Enum):
    ZF = auto()
    MRT = auto()
    ZF_TIME = auto()


class ChannelTypes(Enum):
    LOS_2D = auto()
    LOS = auto()
    RANDOM = auto()


class Domain(Enum):
    TIME = auto()
    FREQ = auto()


@dataclass
class Results:
    time: []
    name: str
    loss_function: str
    final_loss: float
    n_neurons: int
    evm: float
    worst_user_aclr: float
    worst_aclr: float
