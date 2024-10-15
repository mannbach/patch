"""Defines a dataclass to store the configuration of the PATCH model and validate its configuration.
"""
from dataclasses import dataclass

from netin.models import CompoundLFM

class _LFMTValidator:
    """Descriptor to validate the value of the local link formation mechanism.
    """
    _name: str

    def __set_name__(self, owner, name):
        self._name = "_" + name

    def __get__(self, obj, type):
        return getattr(obj, self._name)

    def __set__(self, obj, value):
        if not isinstance(value, CompoundLFM):
            raise ValueError(f"Expected a CompoundLFM, got {type(value)}")
        if not value in (CompoundLFM.UNIFORM, obj.lfm_global):
            raise ValueError(
                (f"Expected a CompoundLFM different from {obj.lfm_global} "
                 f"or {CompoundLFM.UNIFORM}, got {value}"))
        setattr(obj, self._name, int(value))

@dataclass
class ModelConfig:
    """Configuration for the PATCH model.
    """
    N: int
    m: int
    minority_fraction: float
    homophily: float
    tau: float
    lfm_global: CompoundLFM
    lfm_tc: _LFMTValidator = _LFMTValidator() # Validate the local link formation mechanism
    realization: int
