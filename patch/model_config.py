"""Defines a dataclass to store the configuration of the PATCH model and validate its configuration.
"""
from typing import Union, Dict, Any
from dataclasses import dataclass, asdict

from netin.models import CompoundLFM

class _LFMValidator:
    """Descriptor to validate the value of the link formation mechanisms.
    """
    _name: str

    def __set_name__(self, owner, name):
        self._name = "_" + name

    def __get__(self, obj: "ModelConfig", _):
        return getattr(obj, self._name)

    def __set__(self, obj: "ModelConfig", value: Union[str, CompoundLFM]):
        """Sets the LFm values.
        If the value is a string, it is converted to a CompoundLFM.
        If the LFM is for triadic closure edges, it must be either
        UNIFORM or identical to the global LFM.

        Parameters
        ----------
        obj : ModelConfig
            The configuration object.
        value : Union[str, CompoundLFM]
            The value to set.

        Raises
        ------
        ValueError
            If the value is not a CompoundLFM.
        ValueError
            If the LFM is for triadic closure edges and
            the value is not UNIFORM or identical to the global LFM.
        """
        if isinstance(value, str):
            value = CompoundLFM[value]
        if not isinstance(value, CompoundLFM):
            raise ValueError(f"Expected a CompoundLFM, got {type(value)}")
        if (self._name == "_lfm_tc") and (not value in (CompoundLFM.UNIFORM, obj.lfm_global)):
            raise ValueError(
                ("CompoundLFM for triadic closure edges must be "
                 f"{CompoundLFM.UNIFORM} or identical to the global LFM "
                 f"({obj.lfm_global.value}); got {value}."))
        setattr(obj, self._name, value)

@dataclass
class ModelConfig:
    """Configuration for the PATCH model.
    """
    N: int
    m: int
    minority_fraction: float
    homophily: float
    tau: float
    lfm_global: _LFMValidator = _LFMValidator()
    lfm_tc: _LFMValidator = _LFMValidator() # Validate the local link formation mechanism
    realization: int

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        """Creates a ModelConfig from a dictionary.

        Parameters
        ----------
        d : Dict[str, Any]
            The dictionary containing the configuration.

        Returns
        -------
        ModelConfig
            The configuration object.
        """
        return cls(
            N=d["N"],
            m=d["m"],
            minority_fraction=d["minority_fraction"],
            homophily=d["homophily"],
            tau=d["tau"],
            lfm_global=d["lfm_global"],
            lfm_tc=d["lfm_tc"],
            realization=d["realization"]
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts the configuration to a dictionary.

        Returns
        -------
        Dict[str, Any]
            The configuration dictionary.
        """
        return asdict(self)
