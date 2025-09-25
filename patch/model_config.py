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
    f_m: float
    homophily: float
    tau: float
    lfm_global: _LFMValidator = _LFMValidator()
    lfm_tc: _LFMValidator = _LFMValidator() # Validate the local link formation mechanism
    realization: int

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        """Creates a ModelConfig from a dictionary.

        This method automatically detects whether the dictionary uses
        parameter names (N, h_m, h_M) or (n, h_mm, h_MM)
        and converts them to the internal representation.

        Parameters
        ----------
        d : Dict[str, Any]
            The dictionary containing the configuration.

        Returns
        -------
        ModelConfig
            The configuration object.
        """
        d = d.copy()  # Don't modify the original dict

        # Detect and convert netin 2.x parameter names to internal names
        if "n" in d:
            d["N"] = d.pop("n")
        if "k" in d:
            d["m"] = d.pop("k")

        # Handle homophily parameters - detect version and convert
        if "h_mm" in d and "h_MM" in d:
            # netin 2.x format
            assert d["h_mm"] == d["h_MM"], "Homophily values must be identical."
            d["homophily"] = d.pop("h_mm")
            d.pop("h_MM")
        elif "h_m" in d and "h_M" in d:
            # netin 1.x format (legacy)
            assert d["h_m"] == d["h_M"], "Homophily values must be identical."
            d["homophily"] = d.pop("h_m")
            d.pop("h_M")
        # If neither format is present, assume homophily is already in the dict

        return cls(
            N=d["N"],
            m=d["m"],
            f_m=d["f_m"],
            homophily=d["homophily"],
            tau=d["tau"],
            lfm_global=d["lfm_global"],
            lfm_tc=d["lfm_tc"],
            realization=d["realization"]
        )

    def to_dict(self, stringify: bool = False, patch_model: bool = False)\
            -> Dict[str, Any]:
        """Converts the configuration to a dictionary.

        Parameters
        ----------
        stringify : bool, optional
            Whether to convert LFM enums to strings, by default False
        patch_model : bool, optional
            Whether to convert to parameter names used by PATCHModel,
            by default False
            If `True`, the dictionary uses parameter names (n, h_mm, h_MM).
            If `False`, it uses internal names (N, h_m, h_M).

        Returns
        -------
        Dict[str, Any]
            The configuration dictionary.
        """
        d = asdict(self)
        h = self.homophily

        # Handle homophily parameter names based on version
        if patch_model:
            d["h_MM"] = h
            d["h_mm"] = h
            del d["homophily"]

            d["n"] = d["N"]
            del d["N"]

            d["k"] = d["m"]
            del d["m"]

        if stringify:
            d["lfm_global"] = self.lfm_global.value
            d["lfm_tc"] = self.lfm_tc.value
        return d
