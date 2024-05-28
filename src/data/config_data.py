from dataclasses import dataclass, field

from src.features.transformations import Transformation


@dataclass(kw_only=True)
class DataConfigBase:
    """
    Base class for data configuration.

    Creates a list of Transformation objects from callables in the transformations list
    post-init.
    """

    transformations: list[Transformation] = field(default_factory=list)

    def __post_init__(self):
        self.transformations = [
            Transformation(function=t[0], kwargs=t[1])
            if isinstance(t, tuple)  # tuples with kwargs need to be unpacked
            else Transformation(function=t)  # functions without kwargs
            for t in self.transformations
        ]
