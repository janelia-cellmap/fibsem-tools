from typing import Dict, Union, List
from pathlib import Path

JSON = Union[Dict[str, "JSON"], List["JSON"], str, int, float, bool, None]
Attrs = Dict[str, JSON]
PathLike = Union[Path, str]
