from typing import Dict, Union, List

JSON = Union[Dict[str, "JSON"], List["JSON"], str, int, float, bool, None]
Attrs = Dict[str, JSON]
