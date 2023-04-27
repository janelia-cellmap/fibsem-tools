from typing import (
    Generic,
    List,
    Protocol,
    Mapping,
    Any,
    Tuple,
    TypeVar,
    Union,
    Iterable,
    runtime_checkable,
)
from pydantic.generics import GenericModel

Attr = TypeVar("Attr")


class Node(GenericModel, Generic[Attr]):
    name: str
    attrs: Attr

    class Config:
        extra = "forbid"


ArrAttr = TypeVar("ArrAttr")


class Array(GenericModel, Generic[ArrAttr]):
    name: str
    shape: Tuple[int, ...]
    dtype: str
    attrs: ArrAttr


GAttr = TypeVar("GAttr")
GValue = TypeVar("GValue")


class Group(GenericModel, Generic[GAttr, GValue]):
    values: List[GValue]
    attrs: GAttr


@runtime_checkable
class NodeLike(Protocol):
    # in h5py and zarr, this is a full path relative to the root of the container
    name: str
    attrs: Mapping[str, Any]


class ArrayLike(NodeLike, Protocol):
    shape: Tuple[int, ...]
    dtype: str


class GroupLike(NodeLike, Protocol):
    def values(self) -> Iterable[Union["GroupLike", ArrayLike]]:
        """
        Iterable of the children of this group
        """
        ...

    def create_group(self, name: str, **kwargs) -> "GroupLike":
        ...

    def create_array(
        self, name: str, dtype: Any, chunks: Tuple[int, ...], compressor: Any
    ) -> ArrayLike:
        ...


def build_tree(element: Union[GroupLike, ArrayLike]) -> Union[Group, Array]:
    """
    Recursively parse an array-like or group-like into an Array or Group.
    """
    name = element.basename
    attrs = dict(element.attrs)

    if hasattr(element, "dtype"):
        element: ArrayLike
        result = Array(
            shape=element.shape, name=name, dtype=str(element.dtype), attrs=attrs
        )
    else:
        element: GroupLike
        values = [build_tree(val) for val in element.values()]
        result = Group(name=name, attrs=attrs, values=values)
    return result
