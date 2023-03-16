from typing import Dict, List, Optional, Sequence, Tuple, Union, Literal

from pydantic import BaseModel, root_validator
from xarray import DataArray
from fibsem_tools.io.xr import stt_coord

ArrayOrder = Literal["C", "F"]


class STTransform(BaseModel):
    """
    Representation of an N-dimensional scaling -> translation transform for labelled
    axes with units.
    """

    order: Optional[ArrayOrder] = "C"
    axes: Sequence[str]
    units: Sequence[str]
    translate: Sequence[float]
    scale: Sequence[float]

    @root_validator
    def validate_argument_length(
        cls, values: Dict[str, Union[Sequence[str], Sequence[float]]]
    ):
        scale = values.get("scale")
        axes = values.get("axes")
        units = values.get("units")
        translate = values.get("translate")
        if not len(axes) == len(units) == len(translate) == len(scale):
            raise ValueError(
                f"""
                The length of all arguments must match. len(axes) = {len(axes)}, 
                len(units) = {len(units)}, len(translate) = {len(translate)}, 
                len(scale) = {len(scale)}"""
            )
        return values

    def to_coords(self, shape: Tuple[int, ...]) -> List[DataArray]:
        """
        Given an array shape (represented as a dict with dimension names as keys),
        return a dict of numpy arrays representing a bounded coordinate grid derived
        from this transform.
        """
        if self.order == "C":
            axes = self.axes
        else:
            axes = reversed(self.axes)
        return [
            stt_coord(
                shape[idx],
                dim=k,
                scale=self.scale[idx],
                translate=self.translate[idx],
                unit=self.units[idx],
            )
            for idx, k in enumerate(axes)
        ]

    @classmethod
    def fromDataArray(
        cls, array: DataArray, reverse_axes: bool = False
    ) -> "STTransform":
        """
        Generate a spatial transform from a DataArray.

        Parameters
        ----------

        array: DataArray

        reverse_axes: boolean, default=False
            If True, the order of the `axes` in the spatial transform will
            be reversed relative to the order of the dimensions of `array`.


        Returns
        -------

        STTransform

        """

        orderer = slice(None)
        output_order = "C"
        if reverse_axes:
            orderer = slice(-1, None, -1)
            output_order = "F"

        axes = [str(d) for d in array.dims[orderer]]
        # default unit is m
        units = [array.coords[ax].attrs.get("units", "m") for ax in axes]
        translate = [float(array.coords[ax][0]) for ax in axes]
        scale = []
        for ax in axes:
            if len(array.coords[ax]) > 1:
                scale_estimate = abs(
                    float(array.coords[ax][1]) - float(array.coords[ax][0])
                )
            else:
                raise ValueError(
                    f"""
                    Cannot infer scale parameter along dimension {ax} 
                    with length {len(array.coords[ax])}
                    """
                )
            scale.append(scale_estimate)

        return cls(
            axes=axes, units=units, translate=translate, scale=scale, order=output_order
        )
