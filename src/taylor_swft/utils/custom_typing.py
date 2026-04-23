from ..room.spatial_model import SWFTRoom
from numpy import float32, int32
from torch import Tensor
from typing import TypedDict, Callable
import numpy.typing as npt


PointType = npt.NDArray[float32] | list[float]

baseline_func_type = Callable[[SWFTRoom, PointType, PointType], Tensor]


class MeshType(TypedDict):
    vertices: npt.NDArray[float32]
    faces: npt.NDArray[int32]
    material_assignment: list[str]


MeshDataType = dict[str, MeshType]


class MaterialType(TypedDict):
    frequency: float32
    absorption: float32
    scattering: float32


class BRASItem(TypedDict):
    scene_name: str
    loudspeaker: str
    microphone: str
    loudspeaker_type: str
    mesh: MeshType
    source_position: npt.NDArray[float32]
    receiver_position: npt.NDArray[float32]
    material_properties: dict[str, MaterialType]
    waveform: Tensor
    sample_rate: int
    wav_file_name: str


class BRASItemSWFTRoom(BRASItem):
    swft_room: SWFTRoom
