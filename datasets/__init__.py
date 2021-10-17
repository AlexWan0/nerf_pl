from .blender import BlenderDataset
from .llff import LLFFDataset
from .tum import TUMDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'tum': TUMDataset}