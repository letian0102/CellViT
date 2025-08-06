import numpy as np
from PIL import Image
import tifffile


class TiffSlide:
    """Minimal OpenSlide-like interface for generic TIFF images."""

    def __init__(self, path: str):
        self.path = path
        self._array = tifffile.imread(path)
        if self._array.ndim == 2:
            self._array = np.stack([self._array] * 3, axis=-1)
        self.height, self.width = self._array.shape[:2]
        self.dimensions = (self.width, self.height)
        self.level_dimensions = [self.dimensions]
        self.level_count = 1
        self.properties = {
            "openslide.level[0].width": str(self.width),
            "openslide.level[0].height": str(self.height),
        }

    def read_region(self, location, level, size):
        x, y = location
        w, h = size
        region = np.zeros((h, w, self._array.shape[2]), dtype=self._array.dtype)
        x2 = max(0, x)
        y2 = max(0, y)
        region_slice = self._array[y2 : y2 + h, x2 : x2 + w]
        region[: region_slice.shape[0], : region_slice.shape[1]] = region_slice
        return Image.fromarray(region)

    def get_thumbnail(self, size):
        return Image.fromarray(self._array).resize(size, Image.BILINEAR)

    def close(self):
        pass


class DeepZoomGeneratorTiff:
    """Simple DeepZoom generator for :class:`TiffSlide`."""

    def __init__(self, osr, cucim_slide=None, tile_size=254, overlap=0, limit_bounds=True, **kwargs):
        self.osr = osr
        self.tile_size = tile_size
        self.overlap = overlap
        self.level_count = 1
        stride = tile_size - 2 * overlap
        cols = int(np.ceil(osr.dimensions[0] / stride))
        rows = int(np.ceil(osr.dimensions[1] / stride))
        self.level_tiles = [(cols, rows)]

    def get_tile(self, level, address):
        col, row = address
        stride = self.tile_size - 2 * self.overlap
        x = col * stride
        y = row * stride
        return self.osr.read_region((x, y), 0, (self.tile_size, self.tile_size))
