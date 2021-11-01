from dataclasses import dataclass
from typing import List

from shapely.geometry import Point


@dataclass(frozen=True)
class BBox:
    lat_min: float
    lon_min: float
    lat_max: float
    lon_max: float


def get_corner_points(bounding_box: BBox) -> List[Point]:
    corner_points = [
        Point(bounding_box.lat_min, bounding_box.lon_min),
        Point(bounding_box.lat_min, bounding_box.lon_max),
        Point(bounding_box.lat_max, bounding_box.lon_max),
        Point(bounding_box.lat_max, bounding_box.lon_min),
    ]
    return corner_points
