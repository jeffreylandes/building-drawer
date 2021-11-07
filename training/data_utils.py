from dataclasses import dataclass
from typing import List, Tuple

import geopandas as gpd
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Point, Polygon

BUFFER = 0.001
IMG_SHAPE = 500
MEAN_DISTANCE = 0.0001166
STD_DISTANCE = 0.00013627


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


def get_final_bounding_box(
    bounding_box: BBox, data_bounds: Tuple[float, float, float, float]
):
    data_lat_min, data_lon_min, data_lat_max, data_lon_max = data_bounds
    lat_min = min(bounding_box.lat_min, data_lat_min)
    lat_max = max(bounding_box.lat_max, data_lat_max)
    lon_min = min(bounding_box.lon_min, data_lon_min)
    lon_max = max(bounding_box.lon_max, data_lon_max)
    return BBox(lat_min=lat_min, lon_min=lon_min, lat_max=lat_max, lon_max=lon_max)


def get_keep_percentage():
    keep_percentage = np.random.normal(loc=0.75, scale=0.4)
    keep_percentage = max(keep_percentage, 0)
    keep_percentage = min(keep_percentage, 1)
    return keep_percentage


def validate_geometry_type(geometry_type: str):
    assert geometry_type in [
        "polygon",
        "point",
        "line",
    ], f"Geometry type {geometry_type} is not supported. Must be one of: polygon, point, line"


def plot_geometry(geometry: gpd.GeoSeries, bbox: BBox, geometry_type: str):
    validate_geometry_type(geometry_type)
    img = Image.new("L", (IMG_SHAPE, IMG_SHAPE), 255)
    draw = ImageDraw.Draw(img)
    for building in geometry.geometry:
        if geometry_type == "point" or geometry_type == "line":
            x, y = building.coords.xy
        else:
            x, y = building.exterior.coords.xy
        x = np.array(x)
        y = np.array(y)
        x_transformed = (x - bbox.lat_min) / (bbox.lat_max - bbox.lat_min) * IMG_SHAPE
        y_transformed = (y - bbox.lon_min) / (bbox.lon_max - bbox.lon_min) * IMG_SHAPE
        transformed_coordinates = np.array(list(zip(x_transformed, y_transformed)))
        transformed_coordinates = [tuple(c) for c in transformed_coordinates]
        if geometry_type == "point":
            draw.point(transformed_coordinates, fill=0)
        elif geometry_type == "line":
            draw.line(transformed_coordinates, fill=0)
        else:
            draw.polygon(transformed_coordinates, fill=0, outline=0)
    return np.array(img) / 255


def standardize_direction(direction: float):
    return (direction - -180) / 360


def normalize_distance(distance: float):
    return (distance - MEAN_DISTANCE) / STD_DISTANCE


def get_bbox_from_center_point(polygon: Polygon):
    center_point = polygon.centroid.coords
    center_lat, center_lon = list(center_point)[0]
    lat_min, lat_max = center_lat - BUFFER, center_lat + BUFFER
    lon_min, lon_max = center_lon - BUFFER, center_lon + BUFFER
    return BBox(lat_min=lat_min, lon_min=lon_min, lat_max=lat_max, lon_max=lon_max)
