import math
from dataclasses import dataclass
from typing import Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, Point, LineString
from torch.utils.data import Dataset


BUFFER = 0.002
IMG_SHAPE = 800


@dataclass(frozen=True)
class BBox:
    lat_min: float
    lon_min: float
    lat_max: float
    lon_max: float


def plot_and_get_data(geometry: gpd.GeoSeries):
    geometry.plot(figsize=(8, 8), markersize=1)
    plt.axis("off")
    plt.savefig("/tmp/geometry.png")
    plt.close("all")
    img = Image.open("/tmp/geometry.png")
    data = np.array(img)
    return data[:, :, 0].astype(float)


def standardize_direction(direction: float):
    return (direction - -180) / 360


def plot_geometry(geometry: gpd.GeoSeries, corners: gpd.GeoSeries):
    all_geometry = corners.append(geometry)
    all_geometry_data = plot_and_get_data(all_geometry)
    return all_geometry_data


def get_keep_percentage():
    keep_percentage = np.random.normal(loc=0.75, scale=0.4)
    keep_percentage = max(keep_percentage, 0)
    keep_percentage = min(keep_percentage, 1)
    return keep_percentage


def get_bbox_from_center_point(polygon: Polygon):
    center_point = polygon.centroid.coords
    center_lat, center_lon = list(center_point)[0]
    lat_min, lat_max = center_lat - BUFFER, center_lat + BUFFER
    lon_min, lon_max = center_lon - BUFFER, center_lon + BUFFER
    return BBox(lat_min=lat_min, lon_min=lon_min, lat_max=lat_max, lon_max=lon_max)


def get_final_bounding_box(bounding_box: BBox, data_bounds: Tuple[float, float, float, float]):
    data_lat_min, data_lon_min, data_lat_max, data_lon_max = data_bounds
    lat_min = min(bounding_box.lat_min, data_lat_min)
    lat_max = max(bounding_box.lat_max, data_lat_max)
    lon_min = min(bounding_box.lon_min, data_lon_min)
    lon_max = max(bounding_box.lon_max, data_lon_max)
    return BBox(
        lat_min=lat_min,
        lon_min=lon_min,
        lat_max=lat_max,
        lon_max=lon_max
    )


class BuildingData(Dataset):
    def __init__(self, path="data/buildings.shp"):

        self.data = gpd.read_file(path)
        self.data_items = []
        for geometry_index, geometry in enumerate(self.data.geometry):
            items = [
                (geometry_index, coordinate_index)
                for coordinate_index in range(len(geometry.exterior.coords))
            ]
            self.data_items.extend(items)

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, item):
        geometry_index, coordinate_index = self.data_items[item]
        building = self.data.iloc[geometry_index]
        building_polygon: Polygon = building.geometry
        initial_bounding_box = get_bbox_from_center_point(building_polygon)
        intersecting_buildings = self.data.cx[
            initial_bounding_box.lat_min : initial_bounding_box.lat_max,
            initial_bounding_box.lon_min : initial_bounding_box.lon_max,
        ].copy()
        final_bounding_box = get_final_bounding_box(initial_bounding_box, intersecting_buildings.total_bounds)

        keep_percentage = get_keep_percentage()
        kept_buildings = intersecting_buildings.sample(
            frac=keep_percentage, replace=False
        )

        intersection_minus_building = kept_buildings[kept_buildings.id != building.id]

        corner_points = [
            Point(final_bounding_box.lat_min, final_bounding_box.lon_min),
            Point(final_bounding_box.lat_min, final_bounding_box.lon_max),
            Point(final_bounding_box.lat_max, final_bounding_box.lon_max),
            Point(final_bounding_box.lat_max, final_bounding_box.lon_min),
        ]
        corners = gpd.GeoSeries(corner_points)
        corners_plot = plot_geometry(corners, corners)
        coords = list(building_polygon.exterior.coords)
        start_point = gpd.GeoSeries(Point(coords[coordinate_index]))

        buildings = plot_geometry(intersection_minus_building.geometry, corners)
        buildings = np.where(buildings != corners_plot, buildings, 0)
        if coordinate_index > 0:
            first_lines = gpd.GeoSeries(LineString(coords[: coordinate_index + 1]))
            single_building = plot_geometry(first_lines, corners)
        else:
            single_building = np.zeros_like(buildings)
        single_building = np.where(single_building != corners_plot, single_building, 0)
        starting_point = plot_geometry(start_point, corners)
        starting_point = np.where(starting_point != corners_plot, starting_point, 0)

        if coordinate_index == len(coords) - 1:
            target = [0, 0, 1]
            mask = [0, 0, 1]
        else:
            distance = math.dist(coords[coordinate_index], coords[coordinate_index + 1])
            x_src, y_src = coords[coordinate_index]
            x_dst, y_dst = coords[coordinate_index + 1]
            direction = np.arctan2(y_dst - y_src, x_dst - x_src) * 180 / np.pi
            standardized_direction = standardize_direction(direction)
            target = [distance, standardized_direction, 0]
            mask = [1, 1, 0]

        site = np.zeros((3, IMG_SHAPE, IMG_SHAPE))
        site[0] = buildings
        site[1] = single_building
        site[2] = starting_point

        data_item = {"site": site, "target": target, "target_mask": mask}

        return data_item


if __name__ == "__main__":
    from utils.plot import plot_sample
    data = BuildingData()
    print(f"{len(data)} number of samples")
    for i in range(10):
        sample = data[np.random.randint(len(data))]
        plot_sample(sample)
        print(sample["target"])
