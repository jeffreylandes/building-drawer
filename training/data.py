import math

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from torch.utils.data import Dataset

from training.data_utils import (
    get_bbox_from_center_point,
    get_final_bounding_box,
    get_keep_percentage,
    plot_geometry,
    normalize_distance,
    standardize_direction,
    IMG_SHAPE,
)


class BuildingData(Dataset):
    def __init__(self, path="data/sample.shp"):

        self.data = gpd.read_file(path)
        print("Loaded geopandas dataset")
        self.data_items = []
        for geometry_index, geometry in enumerate(self.data.geometry):
            items = [
                (geometry_index, coordinate_index)
                for coordinate_index in range(len(geometry.exterior.coords) - 1)
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
        final_bounding_box = get_final_bounding_box(
            initial_bounding_box, intersecting_buildings.total_bounds
        )

        keep_percentage = get_keep_percentage()
        kept_buildings = intersecting_buildings.sample(
            frac=keep_percentage, replace=False
        )

        intersection_minus_building = kept_buildings[kept_buildings.id != building.id]

        building_coordinates = list(building_polygon.exterior.coords)
        start_point = gpd.GeoSeries(Point(building_coordinates[coordinate_index]))

        buildings_plot = plot_geometry(
            intersection_minus_building.geometry, final_bounding_box, "polygon"
        )
        starting_point_plot = plot_geometry(start_point, final_bounding_box, "point")
        if coordinate_index > 0:
            first_lines = gpd.GeoSeries(
                LineString(building_coordinates[: coordinate_index + 1])
            )
            single_building_plot = plot_geometry(
                first_lines, final_bounding_box, "line"
            )
        else:
            single_building_plot = np.zeros_like(buildings_plot)

        if coordinate_index == len(building_coordinates) - 2:
            target = np.array([0, 0, 1])
            mask = np.array([0, 0, 1])
        else:
            distance = math.dist(
                building_coordinates[coordinate_index],
                building_coordinates[coordinate_index + 1],
            )
            distance_normalized = normalize_distance(distance)
            x_src, y_src = building_coordinates[coordinate_index]
            x_dst, y_dst = building_coordinates[coordinate_index + 1]
            direction = np.arctan2(y_dst - y_src, x_dst - x_src) * 180 / np.pi
            standardized_direction = standardize_direction(direction)
            target = np.array([distance_normalized, standardized_direction, 0])
            mask = np.array([1, 1, 0])

        site = np.zeros((3, IMG_SHAPE, IMG_SHAPE))
        site[0] = buildings_plot
        site[1] = single_building_plot
        site[2] = starting_point_plot

        data_item = {
            "site": site.astype(np.float32),
            "target": target.astype(np.float32),
            "target_mask": mask.astype(np.float32),
        }

        return data_item


if __name__ == "__main__":
    from utils.plot import plot_sample

    data = BuildingData()
    print(f"{len(data)} number of samples")
    for i in range(10):
        sample = data[np.random.randint(len(data))]
        plot_sample(sample)
        print(sample["target"])
