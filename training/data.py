import math

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, Point, LineString
from torch.utils.data import Dataset


BUFFER = 0.006


# TODO: Remove corners
# TODO: Standardize angle
# TODO: Delete building that aren't *within* bounds
# TODO: Figure out how to close figures


def plot_and_get_data(geometry: gpd.GeoSeries):
    geometry.plot(figsize=(8, 8), markersize=1)
    plt.axis('off')
    plt.savefig('/tmp/geometry.png')
    plt.close()
    img = Image.open('/tmp/geometry.png')
    data = np.array(img)
    return data


# TODO: Remove corners
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
    return lat_min, lon_min, lat_max, lon_max


class BuildingData(Dataset):

    def __init__(self, path="../data/buildings.shp"):

        self.data = gpd.read_file(path)
        self.data_items = []
        for geometry_index, geometry in enumerate(self.data.geometry):
            items = [(geometry_index, coordinate_index) for coordinate_index in range(len(geometry.exterior.coords))]
            self.data_items.extend(items)

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, item):
        geometry_index, coordinate_index = self.data_items[item]
        building = self.data.iloc[geometry_index]
        building_polygon: Polygon = building.geometry
        lat_min, lon_min, lat_max, lon_max = get_bbox_from_center_point(building_polygon)
        intersecting_buildings = self.data.cx[lat_min: lat_max, lon_min: lon_max].copy()

        keep_percentage = get_keep_percentage()
        kept_buildings = intersecting_buildings.sample(frac=keep_percentage, replace=False)
        kept_buildings.plot()

        intersection_minus_building = intersecting_buildings[intersecting_buildings.id != building.id]

        corner_points = [
            Point(lat_min, lon_min),
            Point(lat_min, lon_max),
            Point(lat_max, lon_max),
            Point(lat_max, lon_min)
        ]
        corners = gpd.GeoSeries(corner_points)
        coords = list(building_polygon.exterior.coords)
        start_point = gpd.GeoSeries(Point(coords[coordinate_index]))

        buildings = plot_geometry(intersection_minus_building.geometry, corners)
        if coordinate_index > 0:
            first_lines = gpd.GeoSeries(LineString(coords[:3]))
            single_building = plot_geometry(first_lines, corners)
        else:
            single_building = np.zeros_like(buildings)
        starting_point = plot_geometry(start_point, corners)

        if coordinate_index == len(coords) - 1:
            target = [0, 0, 1]
            mask = [0, 0, 1]
        else:
            distance = math.dist(coords[coordinate_index], coords[coordinate_index + 1])
            x_src, y_src = coords[coordinate_index]
            x_dst, y_dst = coords[coordinate_index + 1]
            direction = np.arctan2(y_dst - y_src, x_dst - x_src) * 180 / np.pi
            target = [distance, direction, 0]
            mask = [1, 1, 0]

        site = np.zeros((3, 800, 800))
        site[0] = buildings[:, :, 0]
        site[1] = single_building[:, :, 0]
        site[2] = starting_point[:, :, 0]

        data_item = {
            "site": site,
            "target": target,
            "target_mask": mask
        }

        return data_item


if __name__ == "__main__":
    data = BuildingData()
    for i in range(100):
        data[np.random.randint(len(data))]
    print("done")