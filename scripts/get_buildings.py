import time
from collections import defaultdict
from uuid import uuid4

import geopandas as gpd
import numpy as np
from osmxtract import overpass
from shapely.geometry import Polygon


MAX_ATTEMPTS = 5


def make_request(bounds, attempt_number=0):
    try:
        query = overpass.ql_query(bounds, tag="building")
        response = overpass.request(query)
        elements = response["elements"]
        return elements
    except:
        if attempt_number > MAX_ATTEMPTS:
            raise Exception("Not gonna happen")
        time.sleep(10)
        print("Trying again")
        return make_request(bounds, attempt_number + 1)


def main():
    start_lat, start_lon = 51.321, -0.624
    end_lat, end_lon = 51.826, 0.322
    data_items = defaultdict(list)
    for lat in np.arange(start_lat, end_lat, 0.05):
        for lon in np.arange(start_lon, end_lon, 0.05):
            bounds = (lat, lon, lat + 0.05, lon + 0.05)
            elements = make_request(bounds)

            for element in elements:
                if "geometry" not in element:
                    continue
                geometry = element["geometry"]
                coords = [
                    (coordinate["lon"], coordinate["lat"]) for coordinate in geometry
                ]
                try:
                    building_polygon = Polygon(coords)
                    data_items["id"].append(str(uuid4()))
                    data_items["geometry"].append(building_polygon)
                except ValueError as e:
                    print(e, "skipping")
    print(f"Number of building geometry samples: {len(data_items['geometry'])}")
    df = gpd.GeoDataFrame(data_items, crs="EPSG:4326")
    df.to_file("data/buildings.shp")


if __name__ == "__main__":
    main()
