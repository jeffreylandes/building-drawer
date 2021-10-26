import geopandas as gpd
from osmxtract import location, overpass
from shapely.geometry import Polygon

import matplotlib.pyplot as plt


def main():
    lat, lon = 51.507, 0.127
    bounds = location.from_buffer(lat, lon, buffer_size=1000)
    query = overpass.ql_query(bounds, tag="building")
    response = overpass.request(query)
    elements = response["elements"]
    polygons = []
    for i, element in enumerate(elements):
        geometry = element["geometry"]
        coords = [(coordinate["lat"], coordinate["lon"]) for coordinate in geometry]
        building_polygon = Polygon(coords)
        polygons.append(building_polygon)
        if i > 5:
            break
    series = gpd.GeoSeries(polygons)
    series.plot()
    plt.axis("off")
    plt.savefig("data/sample.png")
    plt.show()


if __name__ == "__main__":
    main()
