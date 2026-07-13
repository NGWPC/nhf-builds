"""USBR RISE feature service as of 7/13/26 https://services1.arcgis.com/ixD30sld6F8MQ7V5/ArcGIS/rest/services/RISE_point_locations_(view)/FeatureServer/0 Calling with default args will save to ./usbr.gpkg"""
import requests
import geopandas as gpd
import argparse
from pathlib import Path


def download_usbr(output: Path, url: str):
    """Download USBR gpkg from RISE ArcGIS Feature Service"""
    all_features = []
    offset = 0
    chunk_size = 1000

    while True:
        params = {
            "where": "1=1",
            "outFields": "*",
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": chunk_size,
            "returnGeometry": "true"
        }
        
        response = requests.post(url, data=params).json()
        features = response.get("features", [])
        
        if not features:
            break
            
        all_features.extend(features)
        offset += len(features)

    print(f"Downloaded {len(all_features)} features.")

    gdf = gpd.GeoDataFrame.from_features(all_features, crs=4326)
    gdf.to_file(output)
    print(f"Saved USBR to {output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default="./usbr.gpkg",
        help="Output path for USBR data",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://services1.arcgis.com/ixD30sld6F8MQ7V5/ArcGIS/rest/services/RISE_point_locations_(view)/FeatureServer/0/query",
        help="Input feature service URL",
    )

    args = parser.parse_args()

    download_usbr(output=Path(args.output), url=args.url)