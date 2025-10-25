from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Point

from hydrofabric_builds.hydrolocations.usgs_gages_builder import merge_minimal_gages


def test_merge_minimal_appends_and_fills(gages_empty: gpd.GeoDataFrame) -> None:
    src = gpd.GeoDataFrame(
        {
            "geometry": [Point(-90, 30)],
            "site_no": ["12345678"],
            "station_nm": ["TXDOT gage @ I-10"],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    out = merge_minimal_gages(gages_empty, src, update_existing=False)
    assert len(out) == 1
    r = out.iloc[0]
    assert r.site_no == "12345678"
    assert r.name_plain == "TXDOT gage @ I-10"
    assert r.state == r.name_raw == r.description == "-"


def test_merge_minimal_updates_existing(gages_seed: gpd.GeoDataFrame) -> None:
    src = gpd.GeoDataFrame(
        {
            "geometry": [Point(-101, 41)],
            "site_no": ["00000001"],
            "station_nm": ["updated-name"],
        },
        geometry="geometry",
        crs=gages_seed.crs,
    )
    out = merge_minimal_gages(gages_seed, src, update_existing=True)
    r = out.iloc[0]
    assert r.name_plain == "updated-name"
    assert (r.geometry.x, r.geometry.y) == (-101, 41)
