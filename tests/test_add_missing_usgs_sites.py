from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd
import pytest

from hydrofabric_builds.hydrolocations.usgs_gages_builder import add_missing_usgs_sites


def test_add_missing_usgs_sites_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    add_missing_usgs_sites should:
      - partition numeric USGS IDs vs non-USGS IDs (with letters)
      - fetch NWIS site metadata for USGS IDs
      - append missing sites with geometry and station name -> name_plain
      - fill other gages columns with '-'
    """
    # Fake NWIS response table (as if read via pd.read_csv(url, sep="\t", comment="#", ...))
    fake = pd.DataFrame(
        {
            "agency_cd": ["USGS", "USGS"],
            "site_no": ["01000000", "02000000"],
            "station_nm": ["Alpha", "Beta"],
            "dec_lat_va": [40.0, 41.0],
            "dec_long_va": [-100.0, -101.0],
            "state_cd": ["01", "02"],
            "county_cd": ["001", "002"],
        }
    )

    def fake_read_csv(
        url: str,
        sep: str = "\t",
        comment: str = "#",
        dtype: Any = str,
        engine: str = "python",
    ) -> pd.DataFrame:
        # Ignore the URL and return our fake NWIS table
        return fake

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    # Empty gages with the expected schema/columns
    empty = gpd.GeoDataFrame(
        {
            "geometry": [],
            "state": [],
            "site_no": [],
            "name_plain": [],
            "name_raw": [],
            "description": [],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    # Include one non-USGS ID (letters) to verify partitioning
    g_updated, usgs_ids, non_usgs, fetched = add_missing_usgs_sites(
        empty, ["01000000", "02DC012", "02000000"]
    )

    # IDs partitioned as expected
    assert set(usgs_ids) == {"01000000", "02000000"}
    assert set(non_usgs) == {"02DC012"}

    # We added two sites
    assert set(g_updated["site_no"]) == {"01000000", "02000000"}

    # Name mapping is station_nm -> name_plain
    names = set(g_updated["name_plain"].astype(str))
    assert names == {"Alpha", "Beta"}

    # Non-mapped fields should be '-'
    assert (g_updated["name_raw"] == "-").all()
    assert (g_updated["state"] == "-").all()
    assert (g_updated["description"] == "-").all()

    # Geometry created from dec_long_va/dec_lat_va
    assert not g_updated.geometry.is_empty.any()
    assert g_updated.crs and g_updated.crs.to_string() in {"EPSG:4326", "epsg:4326"}
