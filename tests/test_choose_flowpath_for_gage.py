import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon

from hydrofabric_builds.hydrolocations.assign_fp_to_gage import choose_flowpath_for_gage


def _square_km2_geodf(area_km2: float, AREA_CRS: str = "EPSG:6933") -> gpd.GeoDataFrame:
    """
    Build a square polygon with exact area (km^2) in AREA_CRS,
    then convert to EPSG:4326 (as the production code expects).
    """
    side_m = (area_km2 * 1_000_000.0) ** 0.5
    # Build square from (0,0) → (side, side) in AREA_CRS
    poly = Polygon([(0, 0), (side_m, 0), (side_m, side_m), (0, side_m)])
    gdf = gpd.GeoDataFrame(geometry=[poly], crs=AREA_CRS).to_crs("EPSG:4326")
    return gdf


def test_choose_flowpath_for_gage_selects_by_area(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Given a mock basin of exactly 100 km², ensure the candidate with totdasqkm closest
    to 100 and within the tolerance is chosen.
    """
    # --- Monkeypatch NLDI calls to return a synthetic 100 km² basin ---
    basin_100 = _square_km2_geodf(100.0)

    def _fake_by_site(site_no: str) -> gpd.GeoDataFrame:
        return basin_100

    def _fake_by_pos(lon: float, lat: float) -> gpd.GeoDataFrame:
        return basin_100

    monkeypatch.setattr("hydrofabric_builds.hydrolocations.usgs_NLDI_API.nldi_basin_by_site", _fake_by_site)
    monkeypatch.setattr(
        "hydrofabric_builds.hydrolocations.usgs_NLDI_API.nldi_basin_by_position", _fake_by_pos
    )

    # --- Two candidate flowpaths: one close (95), one far (130) ---
    candidates = gpd.GeoDataFrame(
        {
            "flowpath_id": ["A", "B"],
            "totdasqkm": [99.0, 130.0],
            "geometry": [
                LineString([(-100.0, 40.0), (-99.9, 40.0)]),
                LineString([(-100.0, 40.1), (-99.9, 40.1)]),
            ],
        },
        crs="EPSG:4326",
    )

    # Within 15% → 95 is 5% low → should select "A"
    sel_id, basin_info, _ = choose_flowpath_for_gage(
        in_buf=candidates,
        site_no="01125490",  # "01125490"
        lon=-71.93,
        lat=41.9278,
        flow_id_col="flowpath_id",
        area_col="totdasqkm",
        area_match_pct=0.15,
    )
    assert sel_id == "A"
    assert pytest.approx(basin_info.area_km2, rel=1e-1) == 94.88
    assert basin_info.source in ("nldi_site", "nldi_position")  # monkeypatched either path OK

    # Tighten tolerance to 1% , should return None
    sel_id_tight, _, _ = choose_flowpath_for_gage(
        in_buf=candidates,
        site_no="01125490",
        lon=-71.93,
        lat=41.9278,
        flow_id_col="flowpath_id",
        area_col="totdasqkm",
        area_match_pct=0.01,
    )
    assert sel_id_tight is None
