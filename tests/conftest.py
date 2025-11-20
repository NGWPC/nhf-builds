"""conftest for test_trace.py with edge case fixtures."""

from collections import deque
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import pytest
import rustworkx as rx
from pyprojroot import here
from shapely.geometry import LineString, Polygon

from hydrofabric_builds.config import HFConfig, TaskSelection
from hydrofabric_builds.schemas.hydrofabric import (
    Aggregations,
    BuildHydrofabricConfig,
    Classifications,
)
from scripts.hf_runner import TaskInstance


@pytest.fixture
def task_instance() -> TaskInstance:
    """Fixture providing a TaskInstance."""
    return TaskInstance()


@pytest.fixture
def mock_geopackages() -> tuple[str, str]:
    """Create paths to test data files."""
    return str(here() / "tests/data/sample_divides.parquet"), str(
        here() / "tests/data/sample_flowpaths.parquet"
    )


@pytest.fixture
def sample_divides() -> gpd.GeoDataFrame:
    """Open the sample test data"""
    _gdf = gpd.read_parquet(here() / "tests/data/sample_divides.parquet")
    _gdf["divide_id"] = _gdf["divide_id"].astype("int").astype("str")
    return _gdf


@pytest.fixture
def sample_flowpaths() -> gpd.GeoDataFrame:
    """Open the sample test data"""
    _gdf = gpd.read_parquet(here() / "tests/data/sample_flowpaths.parquet")
    _gdf["flowpath_id"] = _gdf["flowpath_id"].astype("int").astype("str")
    return _gdf


@pytest.fixture
def sample_config(mock_geopackages: tuple[str, str], tmp_path: Path) -> HFConfig:
    """Fixture providing a sample HFConfig."""
    return HFConfig(
        build=BuildHydrofabricConfig(
            reference_divides_path=mock_geopackages[0], reference_flowpaths_path=mock_geopackages[1]
        ),
        tasks=TaskSelection(
            build_hydrofabric=True,
            divide_attributes=False,
            flowpath_attributes=False,
            waterbodies=False,
            gages=False,
        ),
        output_dir=tmp_path,
    )


@pytest.fixture
def expected_graph() -> dict[str, list[str]]:
    """Fixture providing expected network graph structure."""
    return {
        "6720527": ["6720517", "6720515"],
        "6720413": ["6722029"],
        "6712619": ["6712621"],
        "6720391": ["6722665"],
        "6720467": ["6720439"],
        "6721863": ["6720475", "6720473"],
        "6720417": ["6720413", "6720371"],
        "6720795": ["6721871", "6720607"],
        "6720797": ["6720703", "6720701"],
        "6720475": ["6720383"],
        "6720431": ["6720385"],
        "6720493": ["6720531", "6720497"],
        "6720389": ["6720449", "6720381"],
        "6719083": ["6719081", "6719063"],
        "6720393": ["6722017"],
        "6720525": ["6722667"],
        "6720559": ["6721869"],
        "6722665": ["6720363"],
        "6720497": ["6720437"],
        "6722667": ["6719083"],
        "6720519": ["6720559"],
        "6722035": ["6720431"],
        "6712111": ["6712133"],
        "6720473": ["6720391", "6720389"],
        "6712133": ["6712725"],
        "6720515": ["6721863", "6720551"],
        "6720881": ["6720893"],
        "6719081": ["6720393"],
        "6720689": ["6720681", "6720679"],
        "6720877": ["6720797", "6720795"],
        "6712621": ["6712111"],
        "6722029": ["6722041", "6722035"],
        "6722023": ["6720503"],
        "6720871": ["6720821"],
        "6720679": ["6720677", "6720675"],
        "6722017": ["6720407"],
        "6720531": ["6722499"],
        "6720703": ["6720683", "6720651"],
        "6720801": ["6720643"],
        "6720433": ["6722023"],
        "6720675": ["6722501"],
        "6720521": ["6720519"],
        "6720607": ["6720527", "6720525"],
        "6720407": ["6720433", "6720417"],
        "6720879": ["6720883", "6720877"],
        "6721871": ["6720581"],
        "6722041": ["6720493"],
        "6720645": ["6720609"],
        "6720683": ["6720773", "6720689"],
    }


@pytest.fixture
def sample_aggregate_data() -> Aggregations:
    """Sample aggregation data for testing - matches actual classification results."""
    return Aggregations(
        aggregates=[
            {
                "ref_ids": ["6720879", "6720877", "6720883"],
                "dn_id": "6720879",
                "up_id": "6720877",
                "vpu_id": "01",
                "hydroseq": 4337204982,
                "length_km": 3.799,
                "area_sqkm": 10.62359981550615,
                "div_area_sqkm": 10.62359981550615,
                "line_geometry": LineString([(0, 0), (1, 1), (2, 2)]),
                "polygon_geometry": Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]),
            },
            {
                "ref_ids": ["6720703", "6720683", "6720651"],
                "dn_id": "6720703",
                "up_id": "6720683",
                "vpu_id": "01",
                "hydroseq": 4337205029,
                "length_km": 4.375,
                "area_sqkm": 6.32744983799809,
                "div_area_sqkm": 6.32744983799809,
                "line_geometry": LineString([(3, 3), (4, 4), (5, 5)]),
                "polygon_geometry": Polygon([(3, 3), (6, 3), (6, 6), (3, 6)]),
            },
            {
                "ref_ids": ["6721871", "6720581"],
                "dn_id": "6721871",
                "up_id": "6720581",
                "vpu_id": "01",
                "hydroseq": 4337205024,
                "length_km": 2.649,
                "area_sqkm": 2.3732996894982596,
                "div_area_sqkm": 2.3732996894982596,
                "line_geometry": LineString([(6, 6), (7, 7)]),
                "polygon_geometry": Polygon([(6, 6), (8, 6), (8, 8), (6, 8)]),
            },
            {
                "ref_ids": ["6720681", "6720679", "6720689"],
                "dn_id": "6720689",
                "up_id": "6720679",
                "vpu_id": "01",
                "hydroseq": 4337205031,
                "length_km": 0.735,
                "area_sqkm": 0.33660036900013,
                "div_area_sqkm": 0.33660036900013,
                "line_geometry": LineString([(8, 8), (9, 9), (10, 10)]),
                "polygon_geometry": Polygon([(8, 8), (11, 8), (11, 11), (8, 11)]),
            },
            {
                "ref_ids": ["6720515", "6720527", "6720517"],
                "dn_id": "6720527",
                "up_id": "6720515",
                "vpu_id": "01",
                "hydroseq": 4337205010,
                "length_km": 0.9530000000000001,
                "area_sqkm": 0.44954983800014997,
                "div_area_sqkm": 0.44954983800014997,
                "line_geometry": LineString([(11, 11), (12, 12), (13, 13)]),
                "polygon_geometry": Polygon([(11, 11), (14, 11), (14, 14), (11, 14)]),
            },
            {
                "ref_ids": ["6720675", "6722501"],
                "dn_id": "6720675",
                "up_id": "6722501",
                "vpu_id": "01",
                "hydroseq": 4337205033,
                "length_km": 3.657,
                "area_sqkm": 6.93585036000335,
                "div_area_sqkm": 6.93585036000335,
                "line_geometry": LineString([(14, 14), (15, 15)]),
                "polygon_geometry": Polygon([(14, 14), (16, 14), (16, 16), (14, 16)]),
            },
            {
                "ref_ids": ["6721863", "6720473", "6720475", "6720383"],
                "dn_id": "6721863",
                "up_id": "6720473",
                "vpu_id": "01",
                "hydroseq": 4337205013,
                "length_km": 4.965999999999999,
                "area_sqkm": 5.47739995050145,
                "div_area_sqkm": 5.47739995050145,
                "line_geometry": LineString([(17, 17), (18, 18), (19, 19)]),
                "polygon_geometry": Polygon([(17, 17), (20, 17), (20, 20), (17, 20)]),
            },
            {
                "ref_ids": ["6719083", "6722667"],
                "dn_id": "6722667",
                "up_id": "6719083",
                "vpu_id": "01",
                "hydroseq": 4337204987,
                "length_km": 4.016,
                "area_sqkm": 5.9391003150023,
                "div_area_sqkm": 5.9391003150023,
                "line_geometry": LineString([(21, 21), (22, 22)]),
                "polygon_geometry": Polygon([(21, 21), (23, 21), (23, 23), (21, 23)]),
            },
            {
                "ref_ids": ["6722665", "6720363", "6720391"],
                "dn_id": "6720391",
                "up_id": "6720363",
                "vpu_id": "01",
                "hydroseq": 4337205019,
                "length_km": 1.107,
                "area_sqkm": 4.46804980649853,
                "div_area_sqkm": 4.46804980649853,
                "line_geometry": LineString([(24, 24), (25, 25)]),
                "polygon_geometry": Polygon([(24, 24), (26, 24), (26, 26), (24, 26)]),
            },
            {
                "ref_ids": ["6720381", "6720389", "6720449"],
                "dn_id": "6720389",
                "up_id": "6720449",
                "vpu_id": "01",
                "hydroseq": 4337205016,
                "length_km": 2.211,
                "area_sqkm": 1.25415033749901,
                "div_area_sqkm": 1.25415033749901,
                "line_geometry": LineString([(27, 27), (28, 28)]),
                "polygon_geometry": Polygon([(27, 27), (29, 27), (29, 29), (27, 29)]),
            },
            {
                "ref_ids": ["6720393", "6722017", "6720407"],
                "dn_id": "6720393",
                "up_id": "6720407",
                "vpu_id": "01",
                "hydroseq": 4337204992,
                "length_km": 2.409,
                "area_sqkm": 3.03480071549642,
                "div_area_sqkm": 3.03480071549642,
                "line_geometry": LineString([(30, 30), (31, 31)]),
                "polygon_geometry": Polygon([(30, 30), (32, 30), (32, 32), (30, 32)]),
            },
            {
                "ref_ids": ["6722023", "6720503", "6720433"],
                "dn_id": "6720433",
                "up_id": "6720503",
                "vpu_id": "01",
                "hydroseq": 4337205008,
                "length_km": 3.002,
                "area_sqkm": 4.07744972100535,
                "div_area_sqkm": 4.07744972100535,
                "line_geometry": LineString([(33, 33), (34, 34)]),
                "polygon_geometry": Polygon([(33, 33), (35, 33), (35, 35), (33, 35)]),
            },
            {
                "ref_ids": ["6722029", "6720413"],
                "dn_id": "6720413",
                "up_id": "6722029",
                "vpu_id": "01",
                "hydroseq": 4337204996,
                "length_km": 1.644,
                "area_sqkm": 3.0856503465075,
                "div_area_sqkm": 3.0856503465075,
                "line_geometry": LineString([(36, 36), (37, 37)]),
                "polygon_geometry": Polygon([(36, 36), (38, 36), (38, 38), (36, 38)]),
            },
            {
                "ref_ids": ["6722035", "6720385", "6720431"],
                "dn_id": "6722035",
                "up_id": "6720385",
                "vpu_id": "01",
                "hydroseq": 4337204999,
                "length_km": 2.161,
                "area_sqkm": 4.11794962650009,
                "div_area_sqkm": 4.11794962650009,
                "line_geometry": LineString([(39, 39), (40, 40)]),
                "polygon_geometry": Polygon([(39, 39), (41, 39), (41, 41), (39, 41)]),
            },
            {
                "ref_ids": ["6720437", "6720493", "6720497", "6722499", "6720531"],
                "dn_id": "6720493",
                "up_id": "6720437",
                "vpu_id": "01",
                "hydroseq": 4337205003,
                "length_km": 4.753,
                "area_sqkm": 4.07924977950065,
                "div_area_sqkm": 4.07924977950065,
                "line_geometry": LineString([(42, 42), (43, 43), (44, 44)]),
                "polygon_geometry": Polygon([(42, 42), (45, 42), (45, 45), (42, 45)]),
            },
        ],
        independents=[
            {
                "ref_ids": "6720797",
                "vpu_id": "01",
                "hydroseq": 4337205025,
                "length_km": 4.836,
                "area_sqkm": 4.03380049949858,
                "div_area_sqkm": 4.03380049949858,
                "line_geometry": LineString([(50, 50), (51, 51)]),
                "polygon_geometry": Polygon([(50, 50), (52, 50), (52, 52), (50, 52)]),
            },
            {
                "ref_ids": "6720795",
                "vpu_id": "01",
                "hydroseq": 4337204983,
                "length_km": 7.295,
                "area_sqkm": 16.86870027451226,
                "div_area_sqkm": 16.86870027451226,
                "line_geometry": LineString([(53, 53), (54, 54)]),
                "polygon_geometry": Polygon([(53, 53), (55, 53), (55, 55), (53, 55)]),
            },
            {
                "ref_ids": "6720701",
                "vpu_id": "01",
                "hydroseq": 4337205026,
                "length_km": 7.972,
                "area_sqkm": 6.64425029249713,
                "div_area_sqkm": 6.64425029249713,
                "line_geometry": LineString([(56, 56), (57, 57)]),
                "polygon_geometry": Polygon([(56, 56), (58, 56), (58, 58), (56, 58)]),
            },
            {
                "ref_ids": "6720607",
                "vpu_id": "01",
                "hydroseq": 4337204984,
                "length_km": 4.968,
                "area_sqkm": 10.08899974798643,
                "div_area_sqkm": 10.08899974798643,
                "line_geometry": LineString([(59, 59), (60, 60)]),
                "polygon_geometry": Polygon([(59, 59), (61, 59), (61, 61), (59, 61)]),
            },
            {
                "ref_ids": "6720773",
                "vpu_id": "01",
                "hydroseq": 4337205036,
                "length_km": 1.925,
                "area_sqkm": 4.16610002699482,
                "div_area_sqkm": 4.16610002699482,
                "line_geometry": LineString([(62, 62), (63, 63)]),
                "polygon_geometry": Polygon([(62, 62), (64, 62), (64, 64), (62, 64)]),
            },
            {
                "ref_ids": "6720525",
                "vpu_id": "01",
                "hydroseq": 4337204985,
                "length_km": 6.493,
                "area_sqkm": 11.34675044550291,
                "div_area_sqkm": 11.34675044550291,
                "line_geometry": LineString([(65, 65), (66, 66)]),
                "polygon_geometry": Polygon([(65, 65), (67, 65), (67, 67), (65, 67)]),
            },
            {
                "ref_ids": "6720677",
                "vpu_id": "01",
                "hydroseq": 4337205034,
                "length_km": 1.396,
                "area_sqkm": 1.46385031499701,
                "div_area_sqkm": 1.46385031499701,
                "line_geometry": LineString([(68, 68), (69, 69)]),
                "polygon_geometry": Polygon([(68, 68), (70, 68), (70, 70), (68, 70)]),
            },
            {
                "ref_ids": "6720551",
                "vpu_id": "01",
                "hydroseq": 4337205011,
                "length_km": 1.23,
                "area_sqkm": 2.6792992530002,
                "div_area_sqkm": 2.6792992530002,
                "line_geometry": LineString([(71, 71), (72, 72)]),
                "polygon_geometry": Polygon([(71, 71), (73, 71), (73, 73), (71, 73)]),
            },
            {
                "ref_ids": "6719081",
                "vpu_id": "01",
                "hydroseq": 4337204989,
                "length_km": 3.742,
                "area_sqkm": 12.66614926199803,
                "div_area_sqkm": 12.66614926199803,
                "line_geometry": LineString([(74, 74), (75, 75)]),
                "polygon_geometry": Polygon([(74, 74), (76, 74), (76, 76), (74, 76)]),
            },
            {
                "ref_ids": "6719063",
                "vpu_id": "01",
                "hydroseq": 4337204988,
                "length_km": 4.678,
                "area_sqkm": 11.25540018000239,
                "div_area_sqkm": 11.25540018000239,
                "line_geometry": LineString([(77, 77), (78, 78)]),
                "polygon_geometry": Polygon([(77, 77), (79, 77), (79, 79), (77, 79)]),
            },
            {
                "ref_ids": "6720417",
                "vpu_id": "01",
                "hydroseq": 4337204993,
                "length_km": 3.39,
                "area_sqkm": 3.92804980649456,
                "div_area_sqkm": 3.92804980649456,
                "line_geometry": LineString([(80, 80), (81, 81)]),
                "polygon_geometry": Polygon([(80, 80), (82, 80), (82, 82), (80, 82)]),
            },
            {
                "ref_ids": "6720371",
                "vpu_id": "01",
                "hydroseq": 4337204994,
                "length_km": 0.734,
                "area_sqkm": 4.45139973449734,
                "div_area_sqkm": 4.45139973449734,
                "line_geometry": LineString([(83, 83), (84, 84)]),
                "polygon_geometry": Polygon([(83, 83), (85, 83), (85, 85), (83, 85)]),
            },
            {
                "ref_ids": "6722041",
                "vpu_id": "01",
                "hydroseq": 4337205000,
                "length_km": 1.737,
                "area_sqkm": 4.32990044999297,
                "div_area_sqkm": 4.32990044999297,
                "line_geometry": LineString([(86, 86), (87, 87)]),
                "polygon_geometry": Polygon([(86, 86), (88, 86), (88, 88), (86, 88)]),
            },
        ],
        virtual_flowpaths=[
            {
                "ref_ids": "6720517",
                "line_geometry": LineString([(100, 100), (101, 101)]),
                "polygon_geometry": Polygon([(100, 100), (102, 100), (102, 102), (100, 102)]),
            },
            {
                "ref_ids": "6720681",
                "line_geometry": LineString([(103, 103), (104, 104)]),
                "polygon_geometry": Polygon([(103, 103), (105, 103), (105, 105), (103, 105)]),
            },
            {
                "ref_ids": "6720381",
                "line_geometry": LineString([(106, 106), (107, 107)]),
                "polygon_geometry": Polygon([(106, 106), (108, 106), (108, 108), (106, 108)]),
            },
            {
                "ref_ids": "6720883",
                "line_geometry": LineString([(109, 109), (110, 110)]),
                "polygon_geometry": Polygon([(109, 109), (111, 109), (111, 111), (109, 111)]),
            },
            {
                "ref_ids": "6720475",
                "line_geometry": LineString([(112, 112), (113, 113)]),
                "polygon_geometry": Polygon([(112, 112), (114, 112), (114, 114), (112, 114)]),
            },
            {
                "ref_ids": "6720383",
                "line_geometry": LineString([(115, 115), (116, 116)]),
                "polygon_geometry": Polygon([(115, 115), (117, 115), (117, 117), (115, 117)]),
            },
            {
                "ref_ids": "6720651",
                "line_geometry": LineString([(118, 118), (119, 119)]),
                "polygon_geometry": Polygon([(118, 118), (120, 118), (120, 120), (118, 120)]),
            },
            {
                "ref_ids": "6722499",
                "line_geometry": LineString([(121, 121), (122, 122)]),
                "polygon_geometry": Polygon([(121, 121), (123, 121), (123, 123), (121, 123)]),
            },
            {
                "ref_ids": "6720531",
                "line_geometry": LineString([(124, 124), (125, 125)]),
                "polygon_geometry": Polygon([(124, 124), (126, 124), (126, 126), (124, 126)]),
            },
        ],
        connectors=[],
        small_scale_connectors=[],
    )


@pytest.fixture
def sample_classifications() -> Classifications:
    """Sample classification data for testing."""
    return Classifications(
        aggregation_pairs=[
            ("6720883", "6720879"),
            ("6720879", "6720877"),
            ("6720651", "6720703"),
            ("6720703", "6720683"),
            ("6720581", "6721871"),
            ("6720681", "6720689"),
            ("6720689", "6720679"),
            ("6720517", "6720527"),
            ("6720527", "6720515"),
            ("6722501", "6720675"),
            ("6720475", "6721863"),
            ("6720383", "6721863"),
            ("6721863", "6720473"),
            ("6722667", "6719083"),
            ("6722665", "6720391"),
            ("6720363", "6720391"),
            ("6720381", "6720389"),
            ("6720389", "6720449"),
            ("6720393", "6722017"),
            ("6720393", "6720407"),
            ("6722023", "6720433"),
            ("6720503", "6720433"),
            ("6720413", "6722029"),
            ("6720431", "6722035"),
            ("6720385", "6722035"),
            ("6720531", "6720493"),
            ("6722499", "6720493"),
            ("6720493", "6720497"),
            ("6720493", "6720437"),
        ],
        virtual_flowpaths={
            "6720381",
            "6722499",
            "6720883",
            "6720475",
            "6720651",
            "6720681",
            "6720531",
            "6720517",
            "6720383",
        },
        independent_flowpaths=[
            "6720797",
            "6720795",
            "6720701",
            "6720607",
            "6720773",
            "6720525",
            "6720677",
            "6720551",
            "6719081",
            "6719063",
            "6720417",
            "6720371",
            "6722041",
        ],
        connector_segments=[],
        subdivide_candidates=[],
        upstream_merge_points=["6720879", "6720703", "6720689", "6720527", "6721863"],
        processed_flowpaths={
            "6720773",
            "6720473",
            "6722499",
            "6720795",
            "6720701",
            "6720527",
            "6720371",
            "6720683",
            "6720431",
            "6722041",
            "6720877",
            "6720433",
            "6720607",
            "6720525",
            "6722501",
            "6720651",
            "6720437",
            "6720493",
            "6720497",
            "6720515",
            "6720531",
            "6720679",
            "6719081",
            "6720581",
            "6720383",
            "6720797",
            "6720417",
            "6720689",
            "6722023",
            "6722035",
            "6720883",
            "6720879",
            "6720675",
            "6720475",
            "6722017",
            "6720681",
            "6720391",
            "6719063",
            "6720449",
            "6720503",
            "6721863",
            "6720413",
            "6720385",
            "6720407",
            "6720393",
            "6722029",
            "6722667",
            "6720517",
            "6722665",
            "6720389",
            "6719083",
            "6721871",
            "6720381",
            "6720363",
            "6720703",
            "6720677",
            "6720551",
        },
        cumulative_merge_areas={},
    )


@pytest.fixture
def sample_reference_data() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Sample reference flowpaths and divides for testing."""
    flowpath_ids = [
        "6720877",
        "6720879",
        "6720883",
        "6720683",
        "6720703",
        "6720651",
        "6720581",
        "6721871",
        "6720679",
        "6720681",
        "6720689",
        "6720515",
        "6720517",
        "6720527",
        "6720797",
        "6720795",
        "6720701",
        "6720773",
        "6720677",
        "6722499",
        "6720475",
        "6720383",
        "6720531",
        "6720381",
        "6720607",
        "6720525",
        "6720551",
        "6719081",
        "6719063",
        "6720417",
        "6720371",
        "6722041",
        "6720497",
        "6720449",
        "6719083",
        "6720433",
        "6722029",
        "6720493",
        "6720675",
        "6720437",
        "6720389",
        "6720407",
        "6722023",
        "6720413",
        "6720503",
        "6720393",
        "6720385",
        "6720391",
        "6720473",
        "6720431",
    ]

    num_fps = len(flowpath_ids)

    flowpaths = pd.DataFrame(
        {
            "flowpath_id": flowpath_ids,
            "hydroseq": list(range(1, num_fps + 1)),
            "dnhydroseq": [0] + list(range(1, num_fps)),
            "lengthkm": [1.0 + i * 0.1 for i in range(num_fps)],
            "areasqkm": [2.0 + i * 0.5 for i in range(num_fps)],
            "totdasqkm": [10.0 + i * 5.0 for i in range(num_fps)],
            "streamorder": [1 + (i % 3) for i in range(num_fps)],
            "VPUID": ["01"] * num_fps,
            "geometry": [LineString([(i, i), (i + 1, i + 1)]) for i in range(num_fps)],
        }
    )

    divides = pd.DataFrame(
        {
            "divide_id": flowpath_ids,
            "areasqkm": [2.0 + i * 0.5 for i in range(num_fps)],
            "geometry": [Polygon([(i, i), (i + 2, i), (i + 2, i + 2), (i, i + 2)]) for i in range(num_fps)],
        }
    )

    flowpaths_gdf = gpd.GeoDataFrame(flowpaths, crs="EPSG:5070")
    flowpaths_gdf = flowpaths_gdf.set_index("flowpath_id", drop=False)
    divides_gdf = gpd.GeoDataFrame(divides, crs="EPSG:5070")

    return flowpaths_gdf, divides_gdf


@pytest.fixture
def flowpaths_with_invalid_geometries() -> gpd.GeoDataFrame:
    """Flowpaths with various invalid geometry types for edge case testing."""
    data = {
        "flowpath_id": ["fp1", "fp2", "fp3"],
        "hydroseq": [1, 2, 3],
        "dnhydroseq": [0, 1, 2],
        "lengthkm": [1.0, 2.0, 3.0],
        "areasqkm": [1.0, 2.0, 3.0],
        "totdasqkm": [10.0, 20.0, 30.0],
        "streamorder": [1, 1, 2],
        "VPUID": ["01"] * 3,
        "geometry": [
            LineString([(0, 0), (1, 1)]),  # Valid
            None,  # NULL
            LineString([(0, 0), (0, 0)]),  # Zero length
        ],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:5070")


@pytest.fixture
def flowpaths_with_null_values() -> gpd.GeoDataFrame:
    """Flowpaths with NULL/NaN values in numeric columns."""
    data = {
        "flowpath_id": ["fp1", "fp2", "fp3", "fp4"],
        "hydroseq": [1, 2, 3, 4],
        "dnhydroseq": [0, 1, 2, 3],
        "lengthkm": [1.0, np.nan, 3.0, 4.0],  # NaN value
        "areasqkm": [1.0, 2.0, np.nan, 4.0],  # NaN value
        "totdasqkm": [10.0, 20.0, 30.0, np.nan],  # NaN value
        "streamorder": [1, 1, 2, 2],
        "VPUID": ["01"] * 4,
        "geometry": [LineString([(i, i), (i + 1, i + 1)]) for i in range(4)],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:5070")


@pytest.fixture
def flowpaths_with_negative_values() -> gpd.GeoDataFrame:
    """Flowpaths with negative values in area/length columns."""
    data = {
        "flowpath_id": ["fp1", "fp2", "fp3", "fp4"],
        "hydroseq": [1, 2, 3, 4],
        "dnhydroseq": [0, 1, 2, 3],
        "lengthkm": [-1.0, 2.0, 3.0, 4.0],  # Negative
        "areasqkm": [1.0, -2.0, 3.0, 4.0],  # Negative
        "totdasqkm": [10.0, 20.0, -30.0, 40.0],  # Negative
        "streamorder": [1, 1, 2, 2],
        "VPUID": ["01"] * 4,
        "geometry": [LineString([(i, i), (i + 1, i + 1)]) for i in range(4)],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:5070")


@pytest.fixture
def flowpaths_with_zero_values() -> gpd.GeoDataFrame:
    """Flowpaths with zero values in area/length columns."""
    data = {
        "flowpath_id": ["fp1", "fp2", "fp3", "fp4"],
        "hydroseq": [1, 2, 3, 4],
        "dnhydroseq": [0, 1, 2, 3],
        "lengthkm": [0.0, 2.0, 3.0, 4.0],  # Zero
        "areasqkm": [1.0, 0.0, 3.0, 4.0],  # Zero
        "totdasqkm": [0.0, 20.0, 30.0, 40.0],  # Zero
        "streamorder": [1, 1, 2, 2],
        "VPUID": ["01"] * 4,
        "geometry": [LineString([(i, i), (i + 1, i + 1)]) for i in range(4)],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:5070")


@pytest.fixture
def network_with_cycle() -> tuple[gpd.GeoDataFrame, dict[str, list[str]]]:
    """Network with circular reference (cycle)."""
    data = {
        "flowpath_id": ["fp1", "fp2", "fp3"],
        "hydroseq": [1, 2, 3],
        "dnhydroseq": [3, 1, 2],  # Cycle: fp1 -> fp3 -> fp2 -> fp1
        "lengthkm": [1.0, 2.0, 3.0],
        "areasqkm": [1.0, 2.0, 3.0],
        "totdasqkm": [10.0, 20.0, 30.0],
        "streamorder": [1, 1, 2],
        "VPUID": ["01"] * 3,
        "geometry": [LineString([(i, i), (i + 1, i + 1)]) for i in range(3)],
    }
    flowpaths = gpd.GeoDataFrame(data, crs="EPSG:5070")

    # Cycle in upstream network
    upstream_network = {
        "fp1": ["fp2"],
        "fp2": ["fp3"],
        "fp3": ["fp1"],  # Creates cycle
    }

    return flowpaths, upstream_network


@pytest.fixture
def disconnected_network() -> tuple[gpd.GeoDataFrame, dict[str, list[str]]]:
    """Network with disconnected components."""
    data = {
        "flowpath_id": ["net1_fp1", "net1_fp2", "net2_fp1", "net2_fp2"],
        "hydroseq": [1, 2, 100, 101],
        "dnhydroseq": [0, 1, 0, 100],  # Two separate networks
        "lengthkm": [1.0, 2.0, 3.0, 4.0],
        "areasqkm": [1.0, 2.0, 3.0, 4.0],
        "totdasqkm": [10.0, 20.0, 30.0, 40.0],
        "streamorder": [1, 2, 1, 2],
        "VPUID": ["01"] * 4,
        "geometry": [LineString([(i, i), (i + 1, i + 1)]) for i in range(4)],
    }
    flowpaths = gpd.GeoDataFrame(data, crs="EPSG:5070")

    upstream_network = {
        "net1_fp1": [],
        "net1_fp2": ["net1_fp1"],
        "net2_fp1": [],  # Separate network
        "net2_fp2": ["net2_fp1"],
    }

    return flowpaths, upstream_network


@pytest.fixture
def flowpaths_with_duplicate_ids() -> gpd.GeoDataFrame:
    """Flowpaths with duplicate IDs."""
    data = {
        "flowpath_id": ["fp1", "fp2", "fp1", "fp3"],  # fp1 appears twice
        "hydroseq": [1, 2, 3, 4],
        "dnhydroseq": [0, 1, 2, 3],
        "lengthkm": [1.0, 2.0, 3.0, 4.0],
        "areasqkm": [1.0, 2.0, 3.0, 4.0],
        "totdasqkm": [10.0, 20.0, 30.0, 40.0],
        "streamorder": [1, 1, 2, 2],
        "VPUID": ["01"] * 4,
        "geometry": [LineString([(i, i), (i + 1, i + 1)]) for i in range(4)],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:5070")


@pytest.fixture
def single_flowpath_network() -> gpd.GeoDataFrame:
    """Minimal network with just one flowpath."""
    data = {
        "flowpath_id": ["fp1"],
        "hydroseq": [1],
        "dnhydroseq": [0],
        "lengthkm": [1.0],
        "areasqkm": [1.0],
        "totdasqkm": [10.0],
        "streamorder": [1],
        "VPUID": ["01"],
        "geometry": [LineString([(0, 0), (1, 1)])],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:5070")


@pytest.fixture
def contradictory_classifications() -> Classifications:
    """Classifications with same flowpath in multiple conflicting categories."""
    return Classifications(
        aggregation_pairs=[("fp1", "fp2")],
        virtual_flowpaths={"fp1"},  # fp1 is also in aggregation!
        independent_flowpaths=["fp1"],  # fp1 is also independent!
        connector_segments=["fp2"],  # fp2 is also in aggregation!
        subdivide_candidates=[],
        upstream_merge_points=["fp2"],
        processed_flowpaths={"fp1", "fp2"},
        cumulative_merge_areas={},
    )


@pytest.fixture
def missing_reference_flowpath_ids() -> tuple[Classifications, list[str]]:
    """Classifications referencing flowpath IDs that don't exist in reference data."""
    classifications = Classifications(
        aggregation_pairs=[("fp_exists", "fp_missing")],  # fp_missing doesn't exist
        virtual_flowpaths=set(),
        independent_flowpaths=["fp_also_missing"],  # Also doesn't exist
        connector_segments=[],
        subdivide_candidates=[],
        upstream_merge_points=[],
        processed_flowpaths={"fp_exists", "fp_missing", "fp_also_missing"},
        cumulative_merge_areas={},
    )

    existing_ids = ["fp_exists", "fp1", "fp2", "fp3"]  # Note: missing IDs not here

    return classifications, existing_ids


@pytest.fixture
def extreme_value_flowpaths() -> gpd.GeoDataFrame:
    """Flowpaths with extreme values for length/area."""
    data = {
        "flowpath_id": ["fp1", "fp2", "fp3", "fp4"],
        "hydroseq": [1, 2, 3, 4],
        "dnhydroseq": [0, 1, 2, 3],
        "lengthkm": [1e-10, 1.0, 1e6, 1e10],  # Very small to very large
        "areasqkm": [1e-10, 1.0, 1e6, 1e10],
        "totdasqkm": [1e-10, 10.0, 1e7, 1e12],
        "streamorder": [1, 1, 2, 3],
        "VPUID": ["01"] * 4,
        "geometry": [LineString([(i, i), (i + 1, i + 1)]) for i in range(4)],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:5070")


@pytest.fixture
def multipart_geometries() -> gpd.GeoDataFrame:
    """Flowpaths with MultiLineString geometries."""
    from shapely.geometry import MultiLineString

    data = {
        "flowpath_id": ["fp1", "fp2"],
        "hydroseq": [1, 2],
        "dnhydroseq": [0, 1],
        "lengthkm": [1.0, 2.0],
        "areasqkm": [1.0, 2.0],
        "totdasqkm": [10.0, 20.0],
        "streamorder": [1, 2],
        "VPUID": ["01"] * 2,
        "geometry": [
            MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]),  # Disconnected segments
            MultiLineString([[(i, i), (i + 0.5, i + 0.5)] for i in range(10)]),  # Many small segments
        ],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:5070")


@pytest.fixture
def network_graph_for_trace_tests() -> dict[str, list[str]]:
    """Network graph specifically for test_trace.py tests."""
    return {
        "fp1": ["up1", "up2", "up3"],  # Multiple upstreams
        "fp2": ["fp1"],  # Single upstream
        "fp3": [],  # Headwater
        "fp4": ["fp3"],
        "fp5": ["fp3"],
        "order2_fp": ["order1_up1", "order1_up2"],  # Order 2 with order 1 upstreams
        "mixed_fp": ["large_order1", "small_order1", "order2_up"],  # Mixed orders
    }


@pytest.fixture
def div_ids_for_trace_tests() -> dict[str, str]:
    """Divide ID mappings for test_trace.py tests."""
    return {
        "fp1": "div1",
        "fp2": "div2",
        "fp3": "div3",
        "up1": "div_up1",
        "up2": "div_up2",
        "up3": "div_up3",
        "order1_up1": "div_o1_1",
        "order1_up2": "div_o1_2",
        "order2_fp": "div_o2",
        "mixed_fp": "div_mixed",
        "large_order1": "div_large",
        "small_order1": "div_small",
        "order2_up": "div_o2_up",
    }


@pytest.fixture
def to_process_set_for_trace_tests() -> set[str]:
    """Set of flowpaths to process for test_trace.py tests."""
    return {
        "fp1",
        "fp2",
        "fp3",
        "fp4",
        "fp5",
        "up1",
        "up2",
        "up3",
        "order1_up1",
        "order1_up2",
        "order2_fp",
        "mixed_fp",
        "large_order1",
        "small_order1",
        "order2_up",
    }


@pytest.fixture
def network_graph() -> dict[str, list[str]]:
    """Network graph for trace tests."""
    return {
        "fp1": ["up1", "up2", "up3"],
        "fp2": ["up1"],
        "up1": [],
        "up2": [],
        "up3": [],
    }


@pytest.fixture
def div_ids() -> set[str]:
    """Set of divide IDs (not dict)."""
    return {"fp1", "fp2", "up1", "up2", "up3"}


@pytest.fixture
def to_process() -> deque:
    """Queue for processing (not set)."""
    from collections import deque

    return deque(["fp1", "fp2"])


@pytest.fixture
def sample_network_graph() -> rx.PyDiGraph:
    """Create a sample PyDiGraph from network dictionary for testing."""
    graph = rx.PyDiGraph()

    # Create nodes
    node_map = {}
    network_dict = {
        "fp1": ["fp2", "fp3"],
        "fp2": ["fp4"],
        "fp3": ["fp5"],
        "fp4": [],
        "fp5": [],
    }

    # Add all nodes first
    for node_id in network_dict.keys():
        node_map[node_id] = graph.add_node(node_id)

    # Add edges based on upstream relationships
    for downstream, upstreams in network_dict.items():
        for upstream in upstreams:
            if upstream in node_map:
                graph.add_edge(node_map[downstream], node_map[upstream], None)

    return graph


@pytest.fixture
def circular_network_dict() -> dict[str, list[str]]:
    """Network dictionary with a circular reference."""
    return {
        "fp1": ["fp2"],
        "fp2": ["fp3"],
        "fp3": ["fp1"],  # Creates cycle
    }


@pytest.fixture
def valid_network_dict() -> dict[str, list[str]]:
    """Valid network dictionary without cycles."""
    return {
        "fp1": ["fp2", "fp3"],
        "fp2": ["fp4"],
        "fp3": ["fp5"],
        "fp4": [],
        "fp5": [],
    }


def dict_to_graph(network_dict: dict[str, list[str]]) -> tuple[rx.PyDiGraph, dict]:
    """Helper function to convert network dict to PyDiGraph.

    Parameters
    ----------
    network_dict : dict[str, list[str]]
        Dictionary mapping downstream nodes to lists of upstream nodes

    Returns
    -------
    rx.PyDiGraph
        Graph representation of the network
    dict
        The node inidices map
    """
    graph = rx.PyDiGraph()
    node_map = {}

    # Add all nodes
    for node_id in network_dict.keys():
        node_map[node_id] = graph.add_node(node_id)

    # Add edges
    for downstream, upstreams in network_dict.items():
        for upstream in upstreams:
            if upstream in node_map:
                graph.add_edge(node_map[upstream], node_map[downstream], None)

    return graph, node_map


def create_partition_data_from_dataframes(
    flowpaths_df: pl.DataFrame,
    divides_df: pl.DataFrame | None,
    graph: rx.PyDiGraph,
    node_indices: dict,
) -> dict:
    """Convert DataFrames to partition_data format expected by optimized trace functions."""
    # Create fp_lookup
    _fp_lookup = flowpaths_df.to_dicts()
    fp_lookup = {str(row["flowpath_id"]): row for row in _fp_lookup}

    # Add dummy shapely geometries and normalize column names
    for fp_id in fp_lookup:
        fp_lookup[fp_id]["shapely_geometry"] = LineString([(0, 0), (1, 1)])

        # Normalize column names to match what trace expects
        if "areasqkm" in fp_lookup[fp_id]:
            fp_lookup[fp_id]["area_sqkm"] = fp_lookup[fp_id]["areasqkm"]
        if "lengthkm" in fp_lookup[fp_id]:
            fp_lookup[fp_id]["length_km"] = fp_lookup[fp_id]["lengthkm"]
        if "totdasqkm" in fp_lookup[fp_id]:
            fp_lookup[fp_id]["total_da_sqkm"] = fp_lookup[fp_id]["totdasqkm"]

        # Add flowpath_toid if not present
        if "flowpath_toid" not in fp_lookup[fp_id]:
            fp_lookup[fp_id]["flowpath_toid"] = "0"

    # Create div_lookup
    div_lookup = {}
    if divides_df is not None:
        _div_lookup = divides_df.to_dicts()
        div_lookup = {str(row["divide_id"]): row for row in _div_lookup}
        for div_id in div_lookup:
            div_lookup[div_id]["shapely_geometry"] = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
            # Normalize areasqkm
            if "areasqkm" in div_lookup[div_id]:
                div_lookup[div_id]["area_sqkm"] = div_lookup[div_id]["areasqkm"]

    return {
        "subgraph": graph,
        "node_indices": node_indices,
        "fp_lookup": fp_lookup,
        "div_lookup": div_lookup,
        "flowpaths": flowpaths_df,
        "divides": divides_df,
    }


def create_partition_data_for_build_tests(
    reference_flowpaths: gpd.GeoDataFrame,
    reference_divides: gpd.GeoDataFrame,
    graph: rx.PyDiGraph,
    node_indices: dict,
) -> dict:
    """Convert test fixtures to partition_data format for _build_base_hydrofabric. NEEDS GEOMETRIES

    Parameters
    ----------
    reference_flowpaths : gpd.GeoDataFrame
        Reference flowpaths GeoDataFrame
    reference_divides : gpd.GeoDataFrame
        Reference divides GeoDataFrame
    graph : rx.PyDiGraph
        The graph object
    node_indices : dict
        Node indices mapping

    Returns
    -------
    dict
        partition_data dict ready for _build_base_hydrofabric
    """
    # Convert to Polars with WKB
    pl_flowpaths = pl.from_pandas(reference_flowpaths.to_wkb())
    pl_divides = pl.from_pandas(reference_divides.to_wkb())

    # Create fp_lookup with shapely geometries
    fp_ids = pl_flowpaths["flowpath_id"].cast(pl.Utf8).to_list()
    fp_dicts = pl_flowpaths.to_dicts()
    fp_lookup = {str(row["flowpath_id"]): row for row in fp_dicts}

    # Add shapely geometries to fp_lookup
    for fp_id, geom in zip(fp_ids, reference_flowpaths.geometry, strict=True):
        if fp_id in fp_lookup:
            fp_lookup[fp_id]["shapely_geometry"] = geom

    # Create div_lookup with shapely geometries
    div_ids = pl_divides["divide_id"].cast(pl.Utf8).to_list()
    div_dicts = pl_divides.to_dicts()
    div_lookup = {str(row["divide_id"]): row for row in div_dicts}

    # Add shapely geometries to div_lookup
    for div_id, geom in zip(div_ids, reference_divides.geometry, strict=True):
        if div_id in div_lookup:
            div_lookup[div_id]["shapely_geometry"] = geom

    return {
        "subgraph": graph,
        "node_indices": node_indices,
        "fp_lookup": fp_lookup,
        "div_lookup": div_lookup,
        "flowpaths": pl_flowpaths,
        "divides": pl_divides,
    }
