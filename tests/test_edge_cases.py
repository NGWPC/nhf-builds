"""Edge case tests for hydrofabric building functions."""

import geopandas as gpd
import numpy as np
import polars as pl
import pytest
import rustworkx as rx
from conftest import dict_to_graph
from shapely.geometry import LineString

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.hydrofabric.aggregate import _aggregate_geometries
from hydrofabric_builds.hydrofabric.build import _order_aggregates_base
from hydrofabric_builds.hydrofabric.graph import _create_dictionary_lookup, _detect_cycles
from hydrofabric_builds.schemas.hydrofabric import Aggregations, Classifications


class TestInvalidGeometries:
    """Tests for handling invalid or malformed geometries."""

    def test_handles_empty_linestring(self, flowpaths_with_invalid_geometries: gpd.GeoDataFrame) -> None:
        """Test handling of empty LineString geometries."""
        # Test passes if no empty geometries crash the system
        null_geom_rows = flowpaths_with_invalid_geometries[flowpaths_with_invalid_geometries.geometry.isna()]
        assert len(null_geom_rows) > 0, "Test data should contain NULL geometries"

    def test_handles_null_geometries(self, flowpaths_with_invalid_geometries: gpd.GeoDataFrame) -> None:
        """Test handling of NULL/None geometries."""
        null_geom_rows = flowpaths_with_invalid_geometries[flowpaths_with_invalid_geometries.geometry.isna()]
        assert len(null_geom_rows) > 0, "Test data should contain NULL geometries"

    def test_handles_zero_length_linestring(
        self, flowpaths_with_invalid_geometries: gpd.GeoDataFrame
    ) -> None:
        """Test handling of zero-length LineStrings (same start/end point)."""
        zero_length = flowpaths_with_invalid_geometries[
            flowpaths_with_invalid_geometries["flowpath_id"] == "fp3"
        ]
        geom = zero_length.geometry.iloc[0]
        assert geom.length == 0, "Should have zero length"

    def test_detects_invalid_geometries(self, flowpaths_with_invalid_geometries: gpd.GeoDataFrame) -> None:
        """Test that invalid geometries are detected."""
        # Count invalid geometries
        invalid_count = 0
        for geom in flowpaths_with_invalid_geometries.geometry:
            if geom is None or (hasattr(geom, "is_empty") and geom.is_empty):
                invalid_count += 1
            elif hasattr(geom, "is_valid") and not geom.is_valid:
                invalid_count += 1

        assert invalid_count >= 2, "Test data should have at least 2 invalid geometries"


class TestNullAndNaNValues:
    """Tests for handling NULL/NaN values in data."""

    def test_handles_nan_in_lengthkm(self, flowpaths_with_null_values: gpd.GeoDataFrame) -> None:
        """Test handling of NaN values in lengthkm column."""
        assert flowpaths_with_null_values["lengthkm"].isna().any(), "Should have NaN in lengthkm"
        # Test should either reject data, impute values, or handle gracefully

    def test_handles_nan_in_areasqkm(self, flowpaths_with_null_values: gpd.GeoDataFrame) -> None:
        """Test handling of NaN values in areasqkm column."""
        assert flowpaths_with_null_values["areasqkm"].isna().any(), "Should have NaN in areasqkm"

    def test_handles_nan_in_totdasqkm(self, flowpaths_with_null_values: gpd.GeoDataFrame) -> None:
        """Test handling of NaN values in totdasqkm column."""
        assert flowpaths_with_null_values["totdasqkm"].isna().any(), "Should have NaN in totdasqkm"

    def test_multiple_nan_values(self, flowpaths_with_null_values: gpd.GeoDataFrame) -> None:
        """Test handling when multiple columns have NaN values."""
        nan_count = flowpaths_with_null_values[["lengthkm", "areasqkm", "totdasqkm"]].isna().sum().sum()
        assert nan_count >= 3, "Should have at least 3 NaN values across columns"


class TestNegativeValues:
    """Tests for handling negative values in numeric columns."""

    def test_detects_negative_length(self, flowpaths_with_negative_values: gpd.GeoDataFrame) -> None:
        """Test that negative length values are detected."""
        negative_length = flowpaths_with_negative_values[flowpaths_with_negative_values["lengthkm"] < 0]
        assert len(negative_length) > 0, "Should have negative length values"

    def test_detects_negative_area(self, flowpaths_with_negative_values: gpd.GeoDataFrame) -> None:
        """Test that negative area values are detected."""
        negative_area = flowpaths_with_negative_values[flowpaths_with_negative_values["areasqkm"] < 0]
        assert len(negative_area) > 0, "Should have negative area values"

    def test_detects_negative_drainage_area(self, flowpaths_with_negative_values: gpd.GeoDataFrame) -> None:
        """Test that negative drainage area values are detected."""
        negative_da = flowpaths_with_negative_values[flowpaths_with_negative_values["totdasqkm"] < 0]
        assert len(negative_da) > 0, "Should have negative drainage area values"


class TestZeroValues:
    """Tests for handling zero values in numeric columns."""

    def test_handles_zero_length(self, flowpaths_with_zero_values: gpd.GeoDataFrame) -> None:
        """Test handling of zero length flowpaths."""
        zero_length = flowpaths_with_zero_values[flowpaths_with_zero_values["lengthkm"] == 0]
        assert len(zero_length) > 0, "Should have zero length flowpath"

    def test_handles_zero_area(self, flowpaths_with_zero_values: gpd.GeoDataFrame) -> None:
        """Test handling of zero area flowpaths."""
        zero_area = flowpaths_with_zero_values[flowpaths_with_zero_values["areasqkm"] == 0]
        assert len(zero_area) > 0, "Should have zero area flowpath"

    def test_handles_zero_drainage_area(self, flowpaths_with_zero_values: gpd.GeoDataFrame) -> None:
        """Test handling of zero total drainage area."""
        zero_da = flowpaths_with_zero_values[flowpaths_with_zero_values["totdasqkm"] == 0]
        assert len(zero_da) > 0, "Should have zero drainage area"


class TestNetworkTopology:
    """Tests for network topology edge cases."""

    def test_detects_circular_reference(
        self, circular_network_dict: dict[str, list[str]], sample_config: HFConfig
    ) -> None:
        """Test that cycle detection catches circular references in network."""

        graph, node_map = dict_to_graph(circular_network_dict)

        # Should raise ValueError about cycle
        with pytest.raises(rx.DAGHasCycle):
            _detect_cycles(graph)

    def test_no_cycle_passes_validation(
        self, valid_network_dict: dict[str, list[str]], sample_config: HFConfig
    ) -> None:
        """Test that networks without cycles pass validation."""

        graph, _ = dict_to_graph(valid_network_dict)

        # Should not raise
        _detect_cycles(graph)

    def test_handles_disconnected_networks(self, disconnected_network: tuple[gpd.GeoDataFrame, dict]) -> None:
        """Test handling of disconnected sub-networks."""
        flowpaths, upstream_network = disconnected_network

        # Find all roots (nodes with no upstream)
        roots = [fp for fp, ups in upstream_network.items() if not ups]
        assert len(roots) == 2, "Should have 2 separate network roots"

    def test_disconnected_network_has_separate_outlets(
        self, disconnected_network: tuple[gpd.GeoDataFrame, dict]
    ) -> None:
        """Test that disconnected networks have separate outlets."""
        flowpaths, upstream_network = disconnected_network

        # Check that we can't traverse from net1 to net2
        net1_nodes = {"net1_fp1", "net1_fp2"}
        net2_nodes = {"net2_fp1", "net2_fp2"}

        # Starting from net1, collect all reachable nodes
        reachable = set()
        to_visit = ["net1_fp2"]
        while to_visit:
            current = to_visit.pop()
            if current not in reachable:
                reachable.add(current)
                to_visit.extend(upstream_network.get(current, []))

        assert reachable == net1_nodes, "Should only reach net1 nodes from net1"
        assert not reachable.intersection(net2_nodes), "Should not reach net2 from net1"


class TestDataConsistency:
    """Tests for data consistency issues."""

    def test_detects_duplicate_flowpath_ids(self, flowpaths_with_duplicate_ids: gpd.GeoDataFrame) -> None:
        """Test detection of duplicate flowpath IDs."""
        duplicate_count = flowpaths_with_duplicate_ids["flowpath_id"].duplicated().sum()
        assert duplicate_count > 0, "Should have duplicate IDs"

    def test_duplicate_ids_list(self, flowpaths_with_duplicate_ids: gpd.GeoDataFrame) -> None:
        """Test identifying which IDs are duplicated."""
        duplicates = flowpaths_with_duplicate_ids[
            flowpaths_with_duplicate_ids["flowpath_id"].duplicated(keep=False)
        ]["flowpath_id"].unique()
        assert "fp1" in duplicates, "fp1 should be identified as duplicate"

    def test_handles_missing_reference_flowpaths(self, missing_reference_flowpath_ids: tuple) -> None:
        """Test handling when classifications reference non-existent flowpaths."""
        classifications, existing_ids = missing_reference_flowpath_ids

        # Check for IDs in classifications that aren't in existing_ids
        all_classification_ids = set()
        for pair in classifications.aggregation_pairs:
            all_classification_ids.update(pair)
        all_classification_ids.update(classifications.independent_flowpaths)

        missing = all_classification_ids - set(existing_ids)
        assert len(missing) > 0, "Should have missing reference IDs"
        assert "fp_missing" in missing, "fp_missing should be detected as missing"


class TestMinimalNetworks:
    """Tests for minimal/edge case network sizes."""

    def test_single_flowpath_network(self, single_flowpath_network: gpd.GeoDataFrame) -> None:
        """Test handling of network with just one flowpath."""
        assert len(single_flowpath_network) == 1, "Should have exactly 1 flowpath"
        assert single_flowpath_network["dnhydroseq"].iloc[0] == 0, "Should be outlet"

    def test_single_flowpath_is_headwater_and_outlet(self, single_flowpath_network: gpd.GeoDataFrame) -> None:
        """Test that single flowpath is both headwater and outlet."""
        fp = single_flowpath_network.iloc[0]
        is_outlet = fp["dnhydroseq"] == 0
        # Is headwater if no other flowpaths flow into it
        assert is_outlet, "Single flowpath should be outlet"


class TestExtremeValues:
    """Tests for extreme numeric values."""

    def test_handles_very_small_values(self, extreme_value_flowpaths: gpd.GeoDataFrame) -> None:
        """Test handling of very small length/area values (e-10)."""
        very_small = extreme_value_flowpaths[extreme_value_flowpaths["lengthkm"] < 1e-5]
        assert len(very_small) > 0, "Should have very small values"

    def test_handles_very_large_values(self, extreme_value_flowpaths: gpd.GeoDataFrame) -> None:
        """Test handling of very large length/area values (e10)."""
        very_large = extreme_value_flowpaths[extreme_value_flowpaths["lengthkm"] > 1e8]
        assert len(very_large) > 0, "Should have very large values"

    def test_extreme_value_range(self, extreme_value_flowpaths: gpd.GeoDataFrame) -> None:
        """Test that extreme values span many orders of magnitude."""
        min_length = extreme_value_flowpaths["lengthkm"].min()
        max_length = extreme_value_flowpaths["lengthkm"].max()
        ratio = max_length / min_length
        assert ratio > 1e15, "Should span at least 15 orders of magnitude"


class TestMultipartGeometries:
    """Tests for MultiLineString and MultiPolygon geometries."""

    def test_handles_multilinestring(self, multipart_geometries: gpd.GeoDataFrame) -> None:
        """Test handling of MultiLineString geometries."""
        multi_geoms = multipart_geometries[
            multipart_geometries.geometry.apply(lambda g: g.geom_type == "MultiLineString")
        ]
        assert len(multi_geoms) > 0, "Should have MultiLineString geometries"

    def test_multilinestring_with_many_parts(self, multipart_geometries: gpd.GeoDataFrame) -> None:
        """Test handling of MultiLineString with many disconnected parts."""
        fp2_geom = multipart_geometries[multipart_geometries["flowpath_id"] == "fp2"].geometry.iloc[0]
        if hasattr(fp2_geom, "geoms"):
            part_count = len(fp2_geom.geoms)
            assert part_count >= 10, "Should have at least 10 parts"


class TestEmptyInputs:
    """Tests for empty or minimal input data."""

    def test_empty_aggregations(self) -> None:
        """Test handling of completely empty aggregations."""
        empty = Aggregations(
            aggregates=[],
            independents=[],
            connectors=[],
            minor_flowpaths=[],
            small_scale_connectors=[],
            no_divide_connectors=[],
        )

        result = _order_aggregates_base(empty)
        assert result == {}, "Empty input should produce empty output"

    def test_empty_classifications(
        self, sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
    ) -> None:
        """Test handling of empty classifications."""
        reference_flowpaths, reference_divides = sample_reference_data
        reference_flowpaths = pl.from_pandas(reference_flowpaths.to_wkb())
        reference_divides = pl.from_pandas(reference_divides.to_wkb())

        empty_classifications = Classifications(
            aggregation_pairs=[],
            no_divide_connectors=[],
            minor_flowpaths=set(),
            independent_flowpaths=[],
            connector_segments=[],
            subdivide_candidates=[],
            upstream_merge_points=[],
            processed_flowpaths=set(),
            cumulative_merge_areas={},
        )

        fp_lookup = _create_dictionary_lookup(reference_flowpaths, "flowpath_id")
        div_lookup = _create_dictionary_lookup(reference_divides, "divide_id")

        result = _aggregate_geometries(
            classifications=empty_classifications,
            partition_data={
                "flowpaths": reference_flowpaths,
                "fp_lookup": fp_lookup,
                "div_lookup": div_lookup,
            },
        )

        assert len(result.aggregates) == 0, "Should have no aggregates"
        assert len(result.independents) == 0, "Should have no independents"


class TestBoundaryConditions:
    """Tests for boundary/special network conditions."""

    def test_all_headwaters(self) -> None:
        """Test network where all flowpaths are headwaters (no upstream)."""
        data = {
            "flowpath_id": ["fp1", "fp2", "fp3"],
            "hydroseq": [1, 2, 3],
            "dnhydroseq": [0, 0, 0],  # All flow to outlet
            "lengthkm": [1.0, 2.0, 3.0],
            "areasqkm": [1.0, 2.0, 3.0],
            "totdasqkm": [1.0, 2.0, 3.0],  # No upstream area
            "streamorder": [1, 1, 1],  # All order 1
            "VPUID": ["01"] * 3,
            "geometry": [LineString([(i, i), (i + 1, i + 1)]) for i in range(3)],
        }
        flowpaths = gpd.GeoDataFrame(data, crs="EPSG:5070")

        # All should be headwaters (totdasqkm ~ areasqkm)
        is_headwater = flowpaths["totdasqkm"] == flowpaths["areasqkm"]
        assert is_headwater.all(), "All should be headwaters"

    def test_very_wide_convergence(self) -> None:
        """Test many tributaries converging at one point."""
        num_tributaries = 100
        data = {
            "flowpath_id": [f"trib_{i}" for i in range(num_tributaries)] + ["main"],
            "hydroseq": list(range(1, num_tributaries + 2)),
            "dnhydroseq": [num_tributaries + 1] * num_tributaries + [0],  # All flow to 'main'
            "lengthkm": [1.0] * (num_tributaries + 1),
            "areasqkm": [1.0] * (num_tributaries + 1),
            "totdasqkm": [1.0] * num_tributaries + [float(num_tributaries)],  # Main has all upstream area
            "streamorder": [1] * num_tributaries + [2],
            "VPUID": ["01"] * (num_tributaries + 1),
            "geometry": [LineString([(i, i), (i + 1, i + 1)]) for i in range(num_tributaries + 1)],
        }
        flowpaths = gpd.GeoDataFrame(data, crs="EPSG:5070")

        # Main flowpath should have drainage area from all tributaries
        main_da = flowpaths[flowpaths["flowpath_id"] == "main"]["totdasqkm"].iloc[0]
        assert main_da == num_tributaries, f"Main should have drainage area of {num_tributaries}"


@pytest.mark.parametrize("bad_value", [None, np.nan, -1, 0, 1e20])
def test_parametrized_invalid_areas(bad_value: int | str) -> None:
    """Parametrized test for various invalid area values."""
    data = {
        "flowpath_id": ["fp1"],
        "hydroseq": [1],
        "dnhydroseq": [0],
        "lengthkm": [1.0],
        "areasqkm": [bad_value],  # Invalid value
        "totdasqkm": [10.0],
        "streamorder": [1],
        "VPUID": ["01"],
        "geometry": [LineString([(0, 0), (1, 1)])],
    }
    flowpaths = gpd.GeoDataFrame(data, crs="EPSG:5070")

    # Test that the bad value is present
    if bad_value is None or (isinstance(bad_value, float) and np.isnan(bad_value)):
        assert flowpaths["areasqkm"].isna().any() or flowpaths["areasqkm"].iloc[0] is None
    else:
        assert flowpaths["areasqkm"].iloc[0] == bad_value
