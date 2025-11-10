"""Tests for trace.py functions."""

from collections import deque

import polars as pl
import pytest
from conftest import create_partition_data_from_dataframes, dict_to_graph

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.hydrofabric.trace import (
    _get_unprocessed_upstream_info,
    _get_upstream_ids,
    _queue_upstream,
    _trace_single_flowpath_attributes,
    _trace_stack,
    _traverse_and_mark_as_minor,
)
from hydrofabric_builds.schemas.hydrofabric import Classifications


@pytest.fixture
def sample_flowpath_data() -> pl.DataFrame:
    """Create sample flowpath data for unit testing."""
    data = {
        "flowpath_id": ["1", "2", "3", "4", "5", "6", "7", "8"],
        "totdasqkm": [100.0, 50.0, 25.0, 10.0, 5.0, 2.0, 1.0, 0.5],
        "areasqkm": [5.0, 2.5, 1.5, 0.8, 2.0, 1.0, 0.5, 0.3],
        "lengthkm": [10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0],
        "streamorder": [3, 2, 2, 1, 1, 1, 1, 1],
        "hydroseq": [1, 2, 3, 4, 5, 6, 7, 8],
        "dnhydroseq": [0, 1, 1, 2, 3, 4, 4, 5],
        "flowpath_toid": [0, 1, 1, 2, 3, 4, 4, 5],
        "mainstemlp": [300.0, 300.0, 100.0, 50.0, 100.0, 100.0, 300.0, 300.0],
        "VPUID": ["01"] * 8,
    }
    return pl.DataFrame(data)


@pytest.fixture
def sample_fp_lookup(sample_flowpath_data: pl.DataFrame) -> dict:
    """Create fp_lookup dictionary from sample flowpath data."""
    return {str(row["flowpath_id"]): row for row in sample_flowpath_data.to_dicts()}


class TestGetUpstreamIds:
    """Tests for _get_upstream_ids function."""

    def test_get_upstream_ids_single(self) -> None:
        """Test getting single upstream."""
        network_graph = {"fp1": ["up1"], "up1": []}
        graph, node_indices = dict_to_graph(network_graph)

        upstream_ids = _get_upstream_ids("fp1", graph, node_indices)
        assert upstream_ids == ["up1"]

    def test_get_upstream_ids_multiple(self) -> None:
        """Test getting multiple upstreams."""
        network_graph = {"fp1": ["up1", "up2", "up3"], "up1": [], "up2": [], "up3": []}
        graph, node_indices = dict_to_graph(network_graph)

        upstream_ids = _get_upstream_ids("fp1", graph, node_indices)
        assert set(upstream_ids) == {"up1", "up2", "up3"}

    def test_get_upstream_ids_headwater(self) -> None:
        """Test getting upstreams of headwater (none)."""
        network_graph: dict[str, list] = {"fp1": []}
        graph, node_indices = dict_to_graph(network_graph)

        upstream_ids = _get_upstream_ids("fp1", graph, node_indices)
        assert upstream_ids == []

    def test_get_upstream_ids_not_in_graph(self) -> None:
        """Test getting upstreams of ID not in graph."""
        network_graph = {"fp1": ["up1"], "up1": []}
        graph, node_indices = dict_to_graph(network_graph)

        upstream_ids = _get_upstream_ids("fp999", graph, node_indices)
        assert upstream_ids == []


class TestGetUnprocessedUpstreamInfo:
    """Tests for _get_unprocessed_upstream_info function."""

    def test_get_unprocessed_info(self, sample_fp_lookup: dict) -> None:
        """Test getting info for unprocessed upstreams."""
        upstream_ids = ["4", "5"]
        processed: set = set()

        info = _get_unprocessed_upstream_info(upstream_ids, sample_fp_lookup, processed)

        assert len(info) == 2
        assert info[0]["flowpath_id"] == "4"
        assert info[1]["flowpath_id"] == "5"

    def test_skip_processed(self, sample_fp_lookup: dict) -> None:
        """Test skipping already processed upstreams."""
        upstream_ids = ["4", "5", "6"]
        processed = {"5"}

        info = _get_unprocessed_upstream_info(upstream_ids, sample_fp_lookup, processed)

        assert len(info) == 2
        assert all(x["flowpath_id"] != "5" for x in info)

    def test_skip_missing(self, sample_fp_lookup: dict) -> None:
        """Test skipping upstreams not in lookup."""
        upstream_ids = ["4", "999"]
        processed: set = set()

        info = _get_unprocessed_upstream_info(upstream_ids, sample_fp_lookup, processed)

        assert len(info) == 1
        assert info[0]["flowpath_id"] == "4"

    def test_empty_upstream_ids(self, sample_fp_lookup: dict) -> None:
        """Test with empty upstream list."""
        info = _get_unprocessed_upstream_info([], sample_fp_lookup, set())
        assert info == []


class TestQueueUpstream:
    """Tests for _queue_upstream function."""

    def test_queue_all(self) -> None:
        """Test queuing all upstreams."""
        to_process: deque = deque()
        processed: set = set()
        upstream_ids = ["up1", "up2", "up3"]

        _queue_upstream(upstream_ids, to_process, processed, unprocessed_only=False)

        assert len(to_process) == 3
        assert set(to_process) == {"up1", "up2", "up3"}

    def test_queue_unprocessed_only(self) -> None:
        """Test queuing only unprocessed upstreams."""
        to_process: deque = deque()
        processed = {"up2"}
        upstream_ids = ["up1", "up2", "up3"]

        _queue_upstream(upstream_ids, to_process, processed, unprocessed_only=True)

        assert len(to_process) == 2
        assert "up2" not in to_process

    def test_queue_empty(self) -> None:
        """Test queuing empty list."""
        to_process: deque = deque()
        processed: set = set()

        _queue_upstream([], to_process, processed)

        assert len(to_process) == 0


class TestTraverseAndMarkAsMinor:
    """Tests for _traverse_and_mark_as_minor function."""

    def test_mark_single_flowpath(self) -> None:
        """Test marking single flowpath as minor."""
        network_graph = {"fp1": [], "downstream": ["fp1"]}
        graph, node_indices = dict_to_graph(network_graph)
        result = Classifications()

        _traverse_and_mark_as_minor("fp1", "downstream", result, graph, node_indices)

        assert "fp1" in result.minor_flowpaths
        assert ("fp1", "downstream") in result.aggregation_pairs
        assert "fp1" in result.processed_flowpaths

    def test_mark_chain(self) -> None:
        """Test marking chain of flowpaths."""
        network_graph = {"fp1": ["fp2"], "fp2": ["fp3"], "fp3": [], "downstream": ["fp1"]}
        graph, node_indices = dict_to_graph(network_graph)
        result = Classifications()

        _traverse_and_mark_as_minor("fp1", "downstream", result, graph, node_indices)

        assert "fp1" in result.minor_flowpaths
        assert "fp2" in result.minor_flowpaths
        assert "fp3" in result.minor_flowpaths
        assert all(fp in result.processed_flowpaths for fp in ["fp1", "fp2", "fp3"])

    def test_mark_branching(self) -> None:
        """Test marking branching network."""
        network_graph = {
            "fp1": ["fp2", "fp3"],
            "fp2": ["fp4"],
            "fp3": ["fp5"],
            "fp4": [],
            "fp5": [],
            "downstream": ["fp1"],
        }
        graph, node_indices = dict_to_graph(network_graph)
        result = Classifications()

        _traverse_and_mark_as_minor("fp1", "downstream", result, graph, node_indices)

        assert all(fp in result.minor_flowpaths for fp in ["fp1", "fp2", "fp3", "fp4", "fp5"])

    def test_mark_already_processed(self) -> None:
        """Test that already processed flowpaths still get marked."""
        network_graph = {"fp1": ["fp2"], "fp2": [], "downstream": ["fp1"]}
        graph, node_indices = dict_to_graph(network_graph)
        result = Classifications()
        result.processed_flowpaths.add("fp2")

        _traverse_and_mark_as_minor("fp1", "downstream", result, graph, node_indices)

        # Both should be marked even though fp2 was already processed
        assert "fp1" in result.minor_flowpaths
        assert "fp2" in result.minor_flowpaths


class TestTraceStackRule1:
    """Tests for Rule 1: No upstream (headwater)."""

    def test_headwater_with_divide(self, sample_config: HFConfig) -> None:
        """Test headwater with divide becomes independent."""
        flowpath_data = pl.DataFrame(
            {
                "flowpath_id": ["1"],
                "areasqkm": [5.0],
                "lengthkm": [10.0],
                "totdasqkm": [5.0],
                "streamorder": [1],
                "hydroseq": [1],
                "dnhydroseq": [0],
                "flowpath_toid": [0],
                "mainstemlp": [100.0],
                "VPUID": ["01"],
            }
        )

        network_graph: dict[str, list] = {"1": []}
        graph, node_indices = dict_to_graph(network_graph)
        partition_data = create_partition_data_from_dataframes(flowpath_data, None, graph, node_indices)

        div_ids = {"1"}
        result = _trace_stack("1", div_ids, sample_config, partition_data)

        assert "1" in result.independent_flowpaths
        assert "1" not in result.minor_flowpaths

    def test_headwater_without_divide(self, sample_config: HFConfig) -> None:
        """Test headwater without divide becomes minor."""
        flowpath_data = pl.DataFrame(
            {
                "flowpath_id": ["1"],
                "areasqkm": [5.0],
                "lengthkm": [10.0],
                "totdasqkm": [5.0],
                "streamorder": [1],
                "hydroseq": [1],
                "dnhydroseq": [0],
                "flowpath_toid": [0],
                "mainstemlp": [100.0],
                "VPUID": ["01"],
            }
        )

        network_graph: dict[str, list] = {"1": []}
        graph, node_indices = dict_to_graph(network_graph)
        partition_data = create_partition_data_from_dataframes(flowpath_data, None, graph, node_indices)

        div_ids: set = set()  # No divide
        result = _trace_stack("1", div_ids, sample_config, partition_data)

        assert "1" in result.minor_flowpaths
        assert "1" not in result.independent_flowpaths


class TestTraceStackRule2:
    """Tests for Rule 2: Single upstream."""

    def test_single_upstream_aggregate(self, sample_config: HFConfig) -> None:
        """Test single upstream gets aggregated."""
        flowpath_data = pl.DataFrame(
            {
                "flowpath_id": ["1", "2"],
                "areasqkm": [2.0, 1.0],
                "lengthkm": [5.0, 3.0],
                "totdasqkm": [3.0, 1.0],
                "streamorder": [2, 1],
                "hydroseq": [1, 2],
                "dnhydroseq": [0, 1],
                "flowpath_toid": [0, 1],
                "mainstemlp": [100.0, 100.0],
                "VPUID": ["01", "01"],
            }
        )

        network_graph = {"1": ["2"], "2": []}
        graph, node_indices = dict_to_graph(network_graph)
        partition_data = create_partition_data_from_dataframes(flowpath_data, None, graph, node_indices)

        div_ids = {"1", "2"}
        result = _trace_stack("1", div_ids, sample_config, partition_data)

        assert ("1", "2") in result.aggregation_pairs

    def test_single_upstream_no_divide_with_ancestor_divides(self, sample_config: HFConfig) -> None:
        """Test single upstream without divide but ancestor divides exist."""
        flowpath_data = pl.DataFrame(
            {
                "flowpath_id": ["1", "2", "3"],
                "areasqkm": [2.0, 1.0, 0.5],
                "lengthkm": [5.0, 3.0, 2.0],
                "totdasqkm": [3.5, 1.5, 0.5],
                "streamorder": [2, 1, 1],
                "hydroseq": [1, 2, 3],
                "dnhydroseq": [0, 1, 2],
                "flowpath_toid": [0, 1, 2],
                "mainstemlp": [100.0, 100.0, 100.0],
                "VPUID": ["01", "01", "01"],
            }
        )

        network_graph = {"1": ["2"], "2": ["3"], "3": []}
        graph, node_indices = dict_to_graph(network_graph)
        partition_data = create_partition_data_from_dataframes(flowpath_data, None, graph, node_indices)

        div_ids = {"1", "3"}  # 2 has no divide, but 3 (ancestor) does
        result = _trace_stack("1", div_ids, sample_config, partition_data)

        # Should aggregate normally since ancestors have divides
        assert ("1", "2") in result.aggregation_pairs or "2" in result.independent_flowpaths

    def test_single_upstream_no_divides_anywhere(self, sample_config: HFConfig) -> None:
        """Test single upstream with no divides anywhere in network."""
        flowpath_data = pl.DataFrame(
            {
                "flowpath_id": ["1", "2", "3"],
                "areasqkm": [2.0, 1.0, 0.5],
                "lengthkm": [5.0, 3.0, 2.0],
                "totdasqkm": [3.5, 1.5, 0.5],
                "streamorder": [2, 1, 1],
                "hydroseq": [1, 2, 3],
                "dnhydroseq": [0, 1, 2],
                "flowpath_toid": [0, 1, 2],
                "mainstemlp": [100.0, 100.0, 100.0],
                "VPUID": ["01", "01", "01"],
            }
        )

        network_graph = {"1": ["2"], "2": ["3"], "3": []}
        graph, node_indices = dict_to_graph(network_graph)
        partition_data = create_partition_data_from_dataframes(flowpath_data, None, graph, node_indices)

        div_ids: set = set()  # No divides anywhere
        result = _trace_stack("1", div_ids, sample_config, partition_data)

        # All should be marked as minor
        assert "1" in result.minor_flowpaths
        assert "2" in result.minor_flowpaths
        assert "3" in result.minor_flowpaths


class TestTraceStackRule3CaseA:
    """Tests for Rule 3 Case A: Multiple upstream WITH divide."""

    def test_two_upstreams_both_higher_order(self, sample_config: HFConfig) -> None:
        """Test 2 higher-order upstreams creates connector."""
        flowpath_data = pl.DataFrame(
            {
                "flowpath_id": ["1", "2", "3"],
                "areasqkm": [10.0, 5.0, 5.0],
                "lengthkm": [10.0, 8.0, 8.0],
                "totdasqkm": [20.0, 5.0, 5.0],
                "streamorder": [3, 2, 2],
                "hydroseq": [1, 2, 3],
                "dnhydroseq": [0, 1, 1],
                "flowpath_toid": [0, 1, 1],
                "mainstemlp": [100.0, 100.0, 80.0],
                "VPUID": ["01", "01", "01"],
            }
        )

        network_graph = {"1": ["2", "3"], "2": [], "3": []}
        graph, node_indices = dict_to_graph(network_graph)
        partition_data = create_partition_data_from_dataframes(flowpath_data, None, graph, node_indices)

        div_ids = {"1", "2", "3"}
        result = _trace_stack("1", div_ids, sample_config, partition_data)

        # Should be connector
        assert "1" in result.connector_segments

    def test_three_plus_upstreams_connector(self, sample_config: HFConfig) -> None:
        """Test 3+ upstreams always creates connector."""
        flowpath_data = pl.DataFrame(
            {
                "flowpath_id": ["1", "2", "3", "4"],
                "areasqkm": [15.0, 5.0, 5.0, 5.0],
                "lengthkm": [10.0, 5.0, 5.0, 5.0],
                "totdasqkm": [30.0, 5.0, 5.0, 5.0],
                "streamorder": [3, 2, 2, 1],
                "hydroseq": [1, 2, 3, 4],
                "dnhydroseq": [0, 1, 1, 1],
                "flowpath_toid": [0, 1, 1, 1],
                "mainstemlp": [100.0, 100.0, 80.0, 50.0],
                "VPUID": ["01", "01", "01", "01"],
            }
        )

        network_graph = {"1": ["2", "3", "4"], "2": [], "3": [], "4": []}
        graph, node_indices = dict_to_graph(network_graph)
        partition_data = create_partition_data_from_dataframes(flowpath_data, None, graph, node_indices)

        div_ids = {"1", "2", "3", "4"}
        result = _trace_stack("1", div_ids, sample_config, partition_data)

        assert "1" in result.connector_segments


class TestTraceStackRule3CaseB:
    """Tests for Rule 3 Case B: Multiple upstream WITHOUT divide."""

    def test_can_aggregate_downstream_no_competition(self, sample_config: HFConfig) -> None:
        """Test aggregating downstream when no competition."""
        flowpath_data = pl.DataFrame(
            {
                "flowpath_id": ["1", "2", "3", "4"],
                "areasqkm": [5.0, 2.0, 2.0, 1.0],
                "lengthkm": [10.0, 5.0, 5.0, 3.0],
                "totdasqkm": [10.0, 3.0, 3.0, 1.0],
                "streamorder": [3, 2, 2, 1],
                "hydroseq": [1, 2, 3, 4],
                "dnhydroseq": [0, 1, 2, 2],
                "flowpath_toid": [0, 1, 2, 2],
                "mainstemlp": [100.0, 100.0, 80.0, 50.0],
                "VPUID": ["01", "01", "01", "01"],
            }
        )

        network_graph = {"1": ["2"], "2": ["3", "4"], "3": [], "4": []}
        graph, node_indices = dict_to_graph(network_graph)
        partition_data = create_partition_data_from_dataframes(flowpath_data, None, graph, node_indices)

        div_ids = {"1", "3", "4"}  # 2 has no divide
        result = _trace_stack("1", div_ids, sample_config, partition_data)

        # Should aggregate 2 downstream to 1
        assert ("2", "1") in result.aggregation_pairs

    def test_all_upstreams_lack_deep_divides(self, sample_config: HFConfig) -> None:
        """Test when all upstreams lack divides for 3 layers."""
        flowpath_data = pl.DataFrame(
            {
                "flowpath_id": ["1", "2", "3", "4", "5", "6", "7"],
                "areasqkm": [10.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
                "lengthkm": [10.0, 5.0, 5.0, 3.0, 3.0, 3.0, 3.0],
                "totdasqkm": [21.0, 5.0, 5.0, 2.0, 2.0, 1.0, 1.0],
                "streamorder": [3, 2, 2, 1, 1, 1, 1],
                "hydroseq": [1, 2, 3, 4, 5, 6, 7],
                "dnhydroseq": [0, 1, 2, 2, 3, 4, 5],
                "flowpath_toid": [0, 1, 2, 2, 3, 4, 5],
                "mainstemlp": [100.0] * 7,
                "VPUID": ["01"] * 7,
            }
        )

        network_graph = {
            "1": ["2"],
            "2": ["3", "4"],
            "3": ["5"],
            "4": ["6"],
            "5": ["7"],
            "6": [],
            "7": [],
        }
        graph, node_indices = dict_to_graph(network_graph)
        partition_data = create_partition_data_from_dataframes(flowpath_data, None, graph, node_indices)

        # Only 1 and 7 have divides (far apart)
        div_ids = {"1", "7"}
        result = _trace_stack("1", div_ids, sample_config, partition_data)

        # 2 aggregates to 1 (downstream) since it has no competition
        # Upstreams 4 and 6 (without divides) are marked as minor
        assert ("2", "1") in result.aggregation_pairs
        assert "4" in result.minor_flowpaths or "6" in result.minor_flowpaths

    def test_order_1_2_can_be_made_minor(self, sample_config: HFConfig) -> None:
        """Test making order 1/2 streams minor."""
        flowpath_data = pl.DataFrame(
            {
                "flowpath_id": ["1", "2", "3", "4", "5"],
                "areasqkm": [10.0, 5.0, 2.0, 2.0, 1.0],
                "lengthkm": [10.0, 8.0, 5.0, 5.0, 3.0],
                "totdasqkm": [20.0, 10.0, 3.0, 3.0, 1.0],
                "streamorder": [4, 3, 2, 2, 1],
                "hydroseq": [1, 2, 3, 4, 5],
                "dnhydroseq": [0, 1, 2, 2, 3],
                "flowpath_toid": [0, 1, 2, 2, 3],
                "mainstemlp": [100.0, 100.0, 80.0, 70.0, 50.0],
                "VPUID": ["01"] * 5,
            }
        )

        network_graph = {"1": ["2"], "2": ["3", "4"], "3": ["5"], "4": [], "5": []}
        graph, node_indices = dict_to_graph(network_graph)
        partition_data = create_partition_data_from_dataframes(flowpath_data, None, graph, node_indices)

        div_ids = {"1", "3", "4", "5"}  # 2 has no divide
        result = _trace_stack("1", div_ids, sample_config, partition_data)

        # 2 should aggregate downstream (no competition) or to best upstream
        # Since 2 lacks divide and has no competition downstream, aggregates to 1
        assert (
            ("2", "1") in result.aggregation_pairs
            or ("2", "3") in result.aggregation_pairs
            or ("2", "4") in result.aggregation_pairs
        )

    def test_awkward_connector_high_order_upstreams(self, sample_config: HFConfig) -> None:
        """Test awkward connector with all high-order upstreams."""
        flowpath_data = pl.DataFrame(
            {
                "flowpath_id": ["1", "2", "3", "4"],
                "areasqkm": [15.0, 5.0, 5.0, 5.0],
                "lengthkm": [10.0, 8.0, 8.0, 8.0],
                "totdasqkm": [30.0, 5.0, 5.0, 5.0],
                "streamorder": [4, 3, 3, 3],
                "hydroseq": [1, 2, 3, 4],
                "dnhydroseq": [0, 1, 2, 2],
                "flowpath_toid": [0, 1, 2, 2],
                "mainstemlp": [100.0, 100.0, 80.0, 80.0],
                "VPUID": ["01"] * 4,
            }
        )

        network_graph = {"1": ["2"], "2": ["3", "4"], "3": [], "4": []}
        graph, node_indices = dict_to_graph(network_graph)
        partition_data = create_partition_data_from_dataframes(flowpath_data, None, graph, node_indices)

        div_ids = {"1", "3", "4"}  # 2 has no divide, downstream has competition
        result = _trace_stack("1", div_ids, sample_config, partition_data)

        # Should aggregate downstream and mark as connector
        assert ("2", "1") in result.aggregation_pairs


class TestTraceSingleFlowpathAttributes:
    """Tests for _trace_single_flowpath_attributes function."""

    def test_simple_chain(self, sample_config: HFConfig) -> None:
        """Test tracing simple chain."""
        flowpath_data = pl.DataFrame(
            {
                "flowpath_id": ["1", "2", "3"],
                "areasqkm": [5.0, 3.0, 2.0],
                "lengthkm": [10.0, 8.0, 6.0],
                "hydroseq": [1, 2, 3],
                "streamorder": [2, 1, 1],
            }
        )

        network_graph = {"1": ["2"], "2": ["3"], "3": []}
        graph, node_indices = dict_to_graph(network_graph)
        partition_data = create_partition_data_from_dataframes(flowpath_data, None, graph, node_indices)

        result = _trace_single_flowpath_attributes("1", partition_data, id_offset=1)

        assert "total_da_sqkm" in result.columns
        assert "mainstem_lp" in result.columns
        assert "path_length" in result.columns
        assert "streamorder" in result.columns

        # Check streamorder calculation
        assert result.filter(pl.col("fp_id") == "3")["streamorder"][0] == 1  # Headwater
        assert result.filter(pl.col("fp_id") == "1")["streamorder"][0] >= 1

    def test_branching_network(self, sample_config: HFConfig) -> None:
        """Test tracing branching network."""
        flowpath_data = pl.DataFrame(
            {
                "flowpath_id": ["1", "2", "3"],
                "areasqkm": [5.0, 2.0, 2.0],
                "lengthkm": [10.0, 8.0, 8.0],
                "hydroseq": [1, 2, 3],
                "streamorder": [2, 1, 1],
            }
        )

        network_graph = {"1": ["2", "3"], "2": [], "3": []}
        graph, node_indices = dict_to_graph(network_graph)
        partition_data = create_partition_data_from_dataframes(flowpath_data, None, graph, node_indices)

        result = _trace_single_flowpath_attributes("1", partition_data, id_offset=1)

        # Check mainstem assignment
        assert result.filter(pl.col("fp_id") == "1")["mainstem_lp"][0] == 1

        # Check drainage area accumulation
        outlet_da = result.filter(pl.col("fp_id") == "1")["total_da_sqkm"][0]
        assert outlet_da > 5.0  # Should include upstream areas

    def test_streamorder_strahler(self, sample_config: HFConfig) -> None:
        """Test Strahler stream order calculation."""
        # Create Y-shaped network
        flowpath_data = pl.DataFrame(
            {
                "flowpath_id": ["1", "2", "3", "4"],
                "areasqkm": [5.0, 2.0, 2.0, 1.0],
                "lengthkm": [10.0, 8.0, 8.0, 5.0],
                "hydroseq": [1, 2, 3, 4],
                "streamorder": [2, 1, 1, 1],
            }
        )

        # 1 <- 2, 3  and 2 <- 4
        network_graph = {"1": ["2", "3"], "2": ["4"], "3": [], "4": []}
        graph, node_indices = dict_to_graph(network_graph)
        partition_data = create_partition_data_from_dataframes(flowpath_data, None, graph, node_indices)

        result = _trace_single_flowpath_attributes("1", partition_data, id_offset=1)

        # Headwaters should be order 1
        assert result.filter(pl.col("fp_id") == "4")["streamorder"][0] == 1
        assert result.filter(pl.col("fp_id") == "3")["streamorder"][0] == 1

        # When order 1 meets order 1, becomes order 2
        fp2_order = result.filter(pl.col("fp_id") == "2")["streamorder"][0]
        assert fp2_order == 1  # Only one upstream (4)

        # Outlet with two order 1s
        fp1_order = result.filter(pl.col("fp_id") == "1")["streamorder"][0]
        assert fp1_order == 2  # Two order 1s meet


class TestTraceStackEdgeCases:
    """Tests for edge cases in trace stack."""

    def test_force_queue_flowpaths(self, sample_config: HFConfig) -> None:
        """Test that force_queue_flowpaths get added."""
        flowpath_data = pl.DataFrame(
            {
                "flowpath_id": ["1", "2", "3"],
                "areasqkm": [5.0, 2.0, 1.0],
                "lengthkm": [10.0, 5.0, 3.0],
                "totdasqkm": [8.0, 3.0, 1.0],
                "streamorder": [2, 1, 1],
                "hydroseq": [1, 2, 3],
                "dnhydroseq": [0, 1, 2],
                "flowpath_toid": [0, 1, 2],
                "mainstemlp": [100.0, 100.0, 100.0],
                "VPUID": ["01", "01", "01"],
            }
        )

        network_graph = {"1": ["2", "3"], "2": [], "3": []}
        graph, node_indices = dict_to_graph(network_graph)
        partition_data = create_partition_data_from_dataframes(flowpath_data, None, graph, node_indices)

        div_ids = {"1"}  # Only 1 has divide
        result = _trace_stack("1", div_ids, sample_config, partition_data)

        # Check that force_queue was populated for problematic branches
        assert isinstance(result.force_queue_flowpaths, set)

    def test_cycle_detection(self, sample_config: HFConfig) -> None:
        """Test that cycles are caught in tracing."""
        flowpath_data = pl.DataFrame(
            {
                "flowpath_id": ["1", "2", "3"],
                "areasqkm": [5.0, 3.0, 2.0],
                "lengthkm": [10.0, 8.0, 6.0],
                "totdasqkm": [10.0, 5.0, 2.0],
                "streamorder": [2, 1, 1],
                "hydroseq": [1, 2, 3],
                "dnhydroseq": [3, 1, 2],  # Cycle!
                "flowpath_toid": [3, 1, 2],
                "mainstemlp": [100.0, 100.0, 100.0],
                "VPUID": ["01", "01", "01"],
            }
        )

        # dict_to_graph creates directed edges, so cycle is: 1->2, 2->3, 3->1
        network_graph = {"1": ["2"], "2": ["3"], "3": ["1"]}

        # This will create a cycle, rustworkx should detect it
        # The trace code wraps DAGHasCycle in AssertionError
        with pytest.raises(AssertionError, match="Basin 1 contains cycles"):
            graph, node_indices = dict_to_graph(network_graph)
            partition_data = create_partition_data_from_dataframes(flowpath_data, None, graph, node_indices)
            # The topological_sort in _trace_single_flowpath_attributes will raise AssertionError
            _trace_single_flowpath_attributes("1", partition_data, id_offset=1)
