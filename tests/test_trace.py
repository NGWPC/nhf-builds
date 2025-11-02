"""Tests for trace.py functions."""

from collections import deque

import polars as pl
import pytest
from conftest import dict_to_graph

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.hydrofabric.trace import (
    _rule_aggregate_mixed_upstream_orders,
    _rule_aggregate_order2_with_order1s,
    _rule_aggregate_single_upstream,
    _rule_independent_connector,
)
from hydrofabric_builds.schemas.hydrofabric import Classifications


@pytest.fixture
def sample_flowpath_data() -> pl.DataFrame:
    """Create sample flowpath data for unit testing individual rules."""
    data = {
        "flowpath_id": ["1", "2", "3", "4", "5", "6", "7", "8"],
        "totdasqkm": [100.0, 50.0, 25.0, 10.0, 5.0, 2.0, 1.0, 0.5],
        "areasqkm": [5.0, 2.5, 1.5, 0.8, 2.0, 1.0, 0.5, 0.3],
        "lengthkm": [10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0],
        "streamorder": [3, 2, 2, 1, 1, 1, 1, 1],
        "hydroseq": [1, 2, 3, 4, 5, 6, 7, 8],
        "dnhydroseq": [0, 1, 1, 2, 3, 4, 4, 5],
        "mainstemlp": [300.0, 300.0, 100.0, 50.0, 100.0, 100.0, 300.0, 300.0],
    }
    return pl.DataFrame(data)


@pytest.fixture
def sample_fp_lookup(sample_flowpath_data: pl.DataFrame) -> dict:
    """Create fp_lookup dictionary from sample flowpath data."""
    return {str(row["flowpath_id"]): row for row in sample_flowpath_data.to_dicts()}


def test_connector_aggregates_small_order1_upstreams(
    sample_flowpath_data: pl.DataFrame, sample_config: HFConfig
) -> None:
    """Test that connector aggregates small order 1 upstreams."""
    network_graph = {
        "fp1": ["up1", "up2", "up3"],
        "up1": [],
        "up2": [],
        "up3": [],
    }
    graph, node_indices = dict_to_graph(network_graph)

    div_ids = {"fp1", "up1", "up2", "up3"}

    upstream_info = [
        {"flowpath_id": "up1", "areasqkm": 5.0, "streamorder": 2, "length_km": 5.0, "mainstemlp": 101.0},
        {"flowpath_id": "up2", "areasqkm": 5.0, "streamorder": 2, "length_km": 5.0, "mainstemlp": 100.0},
        {"flowpath_id": "up3", "areasqkm": 0.3, "streamorder": 1, "length_km": 0.8, "mainstemlp": 8.0},
    ]

    result = Classifications()

    # Call function
    is_connector = _rule_independent_connector(
        current_id="fp1",
        upstream_info=upstream_info,
        cfg=sample_config,
        result=result,
        div_ids=div_ids,
        graph=graph,
        node_indices=node_indices,
    )

    assert is_connector
    assert "fp1" in result.connector_segments
    # Small order 1 upstreams should be aggregated
    assert ("up3", "fp1") in result.aggregation_pairs or "up3" in result.minor_flowpaths


class TestRuleAggregateSingleUpstream:
    """Tests for _rule_aggregate_single_upstream function."""

    def test_single_upstream_aggregation(self, sample_fp_lookup: dict, sample_config: HFConfig) -> None:
        """Test basic single upstream aggregation."""
        network_graph = {"5": ["6"], "6": []}
        div_ids = {"5", "6"}
        to_process: deque = deque()
        result = Classifications()
        graph, node_indices = dict_to_graph(network_graph)

        fp_info = {
            "flowpath_id": "5",
            "areasqkm": 2.0,
            "streamorder": 1,
            "length_km": 3.0,
            "mainstemlp": 100.0,
        }

        upstream_info = [
            {"flowpath_id": "6", "areasqkm": 1.0, "streamorder": 1, "length_km": 2.0, "mainstemlp": 100.0}
        ]

        success = _rule_aggregate_single_upstream(
            current_id="5",
            fp_info=fp_info,
            upstream_info=upstream_info,
            cfg=sample_config,
            result=result,
            div_ids=div_ids,
            fp_lookup=sample_fp_lookup,
            to_process=to_process,
            graph=graph,
            node_indices=node_indices,
        )

        assert success
        assert ("6", "5") in result.aggregation_pairs

    def test_cumulative_area_tracking(self, sample_fp_lookup: dict, sample_config: HFConfig) -> None:
        """Test cumulative area tracking stops at threshold."""
        network_graph = {"fp1": ["up1"], "up1": ["up2"], "up2": []}
        graph, node_indices = dict_to_graph(network_graph)
        div_ids = {"fp1", "up1", "up2"}
        to_process: deque = deque()
        result = Classifications()

        fp_info = {
            "flowpath_id": "fp1",
            "areasqkm": 8.0,
            "streamorder": 2,
            "length_km": 10.0,
            "mainstemlp": 100.0,
        }

        upstream_info = [
            {"flowpath_id": "up1", "areasqkm": 3.0, "streamorder": 2, "length_km": 5.0, "mainstemlp": 50.0}
        ]

        success = _rule_aggregate_single_upstream(
            current_id="fp1",
            fp_info=fp_info,
            upstream_info=upstream_info,
            cfg=sample_config,
            result=result,
            div_ids=div_ids,
            graph=graph,
            node_indices=node_indices,
            fp_lookup=sample_fp_lookup,
            to_process=to_process,
        )

        assert success

    def test_large_area_not_aggregated(self, sample_fp_lookup: dict, sample_config: HFConfig) -> None:
        """Test large area upstream not aggregated."""
        network_graph = {"fp1": ["up1"], "up1": []}
        graph, node_indices = dict_to_graph(network_graph)
        div_ids = {"fp1", "up1"}
        to_process: deque = deque()
        result = Classifications()

        fp_info = {
            "flowpath_id": "fp1",
            "areasqkm": 5.0,
            "streamorder": 2,
            "length_km": 10.0,
            "mainstemlp": 100.0,
        }

        upstream_info = [
            {"flowpath_id": "up1", "areasqkm": 15.0, "streamorder": 2, "length_km": 8.0, "mainstemlp": 80.0}
        ]

        success = _rule_aggregate_single_upstream(
            current_id="fp1",
            fp_info=fp_info,
            upstream_info=upstream_info,
            cfg=sample_config,
            result=result,
            div_ids=div_ids,
            graph=graph,
            node_indices=node_indices,
            fp_lookup=sample_fp_lookup,
            to_process=to_process,
        )

        assert success


class TestRuleAggregateOrder2WithOrder1s:
    """Tests for _rule_aggregate_order2_with_order1s function."""

    def test_order2_with_two_order1s(self, sample_fp_lookup: dict, sample_config: HFConfig) -> None:
        """Test order 2 with two order 1 upstreams."""
        network_graph = {"fp1": ["up1", "up2"], "up1": [], "up2": []}
        graph, node_indices = dict_to_graph(network_graph)
        div_ids = {"fp1", "up1", "up2"}
        to_process: deque = deque()
        result = Classifications()

        fp_info = {
            "flowpath_id": "fp1",
            "areasqkm": 5.0,
            "streamorder": 2,
            "length_km": 10.0,
            "mainstemlp": 100.0,
        }

        upstream_info = [
            {"flowpath_id": "up1", "areasqkm": 2.0, "streamorder": 1, "length_km": 5.0, "mainstemlp": 50.0},
            {"flowpath_id": "up2", "areasqkm": 2.5, "streamorder": 1, "length_km": 6.0, "mainstemlp": 60.0},
        ]

        success = _rule_aggregate_order2_with_order1s(
            current_id="fp1",
            fp_info=fp_info,
            upstream_info=upstream_info,
            graph=graph,
            node_indices=node_indices,
            result=result,
            div_ids=div_ids,
            fp_lookup=sample_fp_lookup,
            to_process=to_process,
        )

        assert success

    def test_order2_with_upstream_branches(self, sample_fp_lookup: dict, sample_config: HFConfig) -> None:
        """Test order 2 with branching order 1 upstreams."""
        network_graph = {
            "fp1": ["up1", "up2"],
            "up1": ["up1a", "up1b"],
            "up2": [],
            "up1a": [],
            "up1b": [],
        }
        graph, node_indices = dict_to_graph(network_graph)
        div_ids = {"fp1", "up1", "up2", "up1a", "up1b"}
        to_process: deque = deque()
        result = Classifications()

        fp_info = {
            "flowpath_id": "fp1",
            "areasqkm": 8.0,
            "streamorder": 2,
            "length_km": 10.0,
            "mainstemlp": 100.0,
        }

        upstream_info = [
            {"flowpath_id": "up1", "areasqkm": 3.0, "streamorder": 1, "length_km": 5.0, "mainstemlp": 50.0},
            {"flowpath_id": "up2", "areasqkm": 2.0, "streamorder": 1, "length_km": 4.0, "mainstemlp": 40.0},
        ]

        success = _rule_aggregate_order2_with_order1s(
            current_id="fp1",
            fp_info=fp_info,
            upstream_info=upstream_info,
            graph=graph,
            node_indices=node_indices,
            result=result,
            div_ids=div_ids,
            fp_lookup=sample_fp_lookup,
            to_process=to_process,
        )

        assert success

    def test_not_order2(self, sample_fp_lookup: dict, sample_config: HFConfig) -> None:
        """Test function returns False when not order 2."""
        network_graph = {"fp1": ["up1", "up2"], "up1": [], "up2": []}
        graph, node_indices = dict_to_graph(network_graph)
        div_ids = {"fp1", "up1", "up2"}
        to_process: deque = deque()
        result = Classifications()

        fp_info = {
            "flowpath_id": "fp1",
            "areasqkm": 10.0,
            "streamorder": 3,  # Not order 2
            "length_km": 10.0,
            "mainstemlp": 100.0,
        }

        upstream_info = [
            {"flowpath_id": "up1", "areasqkm": 2.0, "streamorder": 1, "length_km": 5.0, "mainstemlp": 50.0},
            {"flowpath_id": "up2", "areasqkm": 2.5, "streamorder": 1, "length_km": 6.0, "mainstemlp": 60.0},
        ]

        success = _rule_aggregate_order2_with_order1s(
            current_id="fp1",
            fp_info=fp_info,
            upstream_info=upstream_info,
            graph=graph,
            node_indices=node_indices,
            result=result,
            div_ids=div_ids,
            fp_lookup=sample_fp_lookup,
            to_process=to_process,
        )

        assert not success

    def test_not_all_order1(self, sample_fp_lookup: dict, sample_config: HFConfig) -> None:
        """Test function returns False when not all upstreams are order 1."""
        network_graph = {"fp1": ["up1", "up2"], "up1": [], "up2": []}
        graph, node_indices = dict_to_graph(network_graph)
        div_ids = {"fp1", "up1", "up2"}
        to_process: deque = deque()
        result = Classifications()

        fp_info = {
            "flowpath_id": "fp1",
            "areasqkm": 10.0,
            "streamorder": 2,
            "length_km": 10.0,
            "mainstemlp": 100.0,
        }

        upstream_info = [
            {"flowpath_id": "up1", "areasqkm": 2.0, "streamorder": 1, "length_km": 5.0, "mainstemlp": 50.0},
            {
                "flowpath_id": "up2",
                "areasqkm": 5.0,
                "streamorder": 2,
                "length_km": 8.0,
                "mainstemlp": 80.0,
            },  # Order 2
        ]

        success = _rule_aggregate_order2_with_order1s(
            current_id="fp1",
            fp_info=fp_info,
            upstream_info=upstream_info,
            graph=graph,
            node_indices=node_indices,
            result=result,
            div_ids=div_ids,
            fp_lookup=sample_fp_lookup,
            to_process=to_process,
        )

        assert not success

    def test_order2_with_three_order1s(self, sample_fp_lookup: dict, sample_config: HFConfig) -> None:
        """Test order 2 with three order 1 upstreams."""
        network_graph = {"fp1": ["up1", "up2", "up3"], "up1": [], "up2": [], "up3": []}
        graph, node_indices = dict_to_graph(network_graph)
        div_ids = {"fp1", "up1", "up2", "up3"}
        to_process: deque = deque()
        result = Classifications()

        fp_info = {
            "flowpath_id": "fp1",
            "areasqkm": 8.0,
            "streamorder": 2,
            "length_km": 10.0,
            "mainstemlp": 100.0,
        }

        upstream_info = [
            {"flowpath_id": "up1", "areasqkm": 2.0, "streamorder": 1, "length_km": 5.0, "mainstemlp": 50.0},
            {"flowpath_id": "up2", "areasqkm": 2.5, "streamorder": 1, "length_km": 6.0, "mainstemlp": 60.0},
            {"flowpath_id": "up3", "areasqkm": 1.5, "streamorder": 1, "length_km": 4.0, "mainstemlp": 40.0},
        ]

        success = _rule_aggregate_order2_with_order1s(
            current_id="fp1",
            fp_info=fp_info,
            upstream_info=upstream_info,
            graph=graph,
            node_indices=node_indices,
            result=result,
            div_ids=div_ids,
            fp_lookup=sample_fp_lookup,
            to_process=to_process,
        )

        assert success

    def test_order2_with_four_order1s(self, sample_fp_lookup: dict, sample_config: HFConfig) -> None:
        """Test order 2 with four order 1 upstreams."""
        network_graph = {
            "fp1": ["up1", "up2", "up3", "up4"],
            "up1": [],
            "up2": [],
            "up3": [],
            "up4": [],
        }
        graph, node_indices = dict_to_graph(network_graph)
        div_ids = {"fp1", "up1", "up2", "up3", "up4"}
        to_process: deque = deque()
        result = Classifications()

        fp_info = {
            "flowpath_id": "fp1",
            "areasqkm": 10.0,
            "streamorder": 2,
            "length_km": 10.0,
            "mainstemlp": 100.0,
        }

        upstream_info = [
            {"flowpath_id": "up1", "areasqkm": 2.0, "streamorder": 1, "length_km": 5.0, "mainstemlp": 50.0},
            {"flowpath_id": "up2", "areasqkm": 2.5, "streamorder": 1, "length_km": 6.0, "mainstemlp": 60.0},
            {"flowpath_id": "up3", "areasqkm": 1.5, "streamorder": 1, "length_km": 4.0, "mainstemlp": 40.0},
            {"flowpath_id": "up4", "areasqkm": 1.0, "streamorder": 1, "length_km": 3.0, "mainstemlp": 30.0},
        ]

        success = _rule_aggregate_order2_with_order1s(
            current_id="fp1",
            fp_info=fp_info,
            upstream_info=upstream_info,
            graph=graph,
            node_indices=node_indices,
            result=result,
            div_ids=div_ids,
            fp_lookup=sample_fp_lookup,
            to_process=to_process,
        )

        assert success


class TestRuleAggregateMixedUpstreamOrders:
    """Tests for _rule_aggregate_mixed_upstream_orders function."""

    def test_mixed_orders_small_order1_becomes_minor(
        self, sample_fp_lookup: dict, sample_config: HFConfig
    ) -> None:
        """Test mixed orders with small order 1 becoming minor."""
        network_graph = {"fp1": ["up1", "up2"], "up1": [], "up2": []}
        graph, node_indices = dict_to_graph(network_graph)
        div_ids = {"fp1", "up1", "up2"}
        to_process: deque = deque()
        result = Classifications()

        fp_info = {
            "flowpath_id": "fp1",
            "areasqkm": 15.0,
            "streamorder": 3,  # Order 3
            "length_km": 10.0,
            "mainstemlp": 100.0,
        }

        upstream_info = [
            {
                "flowpath_id": "up1",
                "areasqkm": 0.5,
                "streamorder": 1,
                "length_km": 1.0,
                "mainstemlp": 10.0,
            },  # Order 1
            {
                "flowpath_id": "up2",
                "areasqkm": 8.0,
                "streamorder": 3,
                "length_km": 8.0,
                "mainstemlp": 100.0,
            },  # Same order (3)
        ]

        success = _rule_aggregate_mixed_upstream_orders(
            current_id="fp1",
            fp_info=fp_info,
            upstream_info=upstream_info,
            cfg=sample_config,
            result=result,
            div_ids=div_ids,
            graph=graph,
            node_indices=node_indices,
            fp_lookup=sample_fp_lookup,
            to_process=to_process,
        )

        assert success
        assert "up1" in result.minor_flowpaths  # Order 1 should be minor
        assert ("up1", "fp1") in result.aggregation_pairs

    def test_large_order1_not_minor(self, sample_fp_lookup: dict, sample_config: HFConfig) -> None:
        """Test that function still works with large order 1 (area doesn't matter for this rule)."""
        network_graph = {"fp1": ["up1", "up2"], "up1": [], "up2": []}
        graph, node_indices = dict_to_graph(network_graph)
        div_ids = {"fp1", "up1", "up2"}
        to_process: deque = deque()
        result = Classifications()

        fp_info = {
            "flowpath_id": "fp1",
            "areasqkm": 20.0,
            "streamorder": 3,
            "length_km": 10.0,
            "mainstemlp": 100.0,
        }

        upstream_info = [
            {
                "flowpath_id": "up1",
                "areasqkm": 12.0,
                "streamorder": 1,
                "length_km": 8.0,
                "mainstemlp": 80.0,
            },  # Large order 1
            {
                "flowpath_id": "up2",
                "areasqkm": 8.0,
                "streamorder": 3,
                "length_km": 7.0,
                "mainstemlp": 100.0,
            },  # Same order
        ]

        success = _rule_aggregate_mixed_upstream_orders(
            current_id="fp1",
            fp_info=fp_info,
            upstream_info=upstream_info,
            cfg=sample_config,
            result=result,
            div_ids=div_ids,
            graph=graph,
            node_indices=node_indices,
            fp_lookup=sample_fp_lookup,
            to_process=to_process,
        )

        assert success
        # Order 1 is still marked as minor regardless of size
        assert "up1" in result.minor_flowpaths

    def test_multiple_small_order1s_all_minor(self, sample_fp_lookup: dict, sample_config: HFConfig) -> None:
        """Test multiple order 1s all marked as minor with same-order upstream."""
        network_graph = {"fp1": ["up1", "up2", "up3"], "up1": [], "up2": [], "up3": []}
        graph, node_indices = dict_to_graph(network_graph)
        div_ids = {"fp1", "up1", "up2", "up3"}
        to_process: deque = deque()
        result = Classifications()

        fp_info = {
            "flowpath_id": "fp1",
            "areasqkm": 25.0,
            "streamorder": 3,
            "length_km": 10.0,
            "mainstemlp": 100.0,
        }

        upstream_info = [
            {
                "flowpath_id": "up1",
                "areasqkm": 0.5,
                "streamorder": 1,
                "length_km": 1.0,
                "mainstemlp": 10.0,
            },  # Order 1
            {
                "flowpath_id": "up2",
                "areasqkm": 0.8,
                "streamorder": 1,
                "length_km": 1.5,
                "mainstemlp": 15.0,
            },  # Order 1
            {
                "flowpath_id": "up3",
                "areasqkm": 15.0,
                "streamorder": 3,
                "length_km": 9.0,
                "mainstemlp": 100.0,
            },  # Same order
        ]

        success = _rule_aggregate_mixed_upstream_orders(
            current_id="fp1",
            fp_info=fp_info,
            upstream_info=upstream_info,
            cfg=sample_config,
            result=result,
            div_ids=div_ids,
            graph=graph,
            node_indices=node_indices,
            fp_lookup=sample_fp_lookup,
            to_process=to_process,
        )

        assert success
        assert "up1" in result.minor_flowpaths
        assert "up2" in result.minor_flowpaths
        assert "fp1" in result.upstream_merge_points

    def test_only_order1s_not_mixed(self, sample_fp_lookup: dict, sample_config: HFConfig) -> None:
        """Test function returns False when all upstreams are order 1 (no same-order upstream)."""
        network_graph = {"fp1": ["up1", "up2"], "up1": [], "up2": []}
        graph, node_indices = dict_to_graph(network_graph)
        div_ids = {"fp1", "up1", "up2"}
        to_process: deque = deque()
        result = Classifications()

        fp_info = {
            "flowpath_id": "fp1",
            "areasqkm": 10.0,
            "streamorder": 2,
            "length_km": 10.0,
            "mainstemlp": 100.0,
        }

        upstream_info = [
            {"flowpath_id": "up1", "areasqkm": 2.0, "streamorder": 1, "length_km": 5.0, "mainstemlp": 50.0},
            {"flowpath_id": "up2", "areasqkm": 2.5, "streamorder": 1, "length_km": 6.0, "mainstemlp": 60.0},
        ]

        success = _rule_aggregate_mixed_upstream_orders(
            current_id="fp1",
            fp_info=fp_info,
            upstream_info=upstream_info,
            cfg=sample_config,
            result=result,
            div_ids=div_ids,
            graph=graph,
            node_indices=node_indices,
            fp_lookup=sample_fp_lookup,
            to_process=to_process,
        )

        # Should return False - no same-order upstreams
        assert not success

    def test_only_same_order_not_mixed(self, sample_fp_lookup: dict, sample_config: HFConfig) -> None:
        """Test function returns False when no order 1 upstreams (only same-order)."""
        network_graph = {"fp1": ["up1", "up2"], "up1": [], "up2": []}
        graph, node_indices = dict_to_graph(network_graph)
        div_ids = {"fp1", "up1", "up2"}
        to_process: deque = deque()
        result = Classifications()

        fp_info = {
            "flowpath_id": "fp1",
            "areasqkm": 30.0,
            "streamorder": 3,
            "length_km": 10.0,
            "mainstemlp": 100.0,
        }

        upstream_info = [
            {
                "flowpath_id": "up1",
                "areasqkm": 15.0,
                "streamorder": 3,
                "length_km": 8.0,
                "mainstemlp": 80.0,
            },  # Same order
            {
                "flowpath_id": "up2",
                "areasqkm": 12.0,
                "streamorder": 3,
                "length_km": 7.0,
                "mainstemlp": 70.0,
            },  # Same order
        ]

        success = _rule_aggregate_mixed_upstream_orders(
            current_id="fp1",
            fp_info=fp_info,
            upstream_info=upstream_info,
            cfg=sample_config,
            result=result,
            div_ids=div_ids,
            graph=graph,
            node_indices=node_indices,
            fp_lookup=sample_fp_lookup,
            to_process=to_process,
        )

        # Should return False - no order 1 upstreams
        assert not success

    def test_small_order1_with_same_order(self, sample_fp_lookup: dict, sample_config: HFConfig) -> None:
        """Test order 1 tributary with same-order mainstem."""
        network_graph = {"fp1": ["up1", "up2"], "up1": [], "up2": []}
        div_ids = {"fp1", "up1", "up2"}
        to_process: deque = deque()
        result = Classifications()
        graph, node_indices = dict_to_graph(network_graph)
        fp_info = {
            "flowpath_id": "fp1",
            "areasqkm": 1.0,
            "streamorder": 3,
            "length_km": 10.0,
            "mainstemlp": 100.0,
        }

        upstream_info = [
            {
                "flowpath_id": "up1",
                "areasqkm": 0.5,
                "streamorder": 1,
                "length_km": 1.0,
                "mainstemlp": 10.0,
            },  # Small order 1
            {
                "flowpath_id": "up2",
                "areasqkm": 2.0,
                "streamorder": 3,
                "length_km": 9.0,
                "mainstemlp": 100.0,
            },  # Same order
        ]

        success = _rule_aggregate_mixed_upstream_orders(
            current_id="fp1",
            fp_info=fp_info,
            upstream_info=upstream_info,
            cfg=sample_config,
            result=result,
            div_ids=div_ids,
            graph=graph,
            node_indices=node_indices,
            fp_lookup=sample_fp_lookup,
            to_process=to_process,
        )

        assert success
        assert "up1" in result.minor_flowpaths
        assert ("up1", "fp1") in result.aggregation_pairs
