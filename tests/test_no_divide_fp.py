"""Tests for no-divide flowpath cases and corresponding trace rules"""

from collections import deque

import polars as pl
from conftest import dict_to_graph

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.hydrofabric.trace import (
    _check_and_aggregate_same_order_no_divide_chain,
    _rule_aggregate_single_upstream,
    _trace_stack,
)
from hydrofabric_builds.schemas.hydrofabric import Classifications


class TestOrderOneNoDivide:
    """Tests for Rule 2: Order 1 without divide - aggregate entire upstream chain."""

    def test_order1_no_divide_aggregates_upstream_with_divide(self, sample_config: HFConfig) -> None:
        """Test order 1 no-divide aggregates upstream order 1 with divide.

        Pattern: [order-1 WITH divide] → [order-1 NO divide]
        Expected: Both aggregated into one unit with divide
        """
        # Network: 807787 (with divide) → 808457 (no divide)
        network_graph = {"808457": ["807787"], "807787": []}
        graph, node_indices = dict_to_graph(network_graph)

        # Only 807787 has divide
        div_ids = {"807787"}

        # Create flowpath data
        fp_data = pl.DataFrame(
            {
                "flowpath_id": ["808457", "807787"],
                "areasqkm": [0.5, 1.0],
                "streamorder": [1, 1],
                "lengthkm": [0.3, 0.5],
                "hydroseq": [1, 2],
                "dnhydroseq": [0, 1],
                "mainstemlp": [100.0, 100.0],
            }
        )

        result = _trace_stack(
            start_id="808457",
            fp=fp_data,
            div_ids=div_ids,
            cfg=sample_config,
            digraph=graph,
            node_indices=node_indices,
        )

        # 808457 should NOT be marked as connector
        assert "808457" not in result.no_divide_connectors

        # Should create aggregation pair: 807787 aggregates into 808457
        assert ("807787", "808457") in result.aggregation_pairs

        # 808457 should NOT be marked as minor (it's the target of aggregation)
        assert "808457" not in result.minor_flowpaths

        # Both should be processed
        assert "808457" in result.processed_flowpaths
        assert "807787" in result.processed_flowpaths

    def test_order1_no_divide_chain_with_divide_upstream(self, sample_config: HFConfig) -> None:
        """Test order 1 no-divide chain aggregates entire upstream including divides.

        Pattern: [807789 NO div] → [807787 WITH div] → [808457 NO div]
        Expected: All three in one unit with 807787's divide
        """
        network_graph = {"808457": ["807787"], "807787": ["807789"], "807789": []}
        graph, node_indices = dict_to_graph(network_graph)

        # Only 807787 has divide
        div_ids = {"807787"}

        fp_data = pl.DataFrame(
            {
                "flowpath_id": ["808457", "807787", "807789"],
                "areasqkm": [0.5, 1.0, 0.3],
                "streamorder": [1, 1, 1],
                "lengthkm": [0.3, 0.5, 0.2],
                "hydroseq": [1, 2, 3],
                "dnhydroseq": [0, 1, 2],
                "mainstemlp": [100.0, 100.0, 100.0],
            }
        )

        result = _trace_stack(
            start_id="808457",
            fp=fp_data,
            div_ids=div_ids,
            cfg=sample_config,
            digraph=graph,
            node_indices=node_indices,
        )

        # All should be aggregated together
        assert ("807787", "808457") in result.aggregation_pairs
        assert ("807789", "808457") in result.aggregation_pairs

        # None should be marked as connectors
        assert "808457" not in result.no_divide_connectors
        assert "807787" not in result.no_divide_connectors

        # None should be marked as minor
        assert "808457" not in result.minor_flowpaths
        assert "807787" not in result.minor_flowpaths
        assert "807789" not in result.minor_flowpaths


class TestOrderTwoNoDivideChain:
    """Tests for Rule 3: Order 2+ without divide chain checking."""

    def test_order2_no_divide_chain_all_no_divides_marked_minor(self, sample_config: HFConfig) -> None:
        """Test order 2 chain with NO divides anywhere - all marked as minor.

        Pattern: [order-2 NO div] → [order-2 NO div] (no divides anywhere)
        Expected: All aggregated and marked as minor
        """
        network_graph = {"7717438": ["7717428"], "7717428": []}
        graph, node_indices = dict_to_graph(network_graph)

        # NO divides in this chain
        div_ids: set = set()

        fp_data = pl.DataFrame(
            {
                "flowpath_id": ["7717438", "7717428"],
                "areasqkm": [0.5, 0.4],
                "streamorder": [2, 2],
                "lengthkm": [0.3, 0.25],
                "hydroseq": [1, 2],
                "dnhydroseq": [0, 1],
                "mainstemlp": [100.0, 100.0],
            }
        )

        result = _trace_stack(
            start_id="7717438",
            fp=fp_data,
            div_ids=div_ids,
            cfg=sample_config,
            digraph=graph,
            node_indices=node_indices,
        )

        # Both should be marked as minor
        assert "7717438" in result.minor_flowpaths
        assert "7717428" in result.minor_flowpaths

        # Should be aggregated together
        assert ("7717428", "7717438") in result.aggregation_pairs

    def test_order2_no_divide_chain_with_order1_upstream_no_divides(self, sample_config: HFConfig) -> None:
        """Test order 2 chain with order 1 upstream, all no divides - all marked minor.

        Pattern: [order-1 NO div] → [order-2 NO div] → [order-2 NO div]
        Expected: All marked as minor (entire upstream network)
        """
        network_graph = {
            "7717438": ["7717428"],
            "7717428": ["order1_1", "order1_2"],
            "order1_1": [],
            "order1_2": [],
        }
        graph, node_indices = dict_to_graph(network_graph)

        # NO divides anywhere
        div_ids: set = set()

        fp_data = pl.DataFrame(
            {
                "flowpath_id": ["7717438", "7717428", "order1_1", "order1_2"],
                "areasqkm": [0.5, 0.4, 0.2, 0.3],
                "streamorder": [2, 2, 1, 1],
                "lengthkm": [0.3, 0.25, 0.1, 0.15],
                "hydroseq": [1, 2, 3, 4],
                "dnhydroseq": [0, 1, 2, 2],
                "mainstemlp": [100.0, 100.0, 50.0, 60.0],
            }
        )

        result = _trace_stack(
            start_id="7717438",
            fp=fp_data,
            div_ids=div_ids,
            cfg=sample_config,
            digraph=graph,
            node_indices=node_indices,
        )

        # ALL should be marked as minor
        assert "7717438" in result.minor_flowpaths
        assert "7717428" in result.minor_flowpaths
        assert "order1_1" in result.minor_flowpaths
        assert "order1_2" in result.minor_flowpaths

    def test_order2_no_divide_with_divide_upstream_marked_connector(self, sample_config: HFConfig) -> None:
        """Test order 2 no-divide with divide upstream - marked as connector.

        Pattern: [order-2 WITH div] → [order-2 NO div]
        Expected: No-divide marked as connector, not minor
        """
        network_graph = {"808455": ["808459"], "808459": []}
        graph, node_indices = dict_to_graph(network_graph)

        # 808459 has divide, 808455 doesn't
        div_ids = {"808459"}

        fp_data = pl.DataFrame(
            {
                "flowpath_id": ["808455", "808459"],
                "areasqkm": [1.5, 2.0],
                "streamorder": [2, 2],
                "lengthkm": [0.5, 0.8],
                "hydroseq": [1, 2],
                "dnhydroseq": [0, 1],
                "mainstemlp": [100.0, 100.0],
            }
        )

        result = _trace_stack(
            start_id="808455",
            fp=fp_data,
            div_ids=div_ids,
            cfg=sample_config,
            digraph=graph,
            node_indices=node_indices,
        )

        # 808455 should be marked as connector (not minor)
        assert "808455" in result.no_divide_connectors
        assert "808455" not in result.minor_flowpaths

        # 808459 should be processed normally
        assert "808459" in result.processed_flowpaths


class TestSingleUpstreamAreaThreshold:
    """Tests for Rule 9: Single upstream with area threshold checking."""

    def test_order2_below_threshold_aggregates_upstream(self, sample_config: HFConfig) -> None:
        """Test order 2 below threshold aggregates upstream until threshold.

        Pattern: Small order 2 should keep aggregating until DA threshold
        """
        network_graph = {"fp1": ["fp2"], "fp2": ["fp3"], "fp3": []}
        graph, node_indices = dict_to_graph(network_graph)

        div_ids = {"fp1", "fp2", "fp3"}

        # fp1 = 0.3 km², threshold = 10 km² - should aggregate upstream
        fp_data = pl.DataFrame(
            {
                "flowpath_id": ["fp1", "fp2", "fp3"],
                "areasqkm": [0.3, 2.0, 8.0],  # Total = 10.3, will stop at fp3
                "streamorder": [2, 2, 2],
                "lengthkm": [0.2, 0.5, 1.0],
                "hydroseq": [1, 2, 3],
                "dnhydroseq": [0, 1, 2],
                "mainstemlp": [100.0, 100.0, 100.0],
            }
        )

        result = _trace_stack(
            start_id="fp1",
            fp=fp_data,
            div_ids=div_ids,
            cfg=sample_config,
            digraph=graph,
            node_indices=node_indices,
        )

        # Should aggregate fp2 and fp3 into fp1
        assert ("fp1", "fp2") in result.aggregation_pairs
        assert ("fp1", "fp3") in result.aggregation_pairs

    def test_order2_above_threshold_doesnt_aggregate(self, sample_config: HFConfig) -> None:
        """Test order 2 already above threshold doesn't aggregate upstream.

        Pattern: Large order 2 shouldn't aggregate (already independent)
        """
        network_graph = {"fp1": ["fp2"], "fp2": []}
        graph, node_indices = dict_to_graph(network_graph)

        div_ids = {"fp1", "fp2"}

        # fp1 = 15 km² > threshold (10 km²) - should NOT aggregate
        fp_data = pl.DataFrame(
            {
                "flowpath_id": ["fp1", "fp2"],
                "areasqkm": [15.0, 5.0],
                "streamorder": [2, 2],
                "lengthkm": [2.0, 1.0],
                "hydroseq": [1, 2],
                "dnhydroseq": [0, 1],
                "mainstemlp": [100.0, 100.0],
            }
        )

        result = _trace_stack(
            start_id="fp1",
            fp=fp_data,
            div_ids=div_ids,
            cfg=sample_config,
            digraph=graph,
            node_indices=node_indices,
        )

        # Should NOT aggregate
        assert ("fp1", "fp2") not in result.aggregation_pairs

        # fp1 should be independent (large area)
        assert "fp1" in result.independent_flowpaths

    def test_no_divide_upstream_automatically_merged(self, sample_config: HFConfig) -> None:
        """Test no-divide upstream automatically merged, then continues aggregation.

        Pattern: [order-2 WITH div] → [order-2 NO div] → [order-2 WITH div (small)]
        Expected: No-divide merged, continues to aggregate by area
        """
        network_graph = {"fp1": ["fp_no_div"], "fp_no_div": ["fp3"], "fp3": []}
        graph, node_indices = dict_to_graph(network_graph)

        # fp_no_div has NO divide
        div_ids = {"fp1", "fp3"}

        fp_data = pl.DataFrame(
            {
                "flowpath_id": ["fp1", "fp_no_div", "fp3"],
                "areasqkm": [1.0, 0.5, 3.0],  # Total = 4.5, below threshold
                "streamorder": [2, 2, 2],
                "lengthkm": [0.5, 0.2, 0.8],
                "hydroseq": [1, 2, 3],
                "dnhydroseq": [0, 1, 2],
                "mainstemlp": [100.0, 100.0, 100.0],
            }
        )

        result = _trace_stack(
            start_id="fp1",
            fp=fp_data,
            div_ids=div_ids,
            cfg=sample_config,
            digraph=graph,
            node_indices=node_indices,
        )

        # Should aggregate no-divide AND continue to fp3
        assert ("fp1", "fp_no_div") in result.aggregation_pairs
        assert ("fp1", "fp3") in result.aggregation_pairs


class TestNoDivideConnectorAtConfluence:
    """Tests for no-divide connector at confluence points."""

    def test_confluence_with_no_divide_connector(self, sample_config: HFConfig) -> None:
        """Test confluence where connector has no divide.

        Pattern:
        [order-1 WITH div] ↘
                            → [order-2 NO div connector]
        [order-1 WITH div] ↗

        Expected: Connector marked, tributaries preserved
        """
        network_graph = {
            "808455": ["808461", "808457"],
            "808461": ["808365"],
            "808457": ["807787"],
            "808365": [],
            "807787": [],
        }
        graph, node_indices = dict_to_graph(network_graph)

        # 808455 (confluence) has NO divide
        div_ids = {"808461", "808457", "808365", "807787"}

        fp_data = pl.DataFrame(
            {
                "flowpath_id": ["808455", "808461", "808457", "808365", "807787"],
                "areasqkm": [5.0, 2.0, 1.5, 1.0, 0.8],
                "streamorder": [2, 2, 1, 1, 1],
                "lengthkm": [0.8, 0.6, 0.4, 0.3, 0.25],
                "hydroseq": [1, 2, 3, 4, 5],
                "dnhydroseq": [0, 1, 1, 2, 3],
                "mainstemlp": [100.0, 100.0, 50.0, 100.0, 50.0],
            }
        )

        result = _trace_stack(
            start_id="808455",
            fp=fp_data,
            div_ids=div_ids,
            cfg=sample_config,
            digraph=graph,
            node_indices=node_indices,
        )

        # 808455 should be marked as connector
        assert "808455" in result.no_divide_connectors

        # Tributaries should be processed normally (not lost!)
        assert "808461" in result.processed_flowpaths
        assert "808457" in result.processed_flowpaths
        assert "808365" in result.processed_flowpaths
        assert "807787" in result.processed_flowpaths


class TestCheckAndAggregateSameOrderNoDivideChain:
    """Direct tests for _check_and_aggregate_same_order_no_divide_chain function."""

    def test_same_order_chain_no_divides_all_minor(self) -> None:
        """Test same-order chain with no divides - all marked as minor."""
        network_graph = {"fp1": ["fp2"], "fp2": ["fp3"], "fp3": []}
        graph, node_indices = dict_to_graph(network_graph)

        div_ids: set = set()  # NO divides

        fp_data = pl.DataFrame(
            {
                "flowpath_id": ["fp1", "fp2", "fp3"],
                "areasqkm": [1.0, 1.0, 1.0],
                "streamorder": [2, 2, 2],  # All same order
                "lengthkm": [0.5, 0.5, 0.5],
                "hydroseq": [1, 2, 3],
                "dnhydroseq": [0, 1, 2],
                "mainstemlp": [100.0, 100.0, 100.0],
            }
        )

        result = Classifications()

        chain_has_no_divides = _check_and_aggregate_same_order_no_divide_chain(
            start_id="fp1",
            current_order=2,
            fp=fp_data,
            div_ids=div_ids,
            result=result,
            graph=graph,
            node_indices=node_indices,
        )

        assert chain_has_no_divides is True
        # All should be marked as minor
        assert "fp1" in result.minor_flowpaths
        assert "fp2" in result.minor_flowpaths
        assert "fp3" in result.minor_flowpaths

    def test_same_order_chain_with_divide_not_minor(
        self,
    ) -> None:
        """Test same-order chain WITH divide - should NOT be marked as minor."""
        network_graph = {"fp1": ["fp2"], "fp2": ["fp3"], "fp3": []}
        graph, node_indices = dict_to_graph(network_graph)

        # fp2 has a divide
        div_ids = {"fp2"}

        fp_data = pl.DataFrame(
            {
                "flowpath_id": ["fp1", "fp2", "fp3"],
                "areasqkm": [1.0, 1.0, 1.0],
                "streamorder": [2, 2, 2],
                "lengthkm": [0.5, 0.5, 0.5],
                "hydroseq": [1, 2, 3],
                "dnhydroseq": [0, 1, 2],
                "mainstemlp": [100.0, 100.0, 100.0],
            }
        )

        result = Classifications()

        chain_has_no_divides = _check_and_aggregate_same_order_no_divide_chain(
            start_id="fp1",
            current_order=2,
            fp=fp_data,
            div_ids=div_ids,
            result=result,
            graph=graph,
            node_indices=node_indices,
        )

        assert chain_has_no_divides is False
        # Should NOT be marked as minor
        assert "fp1" not in result.minor_flowpaths
        assert "fp2" not in result.minor_flowpaths

    def test_mixed_order_marks_lower_order_as_minor(
        self,
    ) -> None:
        """Test order 2 chain with order 1 upstream - order 1s marked as minor."""
        network_graph = {"fp1": ["fp2"], "fp2": ["order1_a", "order1_b"], "order1_a": [], "order1_b": []}
        graph, node_indices = dict_to_graph(network_graph)

        div_ids: set = set()  # NO divides

        fp_data = pl.DataFrame(
            {
                "flowpath_id": ["fp1", "fp2", "order1_a", "order1_b"],
                "areasqkm": [1.0, 1.0, 0.5, 0.5],
                "streamorder": [2, 2, 1, 1],  # Mixed orders
                "lengthkm": [0.5, 0.5, 0.3, 0.3],
                "hydroseq": [1, 2, 3, 4],
                "dnhydroseq": [0, 1, 2, 2],
                "mainstemlp": [100.0, 100.0, 50.0, 60.0],
            }
        )

        result = Classifications()

        chain_has_no_divides = _check_and_aggregate_same_order_no_divide_chain(
            start_id="fp1",
            current_order=2,
            fp=fp_data,
            div_ids=div_ids,
            result=result,
            graph=graph,
            node_indices=node_indices,
        )

        assert chain_has_no_divides is True
        # Order 2s marked as minor
        assert "fp1" in result.minor_flowpaths
        assert "fp2" in result.minor_flowpaths
        # Order 1s also marked as minor (entire upstream network)
        assert "order1_a" in result.minor_flowpaths
        assert "order1_b" in result.minor_flowpaths


class TestRuleAggregateSingleUpstreamOrder1:
    """Tests for order 1 behavior in single upstream aggregation."""

    def test_order1_aggregates_all_upstream(
        self, sample_flowpaths: pl.DataFrame, sample_config: HFConfig
    ) -> None:
        """Test order 1 aggregates all upstream regardless of area."""
        network_graph = {"fp1": ["fp2"], "fp2": ["fp3"], "fp3": []}
        graph, node_indices = dict_to_graph(network_graph)
        div_ids = {"fp1", "fp2", "fp3"}
        to_process: deque = deque()
        result = Classifications()

        fp_info = {
            "flowpath_id": "fp1",
            "areasqkm": 2.0,
            "streamorder": 1,
            "length_km": 3.0,
            "mainstemlp": 100.0,
        }

        upstream_info = [
            {"flowpath_id": "fp2", "areasqkm": 1.0, "streamorder": 1, "length_km": 2.0, "mainstemlp": 100.0}
        ]

        success = _rule_aggregate_single_upstream(
            current_id="fp1",
            fp_info=fp_info,
            upstream_info=upstream_info,
            cfg=sample_config,
            result=result,
            div_ids=div_ids,
            fp=sample_flowpaths,
            to_process=to_process,
            graph=graph,
            node_indices=node_indices,
        )

        assert success
        # Should aggregate fp2 AND fp3 (entire chain)
        assert ("fp2", "fp1") in result.aggregation_pairs
        assert ("fp3", "fp1") in result.aggregation_pairs
        assert "fp1" in result.processed_flowpaths

    def test_order1_no_upstream(self, sample_flowpaths: pl.DataFrame, sample_config: HFConfig) -> None:
        """Test order 1 with no upstream (headwater)."""
        network_graph: dict[str, list] = {"fp1": []}
        graph, node_indices = dict_to_graph(network_graph)
        div_ids = {"fp1"}
        to_process: deque = deque()
        result = Classifications()

        fp_info = {
            "flowpath_id": "fp1",
            "areasqkm": 2.0,
            "streamorder": 1,
            "length_km": 3.0,
            "mainstemlp": 100.0,
        }

        upstream_info: list = []  # No upstream

        success = _rule_aggregate_single_upstream(
            current_id="fp1",
            fp_info=fp_info,
            upstream_info=upstream_info,
            cfg=sample_config,
            result=result,
            div_ids=div_ids,
            fp=sample_flowpaths,
            to_process=to_process,
            graph=graph,
            node_indices=node_indices,
        )

        assert success
        assert "fp1" in result.processed_flowpaths
        # No aggregation pairs (headwater)
        assert len(result.aggregation_pairs) == 0


class TestRuleAggregateSingleUpstreamHigherOrderAreaThreshold:
    """Tests for higher order (2+) area threshold behavior."""

    def test_order2_already_above_threshold_no_aggregation(
        self, sample_flowpaths: pl.DataFrame, sample_config: HFConfig
    ) -> None:
        """Test order 2 already above threshold doesn't aggregate upstream."""
        network_graph = {"fp1": ["fp2"], "fp2": []}
        graph, node_indices = dict_to_graph(network_graph)
        div_ids = {"fp1", "fp2"}
        to_process: deque = deque()
        result = Classifications()

        # fp1 area = 15.0, threshold = 10.0
        fp_info = {
            "flowpath_id": "fp1",
            "areasqkm": 15.0,  # Above threshold!
            "streamorder": 2,
            "length_km": 5.0,
            "mainstemlp": 100.0,
        }

        upstream_info = [
            {"flowpath_id": "fp2", "areasqkm": 3.0, "streamorder": 2, "length_km": 2.0, "mainstemlp": 100.0}
        ]

        success = _rule_aggregate_single_upstream(
            current_id="fp1",
            fp_info=fp_info,
            upstream_info=upstream_info,
            cfg=sample_config,
            result=result,
            div_ids=div_ids,
            fp=sample_flowpaths,
            to_process=to_process,
            graph=graph,
            node_indices=node_indices,
        )

        assert success
        assert "fp1" in result.processed_flowpaths
        # Should NOT aggregate
        assert ("fp1", "fp2") not in result.aggregation_pairs
        # fp2 should be queued for independent processing
        assert "fp2" in to_process

    def test_order2_below_threshold_aggregates_upstream(self, sample_config: HFConfig) -> None:
        """Test order 2 below threshold aggregates upstream."""
        network_graph = {"fp1": ["fp2"], "fp2": []}
        graph, node_indices = dict_to_graph(network_graph)
        div_ids = {"fp1", "fp2"}
        to_process: deque = deque()
        result = Classifications()

        fp_data = pl.DataFrame(
            {
                "flowpath_id": ["fp1", "fp2"],
                "areasqkm": [0.5, 1.0],  # Total = 1.5, below threshold of 3.0
                "streamorder": [2, 2],
                "lengthkm": [5.0, 2.0],
                "hydroseq": [1, 2],
                "dnhydroseq": [0, 1],
                "mainstemlp": [100.0, 100.0],
            }
        )

        # fp1 area = 0.5, threshold = 3.0 (from config)
        fp_info = {
            "flowpath_id": "fp1",
            "areasqkm": 0.5,  # Below threshold
            "streamorder": 2,
            "length_km": 5.0,
            "mainstemlp": 100.0,
        }

        upstream_info = [
            {"flowpath_id": "fp2", "areasqkm": 1.0, "streamorder": 2, "length_km": 2.0, "mainstemlp": 100.0}
        ]

        success = _rule_aggregate_single_upstream(
            current_id="fp1",
            fp_info=fp_info,
            upstream_info=upstream_info,
            cfg=sample_config,
            result=result,
            div_ids=div_ids,
            fp=fp_data,
            to_process=to_process,
            graph=graph,
            node_indices=node_indices,
        )

        assert success
        # Should aggregate fp2
        assert ("fp1", "fp2") in result.aggregation_pairs

    def test_order3_below_threshold_aggregates_upstream(self, sample_config: HFConfig) -> None:
        """Test order 3 below threshold aggregates upstream."""
        network_graph = {"fp1": ["fp2"], "fp2": []}
        graph, node_indices = dict_to_graph(network_graph)
        div_ids = {"fp1", "fp2"}
        to_process: deque = deque()
        result = Classifications()

        fp_data = pl.DataFrame(
            {
                "flowpath_id": ["fp1", "fp2"],
                "areasqkm": [1.0, 1.5],  # Total = 2.5, below threshold of 3.0
                "streamorder": [3, 3],
                "lengthkm": [8.0, 5.0],
                "hydroseq": [1, 2],
                "dnhydroseq": [0, 1],
                "mainstemlp": [100.0, 100.0],
            }
        )

        fp_info = {
            "flowpath_id": "fp1",
            "areasqkm": 1.0,  # Below threshold
            "streamorder": 3,
            "length_km": 8.0,
            "mainstemlp": 100.0,
        }

        upstream_info = [
            {"flowpath_id": "fp2", "areasqkm": 1.5, "streamorder": 3, "length_km": 5.0, "mainstemlp": 100.0}
        ]

        success = _rule_aggregate_single_upstream(
            current_id="fp1",
            fp_info=fp_info,
            upstream_info=upstream_info,
            cfg=sample_config,
            result=result,
            div_ids=div_ids,
            fp=fp_data,
            to_process=to_process,
            graph=graph,
            node_indices=node_indices,
        )

        assert success
        assert ("fp1", "fp2") in result.aggregation_pairs


class TestRuleAggregateSingleUpstreamNoDivideHandling:
    """Tests for no-divide upstream handling."""

    def test_upstream_no_divide_aggregated_continues_to_next(self, sample_config: HFConfig) -> None:
        """Test upstream without divide is aggregated, then continues to next upstream."""
        network_graph = {"fp1": ["fp_no_div"], "fp_no_div": ["fp3"], "fp3": []}
        graph, node_indices = dict_to_graph(network_graph)

        # fp_no_div has NO divide
        div_ids = {"fp1", "fp3"}
        to_process: deque = deque()
        result = Classifications()

        fp_data = pl.DataFrame(
            {
                "flowpath_id": ["fp1", "fp_no_div", "fp3"],
                "areasqkm": [0.5, 0.3, 1.0],  # Total = 1.8, below threshold
                "streamorder": [2, 2, 2],
                "lengthkm": [2.0, 1.0, 1.5],
                "hydroseq": [1, 2, 3],
                "dnhydroseq": [0, 1, 2],
                "mainstemlp": [100.0, 100.0, 100.0],
            }
        )

        fp_info = {
            "flowpath_id": "fp1",
            "areasqkm": 0.5,
            "streamorder": 2,
            "length_km": 2.0,
            "mainstemlp": 100.0,
        }

        upstream_info = [
            {
                "flowpath_id": "fp_no_div",
                "areasqkm": 0.3,
                "streamorder": 2,
                "length_km": 1.0,
                "mainstemlp": 100.0,
            }
        ]

        success = _rule_aggregate_single_upstream(
            current_id="fp1",
            fp_info=fp_info,
            upstream_info=upstream_info,
            cfg=sample_config,
            result=result,
            div_ids=div_ids,
            fp=fp_data,
            to_process=to_process,
            graph=graph,
            node_indices=node_indices,
        )

        assert success
        # Should aggregate the no-divide
        assert ("fp1", "fp_no_div") in result.aggregation_pairs
        assert "fp_no_div" in result.processed_flowpaths
        # Should continue to fp3 (has divide)
        assert ("fp1", "fp3") in result.aggregation_pairs

    def test_chain_of_no_divides_aggregated(self, sample_config: HFConfig) -> None:
        """Test chain of no-divides all aggregated."""
        network_graph = {
            "fp1": ["fp_no_div1"],
            "fp_no_div1": ["fp_no_div2"],
            "fp_no_div2": ["fp3"],
            "fp3": [],
        }
        graph, node_indices = dict_to_graph(network_graph)

        # Only fp1 and fp3 have divides
        div_ids = {"fp1", "fp3"}
        to_process: deque = deque()
        result = Classifications()

        fp_data = pl.DataFrame(
            {
                "flowpath_id": ["fp1", "fp_no_div1", "fp_no_div2", "fp3"],
                "areasqkm": [0.5, 0.2, 0.3, 1.5],  # Total = 2.5, below threshold
                "streamorder": [2, 2, 2, 2],
                "lengthkm": [2.0, 0.5, 0.3, 1.5],
                "hydroseq": [1, 2, 3, 4],
                "dnhydroseq": [0, 1, 2, 3],
                "mainstemlp": [100.0, 100.0, 100.0, 100.0],
            }
        )

        fp_info = {
            "flowpath_id": "fp1",
            "areasqkm": 0.5,
            "streamorder": 2,
            "length_km": 2.0,
            "mainstemlp": 100.0,
        }

        upstream_info = [
            {
                "flowpath_id": "fp_no_div1",
                "areasqkm": 0.2,
                "streamorder": 2,
                "length_km": 0.5,
                "mainstemlp": 100.0,
            }
        ]

        success = _rule_aggregate_single_upstream(
            current_id="fp1",
            fp_info=fp_info,
            upstream_info=upstream_info,
            cfg=sample_config,
            result=result,
            div_ids=div_ids,
            fp=fp_data,
            to_process=to_process,
            graph=graph,
            node_indices=node_indices,
        )

        assert success
        # Should aggregate both no-divides
        assert ("fp1", "fp_no_div1") in result.aggregation_pairs
        assert ("fp1", "fp_no_div2") in result.aggregation_pairs
        # fp3 has a divide, so it stops after the no-divide chain and queues fp3
        assert "fp3" in to_process

    def test_no_divide_upstream_headwater(
        self, sample_flowpaths: pl.DataFrame, sample_config: HFConfig
    ) -> None:
        """Test no-divide upstream with no further upstream (headwater)."""
        network_graph = {"fp1": ["fp_no_div"], "fp_no_div": []}
        graph, node_indices = dict_to_graph(network_graph)

        # fp_no_div has NO divide
        div_ids = {"fp1"}
        to_process: deque = deque()
        result = Classifications()

        fp_info = {
            "flowpath_id": "fp1",
            "areasqkm": 2.0,
            "streamorder": 2,
            "length_km": 3.0,
            "mainstemlp": 100.0,
        }

        upstream_info = [
            {
                "flowpath_id": "fp_no_div",
                "areasqkm": 0.5,
                "streamorder": 2,
                "length_km": 1.0,
                "mainstemlp": 100.0,
            }
        ]

        success = _rule_aggregate_single_upstream(
            current_id="fp1",
            fp_info=fp_info,
            upstream_info=upstream_info,
            cfg=sample_config,
            result=result,
            div_ids=div_ids,
            fp=sample_flowpaths,
            to_process=to_process,
            graph=graph,
            node_indices=node_indices,
        )

        assert success
        # Should aggregate the no-divide
        assert ("fp1", "fp_no_div") in result.aggregation_pairs
        assert "fp_no_div" in result.processed_flowpaths

    def test_no_divide_upstream_with_confluence(self, sample_config: HFConfig) -> None:
        """Test no-divide upstream that leads to confluence (multiple upstream)."""
        network_graph = {"fp1": ["fp_no_div"], "fp_no_div": ["fp2", "fp3"], "fp2": [], "fp3": []}
        graph, node_indices = dict_to_graph(network_graph)

        # fp_no_div has NO divide
        div_ids = {"fp1", "fp2", "fp3"}
        to_process: deque = deque()
        result = Classifications()

        fp_data = pl.DataFrame(
            {
                "flowpath_id": ["fp1", "fp_no_div", "fp2", "fp3"],
                "areasqkm": [1.5, 0.5, 0.8, 0.7],
                "streamorder": [2, 2, 2, 2],
                "lengthkm": [4.0, 1.0, 1.5, 1.2],
                "hydroseq": [1, 2, 3, 4],
                "dnhydroseq": [0, 1, 2, 2],
                "mainstemlp": [100.0, 100.0, 100.0, 100.0],
            }
        )

        fp_info = {
            "flowpath_id": "fp1",
            "areasqkm": 1.5,  # Below threshold of 3.0
            "streamorder": 2,
            "length_km": 4.0,
            "mainstemlp": 100.0,
        }

        upstream_info = [
            {
                "flowpath_id": "fp_no_div",
                "areasqkm": 0.5,
                "streamorder": 2,
                "length_km": 1.0,
                "mainstemlp": 100.0,
            }
        ]

        success = _rule_aggregate_single_upstream(
            current_id="fp1",
            fp_info=fp_info,
            upstream_info=upstream_info,
            cfg=sample_config,
            result=result,
            div_ids=div_ids,
            fp=fp_data,
            to_process=to_process,
            graph=graph,
            node_indices=node_indices,
        )

        assert success
        # Should aggregate the no-divide
        assert ("fp1", "fp_no_div") in result.aggregation_pairs
        # Multiple upstream should be queued
        assert "fp2" in to_process
        assert "fp3" in to_process

    def test_no_divide_then_divide_continues_area_aggregation(self, sample_config: HFConfig) -> None:
        """Test no-divide followed by divide continues area-based aggregation."""
        network_graph = {"fp1": ["fp_no_div"], "fp_no_div": ["fp3"], "fp3": ["fp4"], "fp4": []}
        graph, node_indices = dict_to_graph(network_graph)

        # fp_no_div has NO divide
        div_ids = {"fp1", "fp3", "fp4"}
        to_process: deque = deque()
        result = Classifications()

        # Total area will be 0.5 + 0.2 + 0.5 + 0.8 = 2.0 (below threshold of 3.0)
        fp_data_with_extras = pl.DataFrame(
            {
                "flowpath_id": ["fp1", "fp_no_div", "fp3", "fp4"],
                "areasqkm": [0.5, 0.2, 0.5, 0.8],  # Total = 2.0, below threshold
                "streamorder": [2, 2, 2, 2],
                "lengthkm": [2.0, 1.0, 1.5, 2.0],
                "hydroseq": [1, 2, 3, 4],
                "dnhydroseq": [0, 1, 2, 3],
                "mainstemlp": [100.0, 100.0, 100.0, 100.0],
            }
        )

        fp_info = {
            "flowpath_id": "fp1",
            "areasqkm": 0.5,
            "streamorder": 2,
            "length_km": 2.0,
            "mainstemlp": 100.0,
        }

        upstream_info = [
            {
                "flowpath_id": "fp_no_div",
                "areasqkm": 0.2,
                "streamorder": 2,
                "length_km": 1.0,
                "mainstemlp": 100.0,
            }
        ]

        success = _rule_aggregate_single_upstream(
            current_id="fp1",
            fp_info=fp_info,
            upstream_info=upstream_info,
            cfg=sample_config,
            result=result,
            div_ids=div_ids,
            fp=fp_data_with_extras,
            to_process=to_process,
            graph=graph,
            node_indices=node_indices,
        )

        assert success
        # Should aggregate no-divide AND continue through fp3 and fp4
        assert ("fp1", "fp_no_div") in result.aggregation_pairs
        assert ("fp1", "fp3") in result.aggregation_pairs
        assert ("fp1", "fp4") in result.aggregation_pairs


class TestRuleAggregateSingleUpstreamAreaAccumulation:
    """Tests for cumulative area tracking and threshold stopping."""

    def test_stops_at_threshold(self, sample_config: HFConfig) -> None:
        """Test aggregation stops when cumulative area hits threshold."""
        network_graph = {"fp1": ["fp2"], "fp2": ["fp3"], "fp3": ["fp4"], "fp4": []}
        graph, node_indices = dict_to_graph(network_graph)
        div_ids = {"fp1", "fp2", "fp3", "fp4"}
        to_process: deque = deque()
        result = Classifications()

        # fp1=0.5, fp2=1.0 = 1.5 (below 3.0), fp3=2.0 would make it 3.5 (exceeds)
        fp_data = pl.DataFrame(
            {
                "flowpath_id": ["fp1", "fp2", "fp3", "fp4"],
                "areasqkm": [0.5, 1.0, 2.0, 5.0],
                "streamorder": [2, 2, 2, 2],
                "lengthkm": [1.0, 1.0, 2.0, 2.0],
                "hydroseq": [1, 2, 3, 4],
                "dnhydroseq": [0, 1, 2, 3],
                "mainstemlp": [100.0, 100.0, 100.0, 100.0],
            }
        )

        fp_info = {
            "flowpath_id": "fp1",
            "areasqkm": 0.5,
            "streamorder": 2,
            "length_km": 1.0,
            "mainstemlp": 100.0,
        }

        upstream_info = [
            {"flowpath_id": "fp2", "areasqkm": 1.0, "streamorder": 2, "length_km": 1.0, "mainstemlp": 100.0}
        ]

        result.processed_flowpaths.add("fp1")
        success = _rule_aggregate_single_upstream(
            current_id="fp1",
            fp_info=fp_info,
            upstream_info=upstream_info,
            cfg=sample_config,
            result=result,
            div_ids=div_ids,
            fp=fp_data,
            to_process=to_process,
            graph=graph,
            node_indices=node_indices,
        )

        assert success
        assert {"fp1", "fp2", "fp3"} == result.processed_flowpaths
        assert "fp4" in to_process
        assert ("fp1", "fp2") in result.aggregation_pairs
        assert ("fp1", "fp3") in result.aggregation_pairs
