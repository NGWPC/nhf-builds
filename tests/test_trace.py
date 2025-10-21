"""
Tests for Hydrofabric tracing rules

Rule 1: Independent - Large Area
- Test case: flowpath with areasqkm > threshold becomes independent
- Test case: flowpath with areasqkm < threshold is NOT marked independent

Rule 2: Independent - Connector Segment
- Test case: small area (< threshold) with two upstream segments where both have stream_order > 1
- Test case: small area (< threshold) with large order-1 upstream (areasqkm > threshold) becomes connector
- Test case: small area (< threshold) with large order-1 upstream aggregates small order-1 upstreams as minor flowpaths
- Test case: connector segment with 3 higher-order upstreams (all stream_order > 1)
- Test case: single upstream does NOT trigger connector rule
- Test case: small order-1 upstream (areasqkm < threshold) does NOT trigger connector rule
- Test case: large area (> threshold) does NOT trigger connector regardless of upstream

Rule 3: Aggregate - Single Upstream
- Test case: small area (< threshold) with single upstream aggregates into that upstream
- Test case: cumulative merge areas are tracked correctly across multiple aggregations
- Test case: multiple upstream does NOT trigger single upstream rule
- Test case: large area (> threshold) is NOT aggregated even with single upstream

Rule 4: Aggregate - Order 1 Stream (All Upstream)
- Test case: stream_order == 1 aggregates all direct upstream flowpaths into current
- Test case: stream_order == 1 recursively aggregates entire upstream branch
- Test case: higher-order streams (order > 1) do NOT trigger this rule
- Test case: order-1 with no upstream does NOT trigger rule

Rule 5: Aggregate - Order 2 with Order 1s
- Test case: stream_order == 2 with two stream_order == 1 upstreams
- Test case: stream_order == 2 with two stream_order == 1 upstreams that have upstream branches
- Test case: stream_order == 2 with three stream_order == 1 upstreams (largest stays non-minor)
- Test case: stream_order == 2 with four stream_order == 1 upstreams (largest stays non-minor)
- Test case: stream_order != 2 does NOT trigger rule
- Test case: less than 2 upstreams does NOT trigger rule
- Test case: all upstreams must be order-1 or rule doesn't trigger

Rule 6: Aggregate - Mixed Upstream Orders
- Test case: small order-1 upstream becomes minor flowpath aggregated to current
- Test case: large order-1 (areasqkm >= threshold) is NOT marked as minor
- Test case: multiple small order-1s all become minor flowpaths
- Test case: small order-1 with large order-1 upstream triggers rule (only small becomes minor)
- Test case: only order-1 upstreams do NOT trigger mixed rule
- Test case: stream_order == 1 does NOT trigger this rule

Rule 7: Aggregate - Same Order with Small Area
- Test case: small area merges into same-order upstream
- Test case: cumulative merge areas tracked correctly
- Test case: large area (> threshold) does NOT trigger rule
- Test case: no same-order upstream does NOT trigger rule

Helper Functions:
- Test case: recursive aggregation of all upstream flowpaths into target
- Test case: recursive aggregation prevents reprocessing already-processed flowpaths
- Test case: get flowpath info from indexed dataframe

Trace and Aggregate Upstream:
- Test case: trace single upstream flowpath
- Test case: trace multiple direct upstream flowpaths
- Test case: trace deep recursive upstream chain
- Test case: trace complex branching network
- Test case: is_minor flag marks flowpaths as minor
- Test case: flowpaths NOT marked minor when is_minor=False
- Test case: current_id equals target_id edge case
- Test case: trace flowpath with no upstream (empty network)
- Test case: preserve existing classifications when tracing
"""

import pandas as pd
import pytest

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.hydrofabric.trace import (
    Classifications,
    _aggregate_all_upstream_recursive,
    _get_flowpath_info,
    _rule_aggregate_mixed_upstream_orders,
    _rule_aggregate_order1_all_upstream,
    _rule_aggregate_order2_with_order1s,
    _rule_aggregate_same_order_small_area,
    _rule_aggregate_single_upstream,
    _rule_independent_connector,
    _rule_independent_large_area,
    _trace_and_aggregate_upstream,
)


@pytest.fixture
def sample_flowpath_data() -> pd.DataFrame:
    """Create sample flowpath data for unit testing individual rules."""
    data = {
        "flowpath_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "totdasqkm": [100.0, 50.0, 25.0, 10.0, 5.0, 2.0, 1.0, 0.5],
        "areasqkm_left": [5.0, 2.5, 1.5, 0.8, 2.0, 1.0, 0.5, 0.3],
        "lengthkm": [10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0],
        "streamorder": [3, 2, 2, 1, 1, 1, 1, 1],
        "hydroseq": [1, 2, 3, 4, 5, 6, 7, 8],
        "dnhydroseq": [0, 1, 1, 2, 3, 4, 4, 5],
        "mainstemlp": [100, 100, 200, 100, 200, 100, 100, 200],
    }
    return pd.DataFrame(data)


class TestRule1IndependentLargeArea:
    """Tests for Rule 1: Independent - Large Area"""

    def test_large_area_becomes_independent(self, sample_config: HFConfig) -> None:
        """Test case: flowpath with areasqkm > threshold becomes independent"""
        result = Classifications()
        fp_info = {"areasqkm": 5.0, "flowpath_id": "fp1"}
        matched = _rule_independent_large_area("fp1", fp_info, sample_config, result)
        assert matched
        assert "fp1" in result.independent_flowpaths
        assert len(result.independent_flowpaths) == 1

    def test_small_area_not_independent(self, sample_config: HFConfig) -> None:
        """Test case: flowpath with areasqkm < threshold is NOT marked independent"""
        result = Classifications()
        fp_info = {"areasqkm": 2.0, "flowpath_id": "fp1"}

        matched = _rule_independent_large_area("fp1", fp_info, sample_config, result)
        assert matched is False
        assert len(result.independent_flowpaths) == 0


class TestRule2IndependentConnector:
    """Tests for Rule 2: Independent - Connector Segment"""

    def test_connector_two_higher_order_upstream(self, sample_config: HFConfig) -> None:
        """Test case: small area (< threshold) with two upstream segments where both have stream_order > 1"""
        result = Classifications()
        network_graph = {"fp1": ["up1", "up2"], "up1": [], "up2": []}
        fp_info = {"areasqkm": 2.0, "flowpath_id": "fp1", "stream_order": 3}
        upstream_info = [
            {"stream_order": 2, "flowpath_id": "up1", "areasqkm": 1.5},
            {"stream_order": 2, "flowpath_id": "up2", "areasqkm": 1.8},
        ]

        matched = _rule_independent_connector(
            "fp1", fp_info, upstream_info, sample_config, network_graph, result
        )

        assert matched
        assert "fp1" in result.connector_segments
        assert len(result.connector_segments) == 1

    def test_connector_with_large_order1_upstream(self, sample_config: HFConfig) -> None:
        """Test case: small area (< threshold) with large order-1 upstream (areasqkm > threshold) becomes connector"""
        result = Classifications()
        network_graph = {"fp1": ["up1", "up2"], "up1": [], "up2": []}
        fp_info = {"areasqkm": 2.0, "flowpath_id": "fp1", "stream_order": 2}
        upstream_info = [
            {"stream_order": 1, "flowpath_id": "up1", "areasqkm": 5.0},
            {"stream_order": 2, "flowpath_id": "up2", "areasqkm": 1.5},
        ]

        matched = _rule_independent_connector(
            "fp1", fp_info, upstream_info, sample_config, network_graph, result
        )

        assert matched
        assert "fp1" in result.connector_segments

    def test_not_connector_single_upstream(self, sample_config: HFConfig) -> None:
        """Test case: single upstream does NOT trigger connector rule"""
        result = Classifications()
        network_graph = {"fp1": ["up1"], "up1": []}
        fp_info = {"areasqkm": 2.0, "flowpath_id": "fp1"}
        upstream_info = [{"stream_order": 2, "flowpath_id": "up1", "areasqkm": 1.5}]

        matched = _rule_independent_connector(
            "fp1", fp_info, upstream_info, sample_config, network_graph, result
        )

        assert matched is False
        assert len(result.connector_segments) == 0

    def test_not_connector_small_order1_upstream(self, sample_config: HFConfig) -> None:
        """Test case: small order-1 upstream (areasqkm < threshold) does NOT trigger connector rule"""
        result = Classifications()
        network_graph = {"fp1": ["up1", "up2"], "up1": [], "up2": []}
        fp_info = {"areasqkm": 2.0, "flowpath_id": "fp1", "stream_order": 2}
        upstream_info = [
            {"stream_order": 1, "flowpath_id": "up1", "areasqkm": 1.0},
            {"stream_order": 2, "flowpath_id": "up2", "areasqkm": 1.5},
        ]

        matched = _rule_independent_connector(
            "fp1", fp_info, upstream_info, sample_config, network_graph, result
        )

        assert matched is False
        assert len(result.connector_segments) == 0

    def test_connector_large_area_not_triggered(self, sample_config: HFConfig) -> None:
        """Test case: large area (> threshold) does NOT trigger connector regardless of upstream"""
        result = Classifications()
        network_graph = {"fp1": ["up1", "up2"], "up1": [], "up2": []}
        fp_info = {"areasqkm": 5.0, "flowpath_id": "fp1", "stream_order": 3}
        upstream_info = [
            {"stream_order": 2, "flowpath_id": "up1", "areasqkm": 1.5},
            {"stream_order": 2, "flowpath_id": "up2", "areasqkm": 1.8},
        ]

        matched = _rule_independent_connector(
            "fp1", fp_info, upstream_info, sample_config, network_graph, result
        )

        assert matched is False

    def test_connector_aggregates_small_order1_upstreams(self, sample_config: HFConfig) -> None:
        """Test case: connector segment with small order-1 upstreams aggregates them as minor flowpaths"""
        result = Classifications()
        network_graph = {
            "fp1": ["up1", "up2", "up3"],
            "up1": [],  # Large order-1 (stays independent)
            "up2": [],  # Higher-order stream
            "up3": ["up3a"],  # Small order-1 with upstream
            "up3a": [],
        }
        fp_info = {"areasqkm": 2.0, "flowpath_id": "fp1", "stream_order": 3}
        upstream_info = [
            {"stream_order": 1, "flowpath_id": "up1", "areasqkm": 5.0},  # Large order-1
            {"stream_order": 2, "flowpath_id": "up2", "areasqkm": 1.5},  # Higher-order
            {"stream_order": 1, "flowpath_id": "up3", "areasqkm": 1.0},  # Small order-1
        ]

        matched = _rule_independent_connector(
            "fp1", fp_info, upstream_info, sample_config, network_graph, result
        )

        assert matched
        assert "fp1" in result.connector_segments
        # Small order-1 upstream should be aggregated and marked as minor
        assert ("up3", "fp1") in result.aggregation_pairs
        assert ("up3a", "fp1") in result.aggregation_pairs
        assert "up3" in result.minor_flowpaths
        assert "up3a" in result.minor_flowpaths
        # Large order-1 should NOT be aggregated or marked as minor
        assert ("up1", "fp1") not in result.aggregation_pairs
        assert "up1" not in result.minor_flowpaths
        # up3 and up3a should be marked as processed
        assert "up3" in result.processed_flowpaths
        assert "up3a" in result.processed_flowpaths

    def test_connector_three_or_more_higher_order_upstreams(self, sample_config: HFConfig) -> None:
        """Test case: connector segment with 3 higher-order upstreams (all stream_order > 1)"""
        result = Classifications()
        network_graph = {"fp1": ["up1", "up2", "up3"], "up1": [], "up2": [], "up3": []}
        fp_info = {"areasqkm": 2.0, "flowpath_id": "fp1", "stream_order": 4}
        upstream_info = [
            {"stream_order": 2, "flowpath_id": "up1", "areasqkm": 1.5},
            {"stream_order": 2, "flowpath_id": "up2", "areasqkm": 1.8},
            {"stream_order": 3, "flowpath_id": "up3", "areasqkm": 2.0},
        ]

        matched = _rule_independent_connector(
            "fp1", fp_info, upstream_info, sample_config, network_graph, result
        )

        assert matched
        assert "fp1" in result.connector_segments
        assert len(result.connector_segments) == 1
        # No aggregations should occur (all are higher-order)
        assert len(result.aggregation_pairs) == 0
        assert len(result.minor_flowpaths) == 0


class TestRule3AggregateSingleUpstream:
    """Tests for Rule 3: Aggregate - Single Upstream"""

    def test_single_upstream_aggregation(self, sample_config: HFConfig) -> None:
        """Test case: small area (< threshold) with single upstream aggregates into that upstream"""
        result = Classifications()
        fp_info = {"areasqkm": 2.0, "flowpath_id": "fp1"}
        upstream_info = [{"flowpath_id": "up1", "areasqkm": 1.5}]

        matched = _rule_aggregate_single_upstream("fp1", fp_info, upstream_info, sample_config, result)

        assert matched
        assert ("fp1", "up1") in result.aggregation_pairs
        assert "up1" in result.cumulative_merge_areas

    def test_cumulative_area_tracking(self, sample_config: HFConfig) -> None:
        """Test case: cumulative merge areas are tracked correctly across multiple aggregations"""
        result = Classifications()
        result.cumulative_merge_areas["fp1"] = 5.0  # Pre-existing cumulative area

        fp_info = {"areasqkm": 2.0, "flowpath_id": "fp1"}
        upstream_info = [{"flowpath_id": "up1", "areasqkm": 1.5}]

        matched = _rule_aggregate_single_upstream("fp1", fp_info, upstream_info, sample_config, result)

        assert matched
        assert result.cumulative_merge_areas["up1"] == 5.0

    def test_not_single_upstream_multiple(self, sample_config: HFConfig) -> None:
        """Test case: multiple upstream does NOT trigger single upstream rule"""
        result = Classifications()
        fp_info = {"areasqkm": 2.0, "flowpath_id": "fp1"}
        upstream_info = [{"flowpath_id": "up1", "areasqkm": 1.5}, {"flowpath_id": "up2", "areasqkm": 1.2}]

        matched = _rule_aggregate_single_upstream("fp1", fp_info, upstream_info, sample_config, result)

        assert matched is False
        assert len(result.aggregation_pairs) == 0

    def test_large_area_not_aggregated(self, sample_config: HFConfig) -> None:
        """Test case: large area (> threshold) is NOT aggregated even with single upstream"""
        result = Classifications()
        fp_info = {"areasqkm": 5.0, "flowpath_id": "fp1"}
        upstream_info = [{"flowpath_id": "up1", "areasqkm": 1.5}]

        matched = _rule_aggregate_single_upstream("fp1", fp_info, upstream_info, sample_config, result)

        assert matched is False
        assert len(result.aggregation_pairs) == 0


class TestRule4AggregateOrder1AllUpstream:
    """Tests for Rule 4: Aggregate - Order 1 Stream (All Upstream)"""

    def test_order1_aggregates_all_upstream(self, sample_config: HFConfig) -> None:
        """Test case: stream_order == 1 aggregates all direct upstream flowpaths into current"""
        result = Classifications()
        network_graph = {"fp1": ["up1", "up2"], "up1": [], "up2": []}

        fp_info = {"stream_order": 1, "flowpath_id": "fp1"}
        upstream_info = [{"flowpath_id": "up1", "stream_order": 1}, {"flowpath_id": "up2", "stream_order": 1}]

        matched = _rule_aggregate_order1_all_upstream("fp1", fp_info, upstream_info, network_graph, result)

        assert matched
        assert ("up1", "fp1") in result.aggregation_pairs
        assert ("up2", "fp1") in result.aggregation_pairs
        assert len(result.aggregation_pairs) == 2

    def test_order1_recursive_aggregation(self, sample_config: HFConfig) -> None:
        """Test case: stream_order == 1 recursively aggregates entire upstream branch"""
        result = Classifications()
        network_graph = {"fp1": ["up1"], "up1": ["up2"], "up2": ["up3"], "up3": []}

        fp_info = {"stream_order": 1, "flowpath_id": "fp1"}
        upstream_info = [{"flowpath_id": "up1", "stream_order": 1}]

        matched = _rule_aggregate_order1_all_upstream("fp1", fp_info, upstream_info, network_graph, result)

        assert matched
        assert ("up1", "fp1") in result.aggregation_pairs
        assert ("up2", "fp1") in result.aggregation_pairs
        assert ("up3", "fp1") in result.aggregation_pairs
        assert "up1" in result.processed_flowpaths
        assert "up2" in result.processed_flowpaths
        assert "up3" in result.processed_flowpaths

    def test_higher_order_not_aggregated(self, sample_config: HFConfig) -> None:
        """Test case: higher-order streams (order > 1) do NOT trigger this rule"""
        result = Classifications()
        network_graph = {"fp1": ["up1"]}

        fp_info = {"stream_order": 2, "flowpath_id": "fp1"}
        upstream_info = [{"flowpath_id": "up1", "stream_order": 1}]

        matched = _rule_aggregate_order1_all_upstream("fp1", fp_info, upstream_info, network_graph, result)

        assert matched is False
        assert len(result.aggregation_pairs) == 0

    def test_order1_no_upstream(self, sample_config: HFConfig) -> None:
        """Test case: order-1 with no upstream does NOT trigger rule"""
        result = Classifications()
        network_graph: dict[str, list] = {"fp1": []}

        fp_info = {"stream_order": 1, "flowpath_id": "fp1"}
        upstream_info: list = []

        matched = _rule_aggregate_order1_all_upstream("fp1", fp_info, upstream_info, network_graph, result)

        assert matched is False
        assert len(result.aggregation_pairs) == 0


class TestRule5AggregateOrder2WithOrder1s:
    """Tests for Rule 5: Aggregate - Order 2 with Two Order 1s"""

    def test_order2_with_two_order1s(self) -> None:
        """Test case: stream_order == 2 with two stream_order == 1 upstreams"""
        result = Classifications()
        network_graph = {"fp1": ["up1", "up2"], "up1": [], "up2": []}
        fp_info = {"stream_order": 2, "flowpath_id": "fp1", "mainstemlp": 100}
        upstream_info = [
            {
                "stream_order": 1,
                "flowpath_id": "up1",
                "total_drainage_area_sqkm": 10.0,
                "mainstemlp": 100,
            },
            {
                "stream_order": 1,
                "flowpath_id": "up2",
                "total_drainage_area_sqkm": 5.0,
                "mainstemlp": 200,
            },
        ]

        matched = _rule_aggregate_order2_with_order1s("fp1", fp_info, upstream_info, network_graph, result)

        assert matched
        # Both upstreams aggregated into fp1
        assert ("up1", "fp1") in result.aggregation_pairs  # On mainstem
        assert ("up2", "fp1") in result.aggregation_pairs  # Off mainstem (minor)
        assert "up2" in result.minor_flowpaths
        assert "up1" not in result.minor_flowpaths  # On mainstem should NOT be minor
        assert "fp1" in result.subdivide_candidates
        assert "fp1" in result.upstream_merge_points
        # Both upstreams and current should be marked as processed
        assert "up1" in result.processed_flowpaths
        assert "up2" in result.processed_flowpaths
        assert "fp1" in result.processed_flowpaths

    def test_order2_with_upstream_branches(self) -> None:
        """Test case: order-2 with order-1 upstreams that each have a single upstream chain"""
        result = Classifications()
        network_graph = {"fp1": ["up1", "up2"], "up1": ["up1a"], "up2": ["up2a"], "up1a": [], "up2a": []}
        fp_info = {"stream_order": 2, "flowpath_id": "fp1", "mainstemlp": 100}
        upstream_info = [
            {
                "stream_order": 1,
                "flowpath_id": "up1",
                "total_drainage_area_sqkm": 10.0,
                "mainstemlp": 100,
            },
            {
                "stream_order": 1,
                "flowpath_id": "up2",
                "total_drainage_area_sqkm": 5.0,
                "mainstemlp": 200,
            },
        ]

        matched = _rule_aggregate_order2_with_order1s("fp1", fp_info, upstream_info, network_graph, result)

        assert matched
        # All should be aggregated into fp1
        assert ("up1", "fp1") in result.aggregation_pairs
        assert ("up1a", "fp1") in result.aggregation_pairs
        assert ("up2", "fp1") in result.aggregation_pairs
        assert ("up2a", "fp1") in result.aggregation_pairs
        # Only up2 branch should be minor
        assert "up2" in result.minor_flowpaths
        assert "up2a" in result.minor_flowpaths
        # up1 branch should NOT be minor
        assert "up1" not in result.minor_flowpaths
        assert "up1a" not in result.minor_flowpaths

    def test_not_order2(self) -> None:
        """Test case: stream_order != 2 does NOT trigger rule"""
        result = Classifications()
        network_graph = {"fp1": ["up1", "up2"], "up1": [], "up2": []}
        fp_info = {"stream_order": 3, "flowpath_id": "fp1", "mainstemlp": 100}
        upstream_info = [
            {
                "stream_order": 1,
                "flowpath_id": "up1",
                "total_drainage_area_sqkm": 10.0,
                "mainstemlp": 100,
            },
            {
                "stream_order": 1,
                "flowpath_id": "up2",
                "total_drainage_area_sqkm": 5.0,
                "mainstemlp": 200,
            },
        ]

        matched = _rule_aggregate_order2_with_order1s("fp1", fp_info, upstream_info, network_graph, result)

        assert matched is False

    def test_less_than_two_upstreams(self) -> None:
        """Test case: less than 2 upstreams does NOT trigger rule"""
        result = Classifications()
        network_graph = {"fp1": ["up1"], "up1": []}
        fp_info = {"stream_order": 2, "flowpath_id": "fp1", "mainstemlp": 100}
        upstream_info = [
            {
                "stream_order": 1,
                "flowpath_id": "up1",
                "total_drainage_area_sqkm": 10.0,
                "mainstemlp": 100,
            }
        ]

        matched = _rule_aggregate_order2_with_order1s("fp1", fp_info, upstream_info, network_graph, result)

        assert matched is False

    def test_not_all_order1(self) -> None:
        """Test case: all upstreams must be order-1 or rule doesn't trigger"""
        result = Classifications()
        network_graph = {"fp1": ["up1", "up2"], "up1": [], "up2": []}
        fp_info = {"stream_order": 2, "flowpath_id": "fp1", "mainstemlp": 100}
        upstream_info = [
            {
                "stream_order": 1,
                "flowpath_id": "up1",
                "total_drainage_area_sqkm": 10.0,
                "mainstemlp": 100,
            },
            {
                "stream_order": 2,
                "flowpath_id": "up2",
                "total_drainage_area_sqkm": 5.0,
                "mainstemlp": 100,
            },
        ]

        matched = _rule_aggregate_order2_with_order1s("fp1", fp_info, upstream_info, network_graph, result)

        assert matched is False

    def test_order2_with_three_order1s(self) -> None:
        """Test case: order-2 stream with 3 order-1 upstreams - largest stays non-minor, others become minor"""
        result = Classifications()
        network_graph = {"fp1": ["up1", "up2", "up3"], "up1": ["up1a"], "up2": [], "up3": [], "up1a": []}
        fp_info = {"stream_order": 2, "flowpath_id": "fp1", "mainstemlp": 100}
        upstream_info = [
            {
                "stream_order": 1,
                "flowpath_id": "up1",
                "total_drainage_area_sqkm": 10.0,
                "mainstemlp": 100,
            },
            {
                "stream_order": 1,
                "flowpath_id": "up2",
                "total_drainage_area_sqkm": 5.0,
                "mainstemlp": 200,
            },
            {
                "stream_order": 1,
                "flowpath_id": "up3",
                "total_drainage_area_sqkm": 3.0,
                "mainstemlp": 300,
            },
        ]

        matched = _rule_aggregate_order2_with_order1s("fp1", fp_info, upstream_info, network_graph, result)

        assert matched
        # All upstreams aggregated into fp1
        assert ("up1", "fp1") in result.aggregation_pairs
        assert ("up1a", "fp1") in result.aggregation_pairs
        assert ("up2", "fp1") in result.aggregation_pairs
        assert ("up3", "fp1") in result.aggregation_pairs
        # Only up1 (on mainstem) should NOT be minor
        assert "up1" not in result.minor_flowpaths
        assert "up1a" not in result.minor_flowpaths
        # Off-mainstem should be minor
        assert "up2" in result.minor_flowpaths
        assert "up3" in result.minor_flowpaths
        # All should be marked as processed
        assert "up1" in result.processed_flowpaths
        assert "up1a" in result.processed_flowpaths
        assert "up2" in result.processed_flowpaths
        assert "up3" in result.processed_flowpaths
        assert "fp1" in result.processed_flowpaths
        # Should be marked for subdivides
        assert "fp1" in result.subdivide_candidates
        assert "fp1" in result.upstream_merge_points

    def test_order2_with_four_order1s(self) -> None:
        """Test case: order-2 stream with 4 order-1 upstreams - on-mainstem stays non-minor, all others become minor"""
        result = Classifications()
        network_graph = {"fp1": ["up1", "up2", "up3", "up4"], "up1": [], "up2": [], "up3": [], "up4": []}
        fp_info = {"stream_order": 2, "flowpath_id": "fp1", "mainstemlp": 100}
        upstream_info = [
            {
                "stream_order": 1,
                "flowpath_id": "up1",
                "total_drainage_area_sqkm": 15.0,
                "mainstemlp": 100,
            },
            {
                "stream_order": 1,
                "flowpath_id": "up2",
                "total_drainage_area_sqkm": 8.0,
                "mainstemlp": 200,
            },
            {
                "stream_order": 1,
                "flowpath_id": "up3",
                "total_drainage_area_sqkm": 5.0,
                "mainstemlp": 300,
            },
            {
                "stream_order": 1,
                "flowpath_id": "up4",
                "total_drainage_area_sqkm": 2.0,
                "mainstemlp": 400,
            },
        ]

        matched = _rule_aggregate_order2_with_order1s("fp1", fp_info, upstream_info, network_graph, result)

        assert matched
        # All upstreams aggregated into fp1
        assert ("up1", "fp1") in result.aggregation_pairs
        assert ("up2", "fp1") in result.aggregation_pairs
        assert ("up3", "fp1") in result.aggregation_pairs
        assert ("up4", "fp1") in result.aggregation_pairs
        # Only up1 (on mainstem) should NOT be minor
        assert "up1" not in result.minor_flowpaths
        # All others should be minor (off mainstem)
        assert "up2" in result.minor_flowpaths
        assert "up3" in result.minor_flowpaths
        assert "up4" in result.minor_flowpaths
        assert len(result.minor_flowpaths) == 3


class TestRule6AggregateMixedUpstreamOrders:
    """Tests for Rule 6: Aggregate - Mixed Upstream Orders"""

    def test_mixed_orders_small_order1_becomes_minor(self, sample_config: HFConfig) -> None:
        """Test case: small order-1 upstream becomes minor flowpath aggregated to current"""
        result = Classifications()
        fp_info = {"stream_order": 3, "flowpath_id": "fp1", "mainstemlp": 100}
        upstream_info = [
            {
                "stream_order": 1,
                "flowpath_id": "up1",
                "areasqkm": 1.0,
                "mainstemlp": 200,
            },
            {
                "stream_order": 2,
                "flowpath_id": "up2",
                "areasqkm": 2.0,
                "mainstemlp": 100,
            },
        ]

        matched = _rule_aggregate_mixed_upstream_orders("fp1", fp_info, upstream_info, sample_config, result)

        assert matched
        assert "up1" in result.minor_flowpaths
        assert ("up1", "fp1") in result.aggregation_pairs
        assert "fp1" in result.upstream_merge_points

    def test_large_order1_not_minor(self, sample_config: HFConfig) -> None:
        """Test case: large order-1 (areasqkm >= threshold) is NOT marked as minor"""
        result = Classifications()
        fp_info = {"stream_order": 3, "flowpath_id": "fp1", "mainstemlp": 100}
        upstream_info = [
            {
                "stream_order": 1,
                "flowpath_id": "up1",
                "areasqkm": 5.0,
                "mainstemlp": 100,
            },
            {
                "stream_order": 2,
                "flowpath_id": "up2",
                "areasqkm": 2.0,
                "mainstemlp": 200,
            },
        ]

        matched = _rule_aggregate_mixed_upstream_orders("fp1", fp_info, upstream_info, sample_config, result)

        assert matched is False
        assert len(result.minor_flowpaths) == 0

    def test_multiple_small_order1s_all_minor(self, sample_config: HFConfig) -> None:
        """Test case: multiple small order-1s all become minor flowpaths"""
        result = Classifications()
        fp_info = {"stream_order": 3, "flowpath_id": "fp1", "mainstemlp": 100}
        upstream_info = [
            {
                "stream_order": 1,
                "flowpath_id": "up1",
                "areasqkm": 1.0,
                "mainstemlp": 200,
            },
            {
                "stream_order": 1,
                "flowpath_id": "up2",
                "areasqkm": 1.5,
                "mainstemlp": 300,
            },
            {
                "stream_order": 2,
                "flowpath_id": "up3",
                "areasqkm": 2.0,
                "mainstemlp": 100,
            },
        ]

        matched = _rule_aggregate_mixed_upstream_orders("fp1", fp_info, upstream_info, sample_config, result)

        assert matched
        assert "up1" in result.minor_flowpaths
        assert "up2" in result.minor_flowpaths
        assert len(result.minor_flowpaths) == 2

    def test_only_order1s_not_mixed(self, sample_config: HFConfig) -> None:
        """Test case: only order-1 upstreams do NOT trigger mixed rule"""
        result = Classifications()
        fp_info = {"stream_order": 2, "flowpath_id": "fp1", "mainstemlp": 100}
        upstream_info = [
            {
                "stream_order": 1,
                "flowpath_id": "up1",
                "areasqkm": 1.0,
                "mainstemlp": 200,
            },
            {
                "stream_order": 1,
                "flowpath_id": "up2",
                "areasqkm": 1.5,
                "mainstemlp": 300,
            },
        ]

        matched = _rule_aggregate_mixed_upstream_orders("fp1", fp_info, upstream_info, sample_config, result)

        assert matched is False

    def test_small_order1_with_large_order1(self, sample_config: HFConfig) -> None:
        """Test small order-1 with large order-1 upstream triggers rule."""
        result = Classifications()
        fp_info = {"stream_order": 2, "flowpath_id": "fp1", "mainstemlp": 100}
        upstream_info = [
            {
                "stream_order": 1,
                "flowpath_id": "up1",
                "areasqkm": 1.0,
                "mainstemlp": 200,
            },
            {
                "stream_order": 1,
                "flowpath_id": "up2",
                "areasqkm": 5.0,
                "mainstemlp": 100,
            },
        ]

        matched = _rule_aggregate_mixed_upstream_orders("fp1", fp_info, upstream_info, sample_config, result)

        assert matched
        assert "up1" in result.minor_flowpaths
        assert "up2" not in result.minor_flowpaths


class TestRule7AggregateSameOrderSmallArea:
    """Tests for Rule 7: Aggregate - Same Order with Small Area"""

    def test_same_order_small_area_aggregates(self, sample_config: HFConfig) -> None:
        """Test case: small area merges into same-order upstream."""
        result = Classifications()
        fp_info = {"stream_order": 2, "flowpath_id": "fp1", "areasqkm": 1.5}
        upstream_info = [
            {"stream_order": 1, "flowpath_id": "up1", "areasqkm": 1.0},
            {"stream_order": 2, "flowpath_id": "up2", "areasqkm": 2.0},
        ]

        matched = _rule_aggregate_same_order_small_area("fp1", fp_info, upstream_info, sample_config, result)

        assert matched
        assert "up1" in result.minor_flowpaths
        assert ("up1", "fp1") in result.aggregation_pairs
        assert ("fp1", "up2") in result.aggregation_pairs
        assert "fp1" in result.upstream_merge_points

    def test_cumulative_area_tracking(self, sample_config: HFConfig) -> None:
        """Test case: cumulative merge areas tracked correctly"""
        result = Classifications()
        result.cumulative_merge_areas["fp1"] = 3.0

        fp_info = {"stream_order": 2, "flowpath_id": "fp1", "areasqkm": 1.5}
        upstream_info = [{"stream_order": 1, "flowpath_id": "up1"}, {"stream_order": 2, "flowpath_id": "up2"}]

        matched = _rule_aggregate_same_order_small_area("fp1", fp_info, upstream_info, sample_config, result)

        assert matched
        assert result.cumulative_merge_areas["up2"] == 3.0

    def test_large_area_not_aggregated(self, sample_config: HFConfig) -> None:
        """Test case: large area (> threshold) does NOT trigger rule"""
        result = Classifications()
        fp_info = {
            "stream_order": 2,
            "flowpath_id": "fp1",
            "areasqkm": 5.0,  # Large
        }
        upstream_info = [{"stream_order": 1, "flowpath_id": "up1"}, {"stream_order": 2, "flowpath_id": "up2"}]

        matched = _rule_aggregate_same_order_small_area("fp1", fp_info, upstream_info, sample_config, result)

        assert matched is False

    def test_no_same_order_upstream(self, sample_config: HFConfig) -> None:
        """Test case: makes sure the upstream flowpaths trigger the rule if they are the same order"""
        result = Classifications()
        fp_info = {"stream_order": 2, "flowpath_id": "fp1", "areasqkm": 1.5}
        upstream_info = [
            {"stream_order": 1, "flowpath_id": "up1"},
            {"stream_order": 3, "flowpath_id": "up2"},  # Different order
        ]

        matched = _rule_aggregate_same_order_small_area("fp1", fp_info, upstream_info, sample_config, result)

        assert matched is False


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_aggregate_all_upstream_recursive(self) -> None:
        """Test recursive upstream aggregation."""
        result = Classifications()
        network_graph = {"fp1": ["up1", "up2"], "up1": ["up3"], "up2": [], "up3": []}

        _aggregate_all_upstream_recursive("fp1", "target", network_graph, result)

        assert ("up1", "target") in result.aggregation_pairs
        assert ("up2", "target") in result.aggregation_pairs
        assert ("up3", "target") in result.aggregation_pairs
        assert "up1" in result.processed_flowpaths
        assert "up2" in result.processed_flowpaths
        assert "up3" in result.processed_flowpaths

    def test_recursive_prevents_reprocessing(self) -> None:
        """Test that recursive aggregation doesn't reprocess flowpaths."""
        result = Classifications()
        result.processed_flowpaths.add("up2")  # Already processed

        network_graph = {"fp1": ["up1", "up2"], "up1": [], "up2": []}

        _aggregate_all_upstream_recursive("fp1", "target", network_graph, result)

        assert ("up1", "target") in result.aggregation_pairs
        # up2 should not be in aggregation_pairs since it was already processed
        assert ("up2", "target") not in result.aggregation_pairs

    def test_get_flowpath_info(self, sample_flowpath_data: pd.DataFrame) -> None:
        """Test getting flowpath info from indexed dataframe."""
        fp_indexed = sample_flowpath_data.set_index("flowpath_id")

        info = _get_flowpath_info("1", fp_indexed)

        assert info["flowpath_id"] == "1"
        assert info["total_drainage_area_sqkm"] == 100.0
        assert info["areasqkm"] == 5.0
        assert info["length_km"] == 10.0
        assert info["stream_order"] == 3
        assert info["mainstemlp"] == 100


class TestTraceAndAggregateUpstream:
    """Tests for the _trace_and_aggregate_upstream helper function"""

    def test_trace_single_upstream(self) -> None:
        """Test tracing a single upstream flowpath"""
        result = Classifications()
        network_graph = {"start": ["up1"], "up1": []}

        _trace_and_aggregate_upstream("start", "target", network_graph, result)

        assert ("start", "target") in result.aggregation_pairs
        assert "start" in result.processed_flowpaths

    def test_trace_multiple_direct_upstream(self) -> None:
        """Test tracing multiple direct upstream flowpaths"""
        result = Classifications()
        network_graph = {"start": ["up1", "up2"], "up1": [], "up2": []}

        _trace_and_aggregate_upstream("start", "target", network_graph, result)

        assert ("start", "target") in result.aggregation_pairs
        assert ("up1", "target") in result.aggregation_pairs
        assert ("up2", "target") in result.aggregation_pairs
        assert "start" in result.processed_flowpaths
        assert "up1" in result.processed_flowpaths
        assert "up2" in result.processed_flowpaths

    def test_trace_recursive_upstream_chain(self) -> None:
        """Test tracing a deep upstream chain"""
        result = Classifications()
        network_graph = {"start": ["up1"], "up1": ["up2"], "up2": ["up3"], "up3": []}

        _trace_and_aggregate_upstream("start", "target", network_graph, result)

        assert ("start", "target") in result.aggregation_pairs
        assert ("up1", "target") in result.aggregation_pairs
        assert ("up2", "target") in result.aggregation_pairs
        assert ("up3", "target") in result.aggregation_pairs
        assert len(result.aggregation_pairs) == 4

    def test_trace_complex_network(self) -> None:
        """Test tracing a complex branching network"""
        result = Classifications()
        network_graph = {
            "start": ["up1", "up2"],
            "up1": ["up1a", "up1b"],
            "up2": ["up2a"],
            "up1a": [],
            "up1b": [],
            "up2a": [],
        }

        _trace_and_aggregate_upstream("start", "target", network_graph, result)

        # All flowpaths should be aggregated into target
        assert ("start", "target") in result.aggregation_pairs
        assert ("up1", "target") in result.aggregation_pairs
        assert ("up2", "target") in result.aggregation_pairs
        assert ("up1a", "target") in result.aggregation_pairs
        assert ("up1b", "target") in result.aggregation_pairs
        assert ("up2a", "target") in result.aggregation_pairs
        assert len(result.aggregation_pairs) == 6
        # All should be marked as processed
        assert len(result.processed_flowpaths) == 6

    def test_trace_with_is_minor_flag(self) -> None:
        """Test that is_minor flag marks flowpaths as minor"""
        result = Classifications()
        network_graph = {"start": ["up1"], "up1": ["up2"], "up2": []}

        _trace_and_aggregate_upstream("start", "target", network_graph, result, is_minor=True)

        # All should be in minor_flowpaths
        assert "start" in result.minor_flowpaths
        assert "up1" in result.minor_flowpaths
        assert "up2" in result.minor_flowpaths
        # And also in aggregation_pairs
        assert ("start", "target") in result.aggregation_pairs
        assert ("up1", "target") in result.aggregation_pairs
        assert ("up2", "target") in result.aggregation_pairs

    def test_trace_without_is_minor_flag(self) -> None:
        """Test that flowpaths are NOT marked minor when is_minor=False"""
        result = Classifications()
        network_graph = {"start": ["up1"], "up1": [], "up2": []}

        _trace_and_aggregate_upstream("start", "target", network_graph, result, is_minor=False)

        # Should NOT be in minor_flowpaths
        assert "start" not in result.minor_flowpaths
        assert "up1" not in result.minor_flowpaths
        # But should be aggregated
        assert ("start", "target") in result.aggregation_pairs
        assert ("up1", "target") in result.aggregation_pairs

    def test_trace_when_current_equals_target(self) -> None:
        """Test that when current_id equals target_id, it's not added to aggregation_pairs"""
        result = Classifications()
        network_graph = {"target": ["up1"], "up1": []}

        _trace_and_aggregate_upstream("target", "target", network_graph, result)

        # target should NOT aggregate into itself
        assert ("target", "target") not in result.aggregation_pairs
        # But upstream should aggregate into target
        assert ("up1", "target") in result.aggregation_pairs
        # target should still be marked as processed
        assert "target" in result.processed_flowpaths

    def test_trace_empty_network(self) -> None:
        """Test tracing a flowpath with no upstream"""
        result = Classifications()
        network_graph: dict = {"start": []}

        _trace_and_aggregate_upstream("start", "target", network_graph, result)

        # Only start should be aggregated (no upstream to trace)
        assert ("start", "target") in result.aggregation_pairs
        assert len(result.aggregation_pairs) == 1
        assert "start" in result.processed_flowpaths

    def test_trace_preserves_existing_classifications(self) -> None:
        """Test that tracing doesn't overwrite existing classifications"""
        result = Classifications()
        result.aggregation_pairs.append(("existing", "previous"))
        result.processed_flowpaths.add("existing")
        network_graph = {"start": ["up1"], "up1": []}

        _trace_and_aggregate_upstream("start", "target", network_graph, result)

        # New aggregations added
        assert ("start", "target") in result.aggregation_pairs
        assert ("up1", "target") in result.aggregation_pairs
        # Old aggregations preserved
        assert ("existing", "previous") in result.aggregation_pairs
        assert "existing" in result.processed_flowpaths
