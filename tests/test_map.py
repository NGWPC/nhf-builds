"""Tests for hydrofabric building functions."""

from typing import Any

import geopandas as gpd
import polars as pl
from conftest import create_partition_data_for_build_tests, dict_to_graph
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

from hydrofabric_builds import HFConfig
from hydrofabric_builds.hydrofabric.aggregate import (
    _aggregate_geometries,
)
from hydrofabric_builds.hydrofabric.build import (
    _build_base_hydrofabric,
    _order_aggregates_base,
)
from hydrofabric_builds.hydrofabric.graph import _create_dictionary_lookup
from hydrofabric_builds.schemas.hydrofabric import Aggregations, Classifications


class TestOrderAggregatesBase:
    """Tests for _order_aggregates_base function."""

    def test_orders_all_aggregate_types(self, sample_aggregate_data: Aggregations) -> None:
        """Test that all aggregate types are included in output.

        Parameters
        ----------
        sample_aggregate_data : Aggregations
            Sample aggregation data fixture
        """
        result = _order_aggregates_base(sample_aggregate_data)

        # Check aggregates from multiple chains
        assert "6720877" in result  # Aggregate chain 1
        assert "6720879" in result
        assert "6720883" in result

        assert "6720683" in result  # Aggregate chain 2
        assert "6720703" in result
        assert "6720651" in result

        # Check independents
        assert "6720797" in result
        assert "6720795" in result
        assert "6720773" in result

    def test_aggregate_structure(self, sample_aggregate_data: Aggregations) -> None:
        """Test that aggregate entries have correct structure.

        Parameters
        ----------
        sample_aggregate_data : Aggregations
            Sample aggregation data fixture
        """
        result = _order_aggregates_base(sample_aggregate_data)

        # First aggregate chain: 6720651 -> 6720703 -> 6720683
        # All three IDs should map to entries with all_ref_ids containing all three
        agg_6720683 = result["6720683"]
        agg_6720703 = result["6720703"]
        agg_6720651 = result["6720651"]

        # All should be aggregates
        assert agg_6720683["type"] == "aggregate"
        assert agg_6720703["type"] == "aggregate"
        assert agg_6720651["type"] == "aggregate"

        # Check one has the correct structure
        assert "unit" in agg_6720683
        assert agg_6720703["up_id"] == "6720683"
        assert agg_6720703["dn_id"] == "6720703"

        # All three entries should have the complete list of ref_ids from the aggregate
        assert set(agg_6720683["all_ref_ids"]) == {"6720683", "6720703", "6720651"}
        assert set(agg_6720703["all_ref_ids"]) == {"6720683", "6720703", "6720651"}
        assert set(agg_6720651["all_ref_ids"]) == {"6720683", "6720703", "6720651"}

    def test_independent_structure(self, sample_aggregate_data: Aggregations) -> None:
        """Test that independent entries have correct structure.

        Parameters
        ----------
        sample_aggregate_data : Aggregations
            Sample aggregation data fixture
        """
        result = _order_aggregates_base(sample_aggregate_data)

        ind = result["6720797"]
        assert ind["type"] == "independent"
        assert "unit" in ind
        assert ind["all_ref_ids"] == ["6720797"]
        assert "up_id" not in ind  # independents don't have up/dn_id
        assert "dn_id" not in ind

    def test_virtual_flowpath_structure(self, sample_aggregate_data: Aggregations) -> None:
        """Test that virtual flowpath entries exist in aggregates.

        Parameters
        ----------
        sample_aggregate_data : Aggregations
            Sample aggregation data fixture
        """
        result = _order_aggregates_base(sample_aggregate_data)

        # virtual flowpaths that are also in aggregates should appear
        # 6720681 is in both virtual_flowpaths and aggregate 4
        assert "6720681" in result
        assert result["6720681"]["type"] == "aggregate"

        # 6720651 is in both virtual_flowpaths and aggregate 2
        assert "6720651" in result
        assert result["6720651"]["type"] == "aggregate"

        # 6720883 is in both virtual_flowpaths and aggregate 1
        assert "6720883" in result
        assert result["6720883"]["type"] == "aggregate"

        # 6720517 is in both virtual_flowpaths and aggregate 5
        assert "6720517" in result
        assert result["6720517"]["type"] == "aggregate"

    def test_preserves_geometries(self, sample_aggregate_data: Aggregations) -> None:
        """Test that geometries are preserved in units.

        Parameters
        ----------
        sample_aggregate_data : Aggregations
            Sample aggregation data fixture
        """
        result = _order_aggregates_base(sample_aggregate_data)

        # Check aggregate geometry preservation
        agg_unit = result["6720703"]["unit"]
        assert "line_geometry" in agg_unit
        assert "polygon_geometry" in agg_unit
        assert isinstance(agg_unit["line_geometry"], LineString)
        assert isinstance(agg_unit["polygon_geometry"], Polygon)

    def test_handles_empty_aggregations(self) -> None:
        """Test handling of empty aggregation data."""
        empty_agg = Aggregations(
            aggregates=[],
            independents=[],
            connectors=[],
            virtual_flowpaths=[],
            small_scale_connectors=[],
        )

        result = _order_aggregates_base(empty_agg)
        assert result == {}

    def test_handles_only_aggregates(self) -> None:
        """Test with only aggregates, no independents or connectors."""
        only_agg = Aggregations(
            aggregates=[
                {
                    "ref_ids": ["fp1", "fp2"],
                    "dn_id": "fp1",
                    "up_id": "fp2",
                    "vpu_id": "01",
                    "hydroseq": 1,
                    "area_sqkm": 3.0,
                    "length_km": 3.0,
                    "div_area_sqkm": 3.0,
                    "line_geometry": LineString([(0, 0), (1, 1)]),
                    "polygon_geometry": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                }
            ],
            independents=[],
            connectors=[],
            virtual_flowpaths=[],
            small_scale_connectors=[],
        )

        result = _order_aggregates_base(only_agg)
        # ref_ids creates one entry for EACH ref_id, so 2 entries total
        assert len(result) == 2
        assert "fp1" in result
        assert "fp2" in result
        # Both point to the same aggregate type
        assert result["fp1"]["type"] == "aggregate"
        assert result["fp2"]["type"] == "aggregate"
        # Both have the complete ref_ids list
        assert set(result["fp1"]["all_ref_ids"]) == {"fp1", "fp2"}
        assert set(result["fp2"]["all_ref_ids"]) == {"fp1", "fp2"}

    def test_uses_correct_keys_for_each_type(self, sample_aggregate_data: Aggregations) -> None:
        """Test that correct keys are used for different aggregate types.

        Parameters
        ----------
        sample_aggregate_data : Aggregations
            Sample aggregation data fixture
        """
        result = _order_aggregates_base(sample_aggregate_data)

        # Aggregates - each ref_id becomes a key
        assert "6720683" in result
        assert "6720703" in result
        assert "6720651" in result
        assert result["6720703"]["dn_id"] == "6720703"

        # Independents use ref_ids as key
        assert "6720797" in result
        assert result["6720797"]["type"] == "independent"


class TestBuildBaseHydrofabric:
    """Tests for _build_base_hydrofabric function."""

    def test_builds_all_layers(
        self,
        sample_aggregate_data: Aggregations,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
        expected_graph: dict[str, list[str]],
        sample_config: HFConfig,
    ) -> None:
        """Test that all hydrofabric layers are created.

        Parameters
        ----------
        sample_aggregate_data : Aggregations
            Sample aggregation data fixture
        sample_classifications : Classifications
            Sample classifications fixture
        sample_reference_data : tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
            Reference flowpaths and divides
        expected_graph : dict[str, list[str]]
            Expected network graph
        sample_config : HFConfig
            Sample configuration
        """
        reference_flowpaths, reference_divides = sample_reference_data

        graph, node_indices = dict_to_graph(expected_graph)

        partition_data = create_partition_data_for_build_tests(
            reference_flowpaths, reference_divides, graph, node_indices
        )

        result = _build_base_hydrofabric(
            start_id="6720797",
            aggregate_data=sample_aggregate_data,
            classifications=sample_classifications,
            partition_data=partition_data,
            cfg=sample_config,
        )

        assert "flowpaths" in result
        assert "divides" in result
        assert "nexus" in result
        assert "base_virtual_flowpaths" in result

    def test_creates_geodataframes(
        self,
        sample_aggregate_data: Aggregations,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
        expected_graph: dict[str, list[str]],
        sample_config: HFConfig,
    ) -> None:
        """Test that outputs are GeoDataFrames.

        Parameters
        ----------
        sample_aggregate_data : Aggregations
            Sample aggregation data fixture
        sample_classifications : Classifications
            Sample classifications fixture
        sample_reference_data : tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
            Reference flowpaths and divides
        expected_graph : dict[str, list[str]]
            Expected network graph
        sample_config : HFConfig
            Sample configuration
        """
        reference_flowpaths, reference_divides = sample_reference_data
        graph, node_indices = dict_to_graph(expected_graph)

        partition_data = create_partition_data_for_build_tests(
            reference_flowpaths, reference_divides, graph, node_indices
        )

        result = _build_base_hydrofabric(
            start_id="6720797",
            aggregate_data=sample_aggregate_data,
            classifications=sample_classifications,
            partition_data=partition_data,
            cfg=sample_config,
        )

        assert isinstance(result["flowpaths"], gpd.GeoDataFrame)
        assert isinstance(result["divides"], gpd.GeoDataFrame)
        assert isinstance(result["nexus"], gpd.GeoDataFrame)
        assert isinstance(result["base_virtual_flowpaths"], list)

    def test_assigns_unique_ids(
        self,
        sample_aggregate_data: Aggregations,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
        expected_graph: dict[str, list[str]],
        sample_config: HFConfig,
    ) -> None:
        """Test that IDs are unique.

        Parameters
        ----------
        sample_aggregate_data : Aggregations
            Sample aggregation data fixture
        sample_classifications : Classifications
            Sample classifications fixture
        sample_reference_data : tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
            Reference flowpaths and divides
        expected_graph : dict[str, list[str]]
            Expected network graph
        sample_config : HFConfig
            Sample configuration
        """
        reference_flowpaths, reference_divides = sample_reference_data
        graph, node_indices = dict_to_graph(expected_graph)

        partition_data = create_partition_data_for_build_tests(
            reference_flowpaths, reference_divides, graph, node_indices
        )

        result = _build_base_hydrofabric(
            start_id="6720797",
            aggregate_data=sample_aggregate_data,
            classifications=sample_classifications,
            partition_data=partition_data,
            cfg=sample_config,
        )

        fp_ids: list[Any] = result["flowpaths"]["fp_id"].tolist()  # type: ignore
        div_ids: list[Any] = result["divides"]["div_id"].tolist()  # type: ignore
        nex_ids: list[Any] = result["nexus"]["nex_id"].tolist()  # type: ignore

        # All IDs should be unique within each layer
        assert len(fp_ids) == len(set(fp_ids))
        assert len(div_ids) == len(set(div_ids))
        assert len(nex_ids) == len(set(nex_ids))

    def test_ids_start_at_one_by_default(
        self,
        sample_aggregate_data: Aggregations,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
        expected_graph: dict[str, list[str]],
        sample_config: HFConfig,
    ) -> None:
        """Test that IDs start at 1 by default.

        Parameters
        ----------
        sample_aggregate_data : Aggregations
            Sample aggregation data fixture
        sample_classifications : Classifications
            Sample classifications fixture
        sample_reference_data : tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
            Reference flowpaths and divides
        expected_graph : dict[str, list[str]]
            Expected network graph
        sample_config : HFConfig
            Sample configuration
        """
        reference_flowpaths, reference_divides = sample_reference_data
        graph, node_indices = dict_to_graph(expected_graph)

        partition_data = create_partition_data_for_build_tests(
            reference_flowpaths, reference_divides, graph, node_indices
        )

        result = _build_base_hydrofabric(
            start_id="6720797",
            aggregate_data=sample_aggregate_data,
            classifications=sample_classifications,
            partition_data=partition_data,
            cfg=sample_config,
        )

        fp_ids: list[Any] = result["flowpaths"]["fp_id"].tolist()  # type: ignore
        assert min(fp_ids) == 1

    def test_respects_id_offset(
        self,
        sample_aggregate_data: Aggregations,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
        expected_graph: dict[str, list[str]],
        sample_config: HFConfig,
    ) -> None:
        """Test that id_offset parameter works correctly.

        Parameters
        ----------
        sample_aggregate_data : Aggregations
            Sample aggregation data fixture
        sample_classifications : Classifications
            Sample classifications fixture
        sample_reference_data : tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
            Reference flowpaths and divides
        expected_graph : dict[str, list[str]]
            Expected network graph
        sample_config : HFConfig
            Sample configuration
        """
        reference_flowpaths, reference_divides = sample_reference_data
        offset = 100
        graph, node_indices = dict_to_graph(expected_graph)

        partition_data = create_partition_data_for_build_tests(
            reference_flowpaths, reference_divides, graph, node_indices
        )

        result = _build_base_hydrofabric(
            start_id="6720797",
            aggregate_data=sample_aggregate_data,
            classifications=sample_classifications,
            partition_data=partition_data,
            cfg=sample_config,
            id_offset=offset,
        )

        fp_ids: list[Any] = result["flowpaths"]["fp_id"].tolist()  # type: ignore
        assert min(fp_ids) == offset + 1

    def test_flowpaths_have_required_columns(
        self,
        sample_aggregate_data: Aggregations,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
        expected_graph: dict[str, list[str]],
        sample_config: HFConfig,
    ) -> None:
        """Test that flowpaths have all required columns.

        Parameters
        ----------
        sample_aggregate_data : Aggregations
            Sample aggregation data fixture
        sample_classifications : Classifications
            Sample classifications fixture
        sample_reference_data : tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
            Reference flowpaths and divides
        expected_graph : dict[str, list[str]]
            Expected network graph
        sample_config : HFConfig
            Sample configuration
        """
        reference_flowpaths, reference_divides = sample_reference_data
        graph, node_indices = dict_to_graph(expected_graph)

        partition_data = create_partition_data_for_build_tests(
            reference_flowpaths, reference_divides, graph, node_indices
        )

        result = _build_base_hydrofabric(
            start_id="6720797",
            aggregate_data=sample_aggregate_data,
            classifications=sample_classifications,
            partition_data=partition_data,
            cfg=sample_config,
        )

        required_cols = ["fp_id", "dn_nex_id", "up_nex_id", "div_id", "geometry"]
        flowpaths: gpd.GeoDataFrame = result["flowpaths"]
        for col in required_cols:
            assert col in flowpaths.columns

    def test_divides_have_required_columns(
        self,
        sample_aggregate_data: Aggregations,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
        expected_graph: dict[str, list[str]],
        sample_config: HFConfig,
    ) -> None:
        """Test that divides have all required columns.

        Parameters
        ----------
        sample_aggregate_data : Aggregations
            Sample aggregation data fixture
        sample_classifications : Classifications
            Sample classifications fixture
        sample_reference_data : tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
            Reference flowpaths and divides
        expected_graph : dict[str, list[str]]
            Expected network graph
        sample_config : HFConfig
            Sample configuration
        """
        reference_flowpaths, reference_divides = sample_reference_data
        graph, node_indices = dict_to_graph(expected_graph)

        partition_data = create_partition_data_for_build_tests(
            reference_flowpaths, reference_divides, graph, node_indices
        )

        result = _build_base_hydrofabric(
            start_id="6720797",
            aggregate_data=sample_aggregate_data,
            classifications=sample_classifications,
            partition_data=partition_data,
            cfg=sample_config,
        )

        required_cols = ["div_id", "type", "geometry"]
        divides: gpd.GeoDataFrame = result["divides"]
        for col in required_cols:
            assert col in divides.columns

    def test_nexus_have_required_columns(
        self,
        sample_aggregate_data: Aggregations,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
        expected_graph: dict[str, list[str]],
        sample_config: HFConfig,
    ) -> None:
        """Test that nexus have all required columns.

        Parameters
        ----------
        sample_aggregate_data : Aggregations
            Sample aggregation data fixture
        sample_classifications : Classifications
            Sample classifications fixture
        sample_reference_data : tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
            Reference flowpaths and divides
        expected_graph : dict[str, list[str]]
            Expected network graph
        sample_config : HFConfig
            Sample configuration
        """
        reference_flowpaths, reference_divides = sample_reference_data
        graph, node_indices = dict_to_graph(expected_graph)

        partition_data = create_partition_data_for_build_tests(
            reference_flowpaths, reference_divides, graph, node_indices
        )

        result = _build_base_hydrofabric(
            start_id="6720797",
            aggregate_data=sample_aggregate_data,
            classifications=sample_classifications,
            partition_data=partition_data,
            cfg=sample_config,
        )

        required_cols = ["nex_id", "dn_fp_id", "geometry"]
        nexus: gpd.GeoDataFrame = result["nexus"]
        for col in required_cols:
            assert col in nexus.columns

    def test_fp_div_relationship(
        self,
        sample_aggregate_data: Aggregations,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
        expected_graph: dict[str, list[str]],
        sample_config: HFConfig,
    ) -> None:
        """Test that flowpaths and divides have matching IDs.

        Parameters
        ----------
        sample_aggregate_data : Aggregations
            Sample aggregation data fixture
        sample_classifications : Classifications
            Sample classifications fixture
        sample_reference_data : tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
            Reference flowpaths and divides
        expected_graph : dict[str, list[str]]
            Expected network graph
        sample_config : HFConfig
            Sample configuration
        """
        reference_flowpaths, reference_divides = sample_reference_data
        graph, node_indices = dict_to_graph(expected_graph)

        partition_data = create_partition_data_for_build_tests(
            reference_flowpaths, reference_divides, graph, node_indices
        )

        result = _build_base_hydrofabric(
            start_id="6720797",
            aggregate_data=sample_aggregate_data,
            classifications=sample_classifications,
            partition_data=partition_data,
            cfg=sample_config,
        )

        flowpaths: gpd.GeoDataFrame = result["flowpaths"]
        divides: gpd.GeoDataFrame = result["divides"]

        # Each flowpath should reference a divide with matching ID
        for _, fp in flowpaths.iterrows():
            div_id = fp["div_id"]
            assert div_id in divides["div_id"].values

    def test_nexus_connectivity(
        self,
        sample_aggregate_data: Aggregations,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
        expected_graph: dict[str, list[str]],
        sample_config: HFConfig,
    ) -> None:
        """Test that nexus points properly connect flowpaths.

        Parameters
        ----------
        sample_aggregate_data : Aggregations
            Sample aggregation data fixture
        sample_classifications : Classifications
            Sample classifications fixture
        sample_reference_data : tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
            Reference flowpaths and divides
        expected_graph : dict[str, list[str]]
            Expected network graph
        sample_config : HFConfig
            Sample configuration
        """
        reference_flowpaths, reference_divides = sample_reference_data
        graph, node_indices = dict_to_graph(expected_graph)

        partition_data = create_partition_data_for_build_tests(
            reference_flowpaths, reference_divides, graph, node_indices
        )

        result = _build_base_hydrofabric(
            start_id="6720797",
            aggregate_data=sample_aggregate_data,
            classifications=sample_classifications,
            partition_data=partition_data,
            cfg=sample_config,
        )

        flowpaths: gpd.GeoDataFrame = result["flowpaths"]
        nexus: gpd.GeoDataFrame = result["nexus"]

        # All nexus IDs referenced in flowpaths should exist
        all_nex_refs = set(flowpaths["dn_nex_id"].unique())
        all_nex_refs.update(flowpaths["up_nex_id"].dropna().unique())
        all_nex_ids = set(nexus["nex_id"].unique())

        assert all_nex_refs.issubset(all_nex_ids)

    def test_divide_types(
        self,
        sample_aggregate_data: Aggregations,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
        expected_graph: dict[str, list[str]],
        sample_config: HFConfig,
    ) -> None:
        """Test that divide types are assigned correctly.

        Parameters
        ----------
        sample_aggregate_data : Aggregations
            Sample aggregation data fixture
        sample_classifications : Classifications
            Sample classifications fixture
        sample_reference_data : tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
            Reference flowpaths and divides
        expected_graph : dict[str, list[str]]
            Expected network graph
        sample_config : HFConfig
            Sample configuration
        """
        reference_flowpaths, reference_divides = sample_reference_data
        graph, node_indices = dict_to_graph(expected_graph)

        partition_data = create_partition_data_for_build_tests(
            reference_flowpaths, reference_divides, graph, node_indices
        )

        result = _build_base_hydrofabric(
            start_id="6720797",
            aggregate_data=sample_aggregate_data,
            classifications=sample_classifications,
            partition_data=partition_data,
            cfg=sample_config,
        )

        divides: gpd.GeoDataFrame = result["divides"]
        types = divides["type"].unique()

        # Should have types from our sample data
        valid_types = {"aggregate", "independent", "connectors"}
        assert all(t in valid_types for t in types)

    def test_geometry_types(
        self,
        sample_aggregate_data: Aggregations,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
        expected_graph: dict[str, list[str]],
        sample_config: HFConfig,
    ) -> None:
        """Test that geometry types are correct for each layer.

        Parameters
        ----------
        sample_aggregate_data : Aggregations
            Sample aggregation data fixture
        sample_classifications : Classifications
            Sample classifications fixture
        sample_reference_data : tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
            Reference flowpaths and divides
        expected_graph : dict[str, list[str]]
            Expected network graph
        sample_config : HFConfig
            Sample configuration
        """
        reference_flowpaths, reference_divides = sample_reference_data
        graph, node_indices = dict_to_graph(expected_graph)

        partition_data = create_partition_data_for_build_tests(
            reference_flowpaths, reference_divides, graph, node_indices
        )

        result = _build_base_hydrofabric(
            start_id="6720797",
            aggregate_data=sample_aggregate_data,
            classifications=sample_classifications,
            partition_data=partition_data,
            cfg=sample_config,
        )

        flowpaths: gpd.GeoDataFrame = result["flowpaths"]
        divides: gpd.GeoDataFrame = result["divides"]
        nexus: gpd.GeoDataFrame = result["nexus"]

        # Flowpaths should be LineStrings or MultiLineStrings
        for geom in flowpaths.geometry:
            assert geom.geom_type in ["LineString", "MultiLineString"]

        # Divides should be Polygons or MultiPolygons
        for geom in divides.geometry:
            assert geom.geom_type in ["Polygon", "MultiPolygon"]

        # Nexus should be Points
        for geom in nexus.geometry:
            assert geom.geom_type == "Point"


class TestAggregateGeometries:
    """Tests for _aggregate_geometries function."""

    def test_creates_aggregations_object(
        self,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
    ) -> None:
        """Test that function returns Aggregations object.

        Parameters
        ----------
        sample_classifications : Classifications
            Sample classifications fixture
        sample_reference_data : tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
            Reference flowpaths and divides
        """
        reference_flowpaths, reference_divides = sample_reference_data
        reference_flowpaths = pl.from_pandas(reference_flowpaths.to_wkb())
        reference_divides = pl.from_pandas(reference_divides.to_wkb())

        # Prepare lookup dictionaries
        fp_lookup = _create_dictionary_lookup(reference_flowpaths, "flowpath_id")
        div_lookup = _create_dictionary_lookup(reference_divides, "divide_id")

        result = _aggregate_geometries(
            classifications=sample_classifications,
            partition_data={
                "flowpaths": reference_flowpaths,
                "fp_lookup": fp_lookup,
                "div_lookup": div_lookup,
            },
        )

        assert isinstance(result, Aggregations)

    def test_aggregates_have_geometries(
        self,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
    ) -> None:
        """Test that aggregated units have line and polygon geometries.

        Parameters
        ----------
        sample_classifications : Classifications
            Sample classifications fixture
        sample_reference_data : tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
            Reference flowpaths and divides
        """
        reference_flowpaths, reference_divides = sample_reference_data
        reference_flowpaths = pl.from_pandas(reference_flowpaths.to_wkb())
        reference_divides = pl.from_pandas(reference_divides.to_wkb())

        # Prepare lookup dictionaries
        fp_lookup = _create_dictionary_lookup(reference_flowpaths, "flowpath_id")
        div_lookup = _create_dictionary_lookup(reference_divides, "divide_id")

        result = _aggregate_geometries(
            classifications=sample_classifications,
            partition_data={
                "flowpaths": reference_flowpaths,
                "fp_lookup": fp_lookup,
                "div_lookup": div_lookup,
            },
        )

        if len(result.aggregates) > 0:
            agg = result.aggregates[0]
            assert "line_geometry" in agg
            assert "polygon_geometry" in agg
            # Geometries might be None if not found, so check type only if present
            if agg["line_geometry"] is not None:
                assert isinstance(agg["line_geometry"], LineString | MultiLineString)
            if agg["polygon_geometry"] is not None:
                assert isinstance(agg["polygon_geometry"], Polygon | MultiPolygon)

    def test_independents_have_geometries(
        self,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
    ) -> None:
        """Test that independent units have geometries.

        Parameters
        ----------
        sample_classifications : Classifications
            Sample classifications fixture
        sample_reference_data : tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
            Reference flowpaths and divides
        """
        reference_flowpaths, reference_divides = sample_reference_data
        reference_flowpaths = pl.from_pandas(reference_flowpaths.to_wkb())
        reference_divides = pl.from_pandas(reference_divides.to_wkb())

        # Prepare lookup dictionaries
        fp_lookup = _create_dictionary_lookup(reference_flowpaths, "flowpath_id")
        div_lookup = _create_dictionary_lookup(reference_divides, "divide_id")

        result = _aggregate_geometries(
            classifications=sample_classifications,
            partition_data={
                "flowpaths": reference_flowpaths,
                "fp_lookup": fp_lookup,
                "div_lookup": div_lookup,
            },
        )

        if len(result.independents) > 0:
            ind = result.independents[0]
            assert "line_geometry" in ind
            assert "polygon_geometry" in ind

    def test_processes_aggregation_pairs(
        self,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
    ) -> None:
        """Test that aggregation pairs are processed.

        Parameters
        ----------
        sample_classifications : Classifications
            Sample classifications fixture
        sample_reference_data : tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
            Reference flowpaths and divides
        """
        reference_flowpaths, reference_divides = sample_reference_data
        reference_flowpaths = pl.from_pandas(reference_flowpaths.to_wkb())
        reference_divides = pl.from_pandas(reference_divides.to_wkb())

        # Ensure we have aggregation pairs
        assert len(sample_classifications.aggregation_pairs) > 0

        # Prepare lookup dictionaries
        fp_lookup = _create_dictionary_lookup(reference_flowpaths, "flowpath_id")
        div_lookup = _create_dictionary_lookup(reference_divides, "divide_id")

        result = _aggregate_geometries(
            classifications=sample_classifications,
            partition_data={
                "flowpaths": reference_flowpaths,
                "fp_lookup": fp_lookup,
                "div_lookup": div_lookup,
            },
        )

        # Should have created aggregates
        assert len(result.aggregates) > 0

    def test_processes_independents(
        self,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
    ) -> None:
        """Test that independent flowpaths are processed.

        Parameters
        ----------
        sample_classifications : Classifications
            Sample classifications fixture
        sample_reference_data : tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
            Reference flowpaths and divides
        """
        reference_flowpaths, reference_divides = sample_reference_data
        reference_flowpaths = pl.from_pandas(reference_flowpaths.to_wkb())
        reference_divides = pl.from_pandas(reference_divides.to_wkb())

        # Ensure we have independents
        assert len(sample_classifications.independent_flowpaths) > 0

        # Prepare lookup dictionaries
        fp_lookup = _create_dictionary_lookup(reference_flowpaths, "flowpath_id")
        div_lookup = _create_dictionary_lookup(reference_divides, "divide_id")

        result = _aggregate_geometries(
            classifications=sample_classifications,
            partition_data={
                "flowpaths": reference_flowpaths,
                "fp_lookup": fp_lookup,
                "div_lookup": div_lookup,
            },
        )

        # Should have created independents
        assert len(result.independents) > 0

    # TODO: Commenting back out when connectors are included in sample data
    # def test_processes_connectors(
    #     self,
    #     sample_classifications: Classifications,
    #     sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
    # ) -> None:
    #     """Test that connector segments are processed.

    #     Parameters
    #     ----------
    #     sample_classifications : Classifications
    #         Sample classifications fixture
    #     sample_reference_data : tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
    #         Reference flowpaths and divides
    #     """
    #     reference_flowpaths, reference_divides = sample_reference_data
    # reference_flowpaths = pl.from_pandas(reference_flowpaths.to_wkb())
    # reference_divides = pl.from_pandas(reference_divides.to_wkb())

    #     # Ensure we have connectors
    #     assert len(sample_classifications.connector_segments) > 0

    #     # Prepare lookup dictionaries
    #     fp_geom_lookup, div_geom_lookup = _prepare_dataframes(reference_flowpaths, reference_divides)

    #     result = _aggregate_geometries(
    #         classifications=sample_classifications,
    #         reference_flowpaths=pl.from_pandas(reference_flowpaths.to_wkb()),
    #         fp_geom_lookup=fp_geom_lookup,
    #         div_geom_lookup=div_geom_lookup,
    #     )

    #     # Should have created connectors
    #     assert len(result.connectors) > 0

    def test_handles_empty_classifications(
        self, sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
    ) -> None:
        """Test handling of empty classifications.

        Parameters
        ----------
        sample_reference_data : tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
            Reference flowpaths and divides
        """
        reference_flowpaths, reference_divides = sample_reference_data
        reference_flowpaths = pl.from_pandas(reference_flowpaths.to_wkb())
        reference_divides = pl.from_pandas(reference_divides.to_wkb())

        empty_classifications = Classifications(
            aggregation_pairs=[],
            virtual_flowpaths=set(),
            independent_flowpaths=set(),
            connector_segments=[],
            subdivide_candidates=[],
            upstream_merge_points=[],
            processed_flowpaths=set(),
            cumulative_merge_areas={},
        )

        # Prepare lookup dictionaries
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

        # Should return empty lists
        assert len(result.aggregates) == 0
        assert len(result.independents) == 0
        assert len(result.connectors) == 0
