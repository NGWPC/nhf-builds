"""Tests for hydrofabric building functions"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

from hydrofabric_builds import HFConfig
from hydrofabric_builds.hydrofabric.aggregate import _aggregate_geometries
from hydrofabric_builds.hydrofabric.build import _build_base_hydrofabric, _order_aggregates_base
from hydrofabric_builds.schemas.hydrofabric import Aggregations, Classifications


class TestOrderAggregatesBase:
    """Tests for _order_aggregates_base function"""

    def test_orders_all_aggregate_types(self, sample_aggregate_data: Aggregations) -> None:
        """Test that all aggregate types are included in output."""
        result = _order_aggregates_base(sample_aggregate_data)

        assert "6720797" in result  # aggregate (dn_id)
        assert "6720703" in result  # aggregate (dn_id)
        assert "6720651" in result  # independent (ref_ids)
        assert "6720681" in result  # connector (ref_ids)

    def test_aggregate_structure(self, sample_aggregate_data: Aggregations) -> None:
        """Test that aggregate entries have correct structure."""
        result = _order_aggregates_base(sample_aggregate_data)

        agg = result["6720797"]
        assert agg["type"] == "aggregate"
        assert "unit" in agg
        assert agg["up_id"] == "6720703"
        assert agg["dn_id"] == "6720797"
        assert agg["all_ref_ids"] == ["6720797"]

    def test_independent_structure(self, sample_aggregate_data: Aggregations) -> None:
        """Test that independent entries have correct structure."""
        result = _order_aggregates_base(sample_aggregate_data)

        ind = result["6720651"]
        assert ind["type"] == "independent"
        assert "unit" in ind
        assert ind["all_ref_ids"] == ["6720651"]
        assert "up_id" not in ind  # independents don't have up/dn_id
        assert "dn_id" not in ind

    def test_connector_structure(self, sample_aggregate_data: Aggregations) -> None:
        """Test that connector entries have correct structure."""
        result = _order_aggregates_base(sample_aggregate_data)

        conn = result["6720681"]
        assert conn["type"] == "small_scale_connector"
        assert "unit" in conn
        assert conn["all_ref_ids"] == ["6720681"]

    def test_preserves_geometries(self, sample_aggregate_data: Aggregations) -> None:
        """Test that geometries are preserved in units."""
        result = _order_aggregates_base(sample_aggregate_data)

        agg_unit = result["6720797"]["unit"]
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
            minor_flowpaths=[],
            small_scale_connectors=[],
        )

        result = _order_aggregates_base(empty_agg)
        assert result == {}

    def test_handles_only_aggregates(self) -> None:
        """Test with only aggregates, no independents or connectors."""
        only_agg = Aggregations(
            aggregates=[
                {
                    "dn_id": "fp1",
                    "up_id": "fp2",
                    "line_geometry": LineString([(0, 0), (1, 1)]),
                    "polygon_geometry": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                }
            ],
            independents=[],
            connectors=[],
            minor_flowpaths=[],
            small_scale_connectors=[],
        )

        result = _order_aggregates_base(only_agg)
        assert len(result) == 1
        assert "fp1" in result
        assert result["fp1"]["type"] == "aggregate"

    def test_uses_correct_keys_for_each_type(self, sample_aggregate_data: Aggregations) -> None:
        """Test that correct keys are used for different aggregate types."""
        result = _order_aggregates_base(sample_aggregate_data)

        # Aggregates use dn_id as key
        assert "6720797" in result
        assert result["6720797"]["dn_id"] == "6720797"

        # Independents use ref_ids as key
        assert "6720651" in result

        # Connectors use ref_ids as key
        assert "6720681" in result


class TestBuildBaseHydrofabric:
    """Tests for _build_base_hydrofabric function"""

    def test_builds_all_layers(
        self,
        sample_aggregate_data: Aggregations,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
        expected_graph: dict[str, list[str]],
        sample_config: HFConfig,
    ) -> None:
        """Test that all hydrofabric layers are created."""
        reference_flowpaths, reference_divides = sample_reference_data

        result = _build_base_hydrofabric(
            start_id="6720797",
            aggregate_data=sample_aggregate_data,
            classifications=sample_classifications,
            reference_divides=reference_divides,
            reference_flowpaths=reference_flowpaths,
            upstream_network=expected_graph,
            cfg=sample_config,
        )

        assert "flowpaths" in result
        assert "divides" in result
        assert "nexus" in result
        assert "base_minor_flowpaths" in result

    def test_creates_geodataframes(
        self,
        sample_aggregate_data: Aggregations,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
        expected_graph: dict[str, list[str]],
        sample_config: HFConfig,
    ) -> None:
        """Test that outputs are GeoDataFrames."""
        reference_flowpaths, reference_divides = sample_reference_data

        result = _build_base_hydrofabric(
            start_id="6720797",
            aggregate_data=sample_aggregate_data,
            classifications=sample_classifications,
            reference_divides=reference_divides,
            reference_flowpaths=reference_flowpaths,
            upstream_network=expected_graph,
            cfg=sample_config,
        )

        assert isinstance(result["flowpaths"], gpd.GeoDataFrame)
        assert isinstance(result["divides"], gpd.GeoDataFrame)
        assert isinstance(result["nexus"], gpd.GeoDataFrame)
        assert isinstance(result["base_minor_flowpaths"], list)

    def test_assigns_unique_ids(
        self,
        sample_aggregate_data: Aggregations,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
        expected_graph: dict[str, list[str]],
        sample_config: HFConfig,
    ) -> None:
        """Test that IDs are unique."""
        reference_flowpaths, reference_divides = sample_reference_data

        result = _build_base_hydrofabric(
            start_id="6720797",
            aggregate_data=sample_aggregate_data,
            classifications=sample_classifications,
            reference_divides=reference_divides,
            reference_flowpaths=reference_flowpaths,
            upstream_network=expected_graph,
            cfg=sample_config,
        )

        fp_ids: dict[str, pd.DataFrame] = result["flowpaths"]["fp_id"].tolist()  # type: ignore
        div_ids: dict[str, pd.DataFrame] = result["divides"]["div_id"].tolist()  # type: ignore
        nex_ids: dict[str, pd.DataFrame] = result["nexus"]["nex_id"].tolist()  # type: ignore

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
        """Test that IDs start at 1 when no offset provided."""
        reference_flowpaths, reference_divides = sample_reference_data

        result = _build_base_hydrofabric(
            start_id="6720797",
            aggregate_data=sample_aggregate_data,
            classifications=sample_classifications,
            reference_divides=reference_divides,
            reference_flowpaths=reference_flowpaths,
            upstream_network=expected_graph,
            cfg=sample_config,
        )

        assert min(result["flowpaths"]["fp_id"]) == 1  # type: ignore
        assert min(result["divides"]["div_id"]) == 1  # type: ignore
        assert min(result["nexus"]["nex_id"]) == 1  # type: ignore

    def test_applies_id_offset(
        self,
        sample_aggregate_data: Aggregations,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
        expected_graph: dict[str, list[str]],
        sample_config: HFConfig,
    ) -> None:
        """Test that ID offset is applied correctly."""
        reference_flowpaths, reference_divides = sample_reference_data

        offset = 100
        result = _build_base_hydrofabric(
            start_id="6720797",
            aggregate_data=sample_aggregate_data,
            classifications=sample_classifications,
            reference_divides=reference_divides,
            reference_flowpaths=reference_flowpaths,
            upstream_network=expected_graph,
            cfg=sample_config,
            id_offset=offset,
        )

        # All IDs should start at offset + 1
        assert min(result["flowpaths"]["fp_id"]) == offset + 1  # type: ignore
        assert min(result["divides"]["div_id"]) == offset + 1  # type: ignore
        assert min(result["nexus"]["nex_id"]) == offset + 1  # type: ignore

    def test_flowpath_divide_relationship(
        self,
        sample_aggregate_data: Aggregations,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
        expected_graph: dict[str, list[str]],
        sample_config: HFConfig,
    ) -> None:
        """Test that each flowpath has corresponding divide with same ID."""
        reference_flowpaths, reference_divides = sample_reference_data

        result = _build_base_hydrofabric(
            start_id="6720797",
            aggregate_data=sample_aggregate_data,
            classifications=sample_classifications,
            reference_divides=reference_divides,
            reference_flowpaths=reference_flowpaths,
            upstream_network=expected_graph,
            cfg=sample_config,
        )

        fp_div_ids = result["flowpaths"]["div_id"].tolist()  # type: ignore
        div_ids = result["divides"]["div_id"].tolist()  # type: ignore

        # Every flowpath's div_id should exist in divides
        assert all(div_id in div_ids for div_id in fp_div_ids)
        # Should be 1:1 relationship
        assert len(fp_div_ids) == len(div_ids)

    def test_nexus_connectivity(
        self,
        sample_aggregate_data: Aggregations,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
        expected_graph: dict[str, list[str]],
        sample_config: HFConfig,
    ) -> None:
        """Test that nexus points connect flowpaths correctly."""
        reference_flowpaths, reference_divides = sample_reference_data

        result: dict[str, pd.DataFrame] = _build_base_hydrofabric(
            start_id="6720797",
            aggregate_data=sample_aggregate_data,
            classifications=sample_classifications,
            reference_divides=reference_divides,
            reference_flowpaths=reference_flowpaths,
            upstream_network=expected_graph,
            cfg=sample_config,
        )

        flowpaths: pd.DataFrame = result["flowpaths"]
        nexus: pd.DataFrame = result["nexus"]

        # Every flowpath should have a downstream nexus
        assert all(pd.notna(fp_row["dn_nex_id"]) for _, fp_row in flowpaths.iterrows())

        # All downstream nexus IDs should exist in nexus layer
        dn_nex_ids = flowpaths["dn_nex_id"].unique()
        nex_ids = nexus["nex_id"].tolist()
        assert all(nex_id in nex_ids for nex_id in dn_nex_ids)

    def test_preserves_crs(
        self,
        sample_aggregate_data: Aggregations,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
        expected_graph: dict[str, list[str]],
        sample_config: HFConfig,
    ) -> None:
        """Test that CRS is preserved in output."""
        reference_flowpaths, reference_divides = sample_reference_data

        result: dict[str, gpd.GeoDataFrame] = _build_base_hydrofabric(
            start_id="6720797",
            aggregate_data=sample_aggregate_data,
            classifications=sample_classifications,
            reference_divides=reference_divides,
            reference_flowpaths=reference_flowpaths,
            upstream_network=expected_graph,
            cfg=sample_config,
        )

        assert result["flowpaths"].crs == sample_config.crs
        assert result["divides"].crs == sample_config.crs
        assert result["nexus"].crs == sample_config.crs

    def test_creates_base_minor_flowpaths(
        self,
        sample_aggregate_data: Aggregations,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
        expected_graph: dict[str, list[str]],
        sample_config: HFConfig,
    ) -> None:
        """Test that base_minor_flowpaths are created for aggregates."""
        reference_flowpaths, reference_divides = sample_reference_data

        result = _build_base_hydrofabric(
            start_id="6720797",
            aggregate_data=sample_aggregate_data,
            classifications=sample_classifications,
            reference_divides=reference_divides,
            reference_flowpaths=reference_flowpaths,
            upstream_network=expected_graph,
            cfg=sample_config,
        )

        base_minor = result["base_minor_flowpaths"]
        assert isinstance(base_minor, list)

        # Should have entries for aggregate types
        if len(base_minor) > 0:
            assert all("fp_id" in entry for entry in base_minor)
            assert all("dn_ref_id" in entry for entry in base_minor)
            assert all("up_ref_id" in entry for entry in base_minor)

    def test_handles_intersection_points(
        self,
        sample_aggregate_data: Aggregations,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
        expected_graph: dict[str, list[str]],
        sample_config: HFConfig,
    ) -> None:
        """Test nexus points connecting multiple flowpaths are set up_nex_id correctly."""
        reference_flowpaths, reference_divides = sample_reference_data

        result = _build_base_hydrofabric(
            start_id="6720797",
            aggregate_data=sample_aggregate_data,
            classifications=sample_classifications,
            reference_divides=reference_divides,
            reference_flowpaths=reference_flowpaths,
            upstream_network=expected_graph,
            cfg=sample_config,
        )

        flowpaths: pd.DataFrame = result["flowpaths"]

        # Find flowpaths that share a downstream nexus (intersection)
        dn_nexus_counts = flowpaths["dn_nex_id"].value_counts()
        intersection_nexus = dn_nexus_counts[dn_nexus_counts > 1]

        # If there are intersections, those flowpaths should have up_nex_id set
        if len(intersection_nexus) > 0:
            for nex_id in intersection_nexus.index:
                fps_at_intersection = flowpaths[flowpaths["dn_nex_id"] == nex_id]
                # At least one should have up_nex_id set
                assert fps_at_intersection["up_nex_id"].notna().any()

    def test_assigns_divide_types(
        self,
        sample_aggregate_data: Aggregations,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
        expected_graph: dict[str, list[str]],
        sample_config: HFConfig,
    ) -> None:
        """Test that divide types are assigned correctly."""
        reference_flowpaths, reference_divides = sample_reference_data

        result = _build_base_hydrofabric(
            start_id="6720797",
            aggregate_data=sample_aggregate_data,
            classifications=sample_classifications,
            reference_divides=reference_divides,
            reference_flowpaths=reference_flowpaths,
            upstream_network=expected_graph,
            cfg=sample_config,
        )

        divides: pd.DataFrame = result["divides"]
        types = divides["type"].unique()

        # Should have types from our sample data
        valid_types = {"aggregate", "independent", "small_scale_connector"}
        assert all(t in valid_types for t in types)

    def test_geometry_types(
        self,
        sample_aggregate_data: Aggregations,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
        expected_graph: dict[str, list[str]],
        sample_config: HFConfig,
    ) -> None:
        """Test that geometry types are correct for each layer."""
        reference_flowpaths, reference_divides = sample_reference_data

        result: dict[str, gpd.GeoDataFrame] = _build_base_hydrofabric(
            start_id="6720797",
            aggregate_data=sample_aggregate_data,
            classifications=sample_classifications,
            reference_divides=reference_divides,
            reference_flowpaths=reference_flowpaths,
            upstream_network=expected_graph,
            cfg=sample_config,
        )

        # Flowpaths should be LineStrings or MultiLineStrings
        for geom in result["flowpaths"].geometry:
            assert geom.geom_type in ["LineString", "MultiLineString"]

        # Divides should be Polygons or MultiPolygons
        for geom in result["divides"].geometry:
            assert geom.geom_type in ["Polygon", "MultiPolygon"]

        # Nexus should be Points
        for geom in result["nexus"].geometry:
            assert geom.geom_type == "Point"


class TestAggregateGeometries:
    """Tests for _aggregate_geometries function"""

    def test_creates_aggregations_object(
        self,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
    ) -> None:
        """Test that function returns Aggregations object."""
        reference_flowpaths, reference_divides = sample_reference_data

        result = _aggregate_geometries(
            classifications=sample_classifications,
            reference_divides=reference_divides,
            reference_flowpaths=reference_flowpaths,
        )

        assert isinstance(result, Aggregations)

    def test_aggregates_have_geometries(
        self,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
    ) -> None:
        """Test that aggregated units have line and polygon geometries."""
        reference_flowpaths, reference_divides = sample_reference_data

        result = _aggregate_geometries(
            classifications=sample_classifications,
            reference_divides=reference_divides,
            reference_flowpaths=reference_flowpaths,
        )

        if len(result.aggregates) > 0:
            agg = result.aggregates[0]
            assert "line_geometry" in agg
            assert "polygon_geometry" in agg
            # Geometries might be None if not found, so check type only if present
            if agg["line_geometry"] is not None:
                assert isinstance(agg["line_geometry"], MultiLineString)
            if agg["polygon_geometry"] is not None:
                assert isinstance(agg["polygon_geometry"], MultiPolygon)

    def test_independents_have_geometries(
        self,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
    ) -> None:
        """Test that independent units have geometries."""
        reference_flowpaths, reference_divides = sample_reference_data

        result = _aggregate_geometries(
            classifications=sample_classifications,
            reference_divides=reference_divides,
            reference_flowpaths=reference_flowpaths,
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
        """Test that aggregation pairs are processed."""
        reference_flowpaths, reference_divides = sample_reference_data

        # Ensure we have aggregation pairs
        assert len(sample_classifications.aggregation_pairs) > 0

        result = _aggregate_geometries(
            classifications=sample_classifications,
            reference_divides=reference_divides,
            reference_flowpaths=reference_flowpaths,
        )

        # Should have created aggregates
        assert len(result.aggregates) > 0

    def test_processes_independents(
        self,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
    ) -> None:
        """Test that independent flowpaths are processed."""
        reference_flowpaths, reference_divides = sample_reference_data

        # Ensure we have independents
        assert len(sample_classifications.independent_flowpaths) > 0

        result = _aggregate_geometries(
            classifications=sample_classifications,
            reference_divides=reference_divides,
            reference_flowpaths=reference_flowpaths,
        )

        # Should have created independents
        assert len(result.independents) > 0

    def test_processes_connectors(
        self,
        sample_classifications: Classifications,
        sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
    ) -> None:
        """Test that connector segments are processed."""
        reference_flowpaths, reference_divides = sample_reference_data

        # Ensure we have connectors
        assert len(sample_classifications.connector_segments) > 0

        result = _aggregate_geometries(
            classifications=sample_classifications,
            reference_divides=reference_divides,
            reference_flowpaths=reference_flowpaths,
        )

        # Should have created connectors
        assert len(result.connectors) > 0

    def test_handles_empty_classifications(
        self, sample_reference_data: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
    ) -> None:
        """Test handling of empty classifications."""
        reference_flowpaths, reference_divides = sample_reference_data

        empty_classifications = Classifications(
            aggregation_pairs=[],
            minor_flowpaths=[],
            independent_flowpaths=[],
            connector_segments=[],
            subdivide_candidates=[],
            upstream_merge_points=[],
            processed_flowpaths=set(),
            cumulative_merge_areas={},
        )

        result = _aggregate_geometries(
            classifications=empty_classifications,
            reference_divides=reference_divides,
            reference_flowpaths=reference_flowpaths,
        )

        # Should return empty lists
        assert len(result.aggregates) == 0
        assert len(result.independents) == 0
        assert len(result.connectors) == 0
