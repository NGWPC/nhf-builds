"""Tests the download pipeline"""

import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Polygon

from hydrofabric_builds.pipeline.download import _validate_and_fix_geometries, download_reference_data
from scripts.hf_runner import LocalRunner


class TestGeometryValidation:
    """Tests for geometry validation and fixing."""

    @pytest.fixture
    def valid_divides(self, sample_divides: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Get only valid geometries from sample divides."""
        return sample_divides[sample_divides.geometry.is_valid].copy()

    @pytest.fixture
    def invalid_geometry(self) -> Polygon:
        """Create a self-intersecting (invalid) polygon - bowtie shape."""
        # Creates a bowtie/figure-8 polygon that intersects itself
        coords = [
            (0, 0),
            (2, 2),
            (2, 0),
            (0, 2),
            (0, 0),
        ]
        return Polygon(coords)

    @pytest.fixture
    def divides_with_invalid_geom(
        self, valid_divides: gpd.GeoDataFrame, invalid_geometry: Polygon
    ) -> gpd.GeoDataFrame:
        """Create a GeoDataFrame with one invalid geometry."""
        gdf = valid_divides.head(5).copy()
        # Replace first geometry with invalid one
        gdf.at[gdf.index[0], "geometry"] = invalid_geometry
        return gdf

    def test_all_valid_geometries_pass(self, valid_divides: gpd.GeoDataFrame) -> None:
        """Test that GeoDataFrame with all valid geometries passes validation."""
        result = _validate_and_fix_geometries(valid_divides.head(10), geom_type="divides")

        assert len(result) == 10
        assert result.geometry.is_valid.all(), "All geometries should remain valid"
        assert len(result) == len(valid_divides.head(10)), "Should not lose any features"

    def test_fixes_invalid_geometry(self, divides_with_invalid_geom: gpd.GeoDataFrame) -> None:
        """Test that invalid geometry is fixed."""
        assert not divides_with_invalid_geom.iloc[0].geometry.is_valid

        result = _validate_and_fix_geometries(divides_with_invalid_geom, geom_type="divides")

        assert result.geometry.is_valid.all(), "All geometries should be fixed"
        assert len(result) == len(divides_with_invalid_geom), "Should not lose features"
        assert not result.iloc[0].geometry.is_empty, "Fixed geometry should not be empty"

    def test_preserves_valid_geometries(self, divides_with_invalid_geom: gpd.GeoDataFrame) -> None:
        """Test that valid geometries are not modified."""
        original_geom = divides_with_invalid_geom.iloc[1].geometry

        result = _validate_and_fix_geometries(divides_with_invalid_geom, geom_type="divides")
        assert result.iloc[1].geometry.equals(original_geom), "Valid geometries should not be modified"

    def test_handles_empty_geodataframe(self) -> None:
        """Test that empty GeoDataFrame is handled gracefully."""
        empty_gdf = gpd.GeoDataFrame(columns=["divide_id", "geometry"], geometry="geometry")

        result = _validate_and_fix_geometries(empty_gdf, geom_type="divides")

        assert len(result) == 0
        assert isinstance(result, gpd.GeoDataFrame)

    def test_works_with_flowpaths(self, sample_flowpaths: gpd.GeoDataFrame) -> None:
        """Test validation works with sample flowpaths data."""
        result = _validate_and_fix_geometries(
            sample_flowpaths.head(10),
            geom_type="flowpaths",
        )

        assert result.geometry.is_valid.all()
        assert len(result) == 10

    def test_preserves_attributes(self, divides_with_invalid_geom: gpd.GeoDataFrame) -> None:
        """Test that all non-geometry columns are preserved."""
        original_columns = set(divides_with_invalid_geom.columns)

        result = _validate_and_fix_geometries(divides_with_invalid_geom, geom_type="divides")

        result_columns = set(result.columns)
        assert original_columns == result_columns, "Should preserve all columns"

        # Check that non-geometry data is unchanged for valid rows
        for col in divides_with_invalid_geom.columns:
            if col != "geometry":
                assert divides_with_invalid_geom[col].equals(result[col]), f"Column {col} should be unchanged"

    def test_complex_multipolygon_invalid(self) -> None:
        """Test fixing complex invalid multipolygon."""
        # Create a complex invalid multipolygon
        poly1 = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])  # Invalid bowtie
        poly2 = Polygon([(3, 3), (5, 3), (5, 5), (3, 5), (3, 3)])  # Valid square
        invalid_multipoly = MultiPolygon([poly1, poly2])

        gdf = gpd.GeoDataFrame(
            {"divide_id": ["test_1"], "geometry": [invalid_multipoly]}, geometry="geometry"
        )

        result = _validate_and_fix_geometries(gdf, geom_type="test_divides")

        assert result.geometry.is_valid.all(), "Should fix complex multipolygon"
        assert not result.geometry.iloc[0].is_empty, "Should not result in empty geometry"


class TestDownloadReferenceData:
    """Tests for download_reference_data function."""

    def test_loads_flowpaths(self, runner: LocalRunner) -> None:
        """Test that flowpaths are loaded from GeoPackage."""
        # Run the task
        runner.run_task(task_id="download", python_callable=download_reference_data, op_kwargs={})

        # Check XCom for flowpaths
        flowpaths = runner.ti.xcom_pull("download", key="reference_flowpaths")

        assert flowpaths is not None
        assert isinstance(flowpaths, gpd.GeoDataFrame)
        assert len(flowpaths) == 85
        assert "flowpath_id" in flowpaths.columns
        assert "VPUID" in flowpaths.columns

    def test_loads_divides(self, runner: LocalRunner) -> None:
        """Test that divides are loaded from GeoPackage."""

        # Run the task
        runner.run_task(task_id="download", python_callable=download_reference_data, op_kwargs={})

        # Check XCom for divides
        divides = runner.ti.xcom_pull("download", key="reference_divides")

        assert divides is not None
        assert isinstance(divides, gpd.GeoDataFrame)
        assert len(divides) == 85
        assert "divide_id" in divides.columns
