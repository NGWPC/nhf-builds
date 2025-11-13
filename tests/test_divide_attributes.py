from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from astropy.stats.circstats import circmean
from pandas.testing import assert_frame_equal
from pyprojroot import here

from hydrofabric_builds.config import HYDROFABRIC_OUTPUT_FILE
from hydrofabric_builds.helpers.stats import weighted_circular_mean, weighted_geometric_mean
from hydrofabric_builds.hydrofabric.divide_attributes import (
    _calculate_attribute,
    _config_reader,
    _prep_multiprocessing_configs,
    _prep_multiprocessing_rasters,
    _vpu_splitter,
    divide_attributes_pipeline_parallel,
    divide_attributes_pipeline_single,
)
from hydrofabric_builds.schemas.hydrofabric import (
    AggTypeEnum,
    DivideAttributeConfig,
    DivideAttributeModelConfig,
)


class TestDivideAttributesSchemas:
    """Tests for Divide Attribute Schemas"""

    def test_divide_attribute_config_pydantic(self) -> None:
        """DivideAttributeConfig model - no defaults"""
        model = DivideAttributeConfig(
            agg_type="mean",
            field_name="test",
            data_loader="tif",
            file_name="./data/test.tif",
            tmp="/tmp/divide-attributes/test.parquet",
        )
        assert model.agg_type == AggTypeEnum.mean
        assert model.field_name == "test"
        assert model.file_name == Path("./data/test.tif")
        assert model.tmp == Path("/tmp/divide-attributes/test.parquet")

    def test_divide_attribute_config_pydantic__defaults(self) -> None:
        """DivideAttributeConfig model - default factory for DivideAttributeConfig.tmp"""
        model = DivideAttributeConfig(
            agg_type="mean", field_name="test", data_loader="tif", file_name="./data/test.tif"
        )
        assert model.agg_type == AggTypeEnum.mean
        assert model.field_name == "test"
        assert model.file_name == Path("./data/test.tif")
        assert model.tmp == Path("/tmp/divide-attributes/tmp_test.parquet")  # default

    def test_divide_attribute_model_config_pydantic(self) -> None:
        """DivideAttributeModelConfig model - no defaults"""
        tmp_dir = Path("./data/tmp/divide-attributes")
        attributes = [
            DivideAttributeConfig(
                agg_type="mean", field_name="test", data_loader="tif", file_name="test.tif"
            ),
            DivideAttributeConfig(
                agg_type="mode", field_name="test2", data_loader="tif", file_name="test2.tif"
            ),
        ]
        model = DivideAttributeModelConfig(
            data_dir="./data",
            divides_path="./data/divides.gpkg",
            divide_id="div_id",
            output="./data/divides.gpkg",
            divides_path_list=["./data/div1.gpkg", "./data/div2.gpkg"],
            tmp_dir=tmp_dir,
            attributes=attributes,
            split_vpu=False,
            debug=True,
        )

        # model validator creates tmp folder
        assert tmp_dir.exists()

        assert model.data_dir == Path("./data")
        assert model.divides_path == Path("./data/divides.gpkg")
        assert model.divide_id == "div_id"
        assert model.attributes == attributes
        assert model.output == Path("./data/divides.gpkg")
        assert model.divides_path_list == [Path("./data/div1.gpkg"), Path("./data/div2.gpkg")]
        assert model.tmp_dir == Path("./data/tmp/divide-attributes")
        assert model.split_vpu is False
        assert model.debug is True

        # teardown
        if tmp_dir.is_dir():
            try:
                tmp_dir.rmdir()
            except OSError:
                return

    def test_divide_attribute_model_config_pydantic__defaults(self) -> None:
        """DivideAttributeModelConfig model - defaults [divide_id, crs, tmp_dir, split_vpu, output]"""
        tmp_dir = Path("/tmp/divide-attributes")
        attributes = [
            DivideAttributeConfig(
                agg_type="mean", field_name="test", data_loader="tif", file_name="test.tif"
            ),
            DivideAttributeConfig(
                agg_type="mode", field_name="test2", data_loader="tif", file_name="test2.tif"
            ),
        ]
        model = DivideAttributeModelConfig(attributes=attributes)

        # model validator creates tmp folder
        assert tmp_dir.exists()

        assert model.data_dir == here() / "data"  # default
        assert model.divides_path == HYDROFABRIC_OUTPUT_FILE  # default
        assert model.divide_id == "divide_id"  # default
        assert model.attributes == attributes
        assert model.output == HYDROFABRIC_OUTPUT_FILE  # default
        assert model.divides_path_list is None  # default
        assert model.tmp_dir == Path("/tmp/divide-attributes")  # default
        assert model.split_vpu is False  # default
        assert model.debug is False  # default

        # teardown
        if tmp_dir.is_dir():
            try:
                tmp_dir.rmdir()
            except OSError:
                return


class TestDivideAttributes:
    test_tmp_dir = here() / "tests/data/divide_attributes/tmp/divide-attributes"

    def test_config_reader(self, divide_attributes_config_yaml: str) -> None:
        """Check path logic"""
        data = _config_reader(divide_attributes_config_yaml)

        assert data.attributes[0].file_name == Path("./tests/data/divide_attributes/bexp_0.tif")
        assert data.attributes[1].file_name == Path("./tests/data/divide_attributes/dksat_0.tif")
        assert data.attributes[2].file_name == Path(
            "./tests/data/divide_attributes/usgs_250m_aspect_5070.tif"
        )
        assert data.attributes[3].file_name == Path("./tests/data/divide_attributes/twi.tif")

    def test_vpu_splitter(self, divide_attributes_model_config: DivideAttributeModelConfig) -> None:
        """VPU spliter splits gpkg with 2 VPUs"""
        gpkgs = _vpu_splitter(divide_attributes_model_config)

        try:
            expected = [divide_attributes_model_config.tmp_dir / f"vpu_{vpu}.gpkg" for vpu in ["03N", "05"]]
            assert set(gpkgs) == set(expected)
            assert len(gpkgs) == len(expected)

            for gpkg in gpkgs:
                assert gpkg.exists() is True
                gdf = gpd.read_file(gpkg, layer="divides")
                assert len(gdf["vpuid"].unique()) == 1
        finally:
            for gpkg in gpkgs:
                gpkg.unlink(missing_ok=True)

    def test_calculate_attributes__mode(
        self,
        divide_attributes_model_config: DivideAttributeModelConfig,
        divide_attributes_bexp: dict[str, Any],
    ) -> None:
        """Fixture includes config, VPU03N, and result"""
        try:
            _calculate_attribute(
                divide_attributes_model_config,
                divide_attributes_bexp["config"],
                alt_divides_path=divide_attributes_bexp["vpu_path"],
            )

            assert divide_attributes_bexp["config"].tmp.exists() is True
            df = pd.read_parquet(divide_attributes_bexp["config"].tmp)
            assert_frame_equal(df[["bexp_mode"]], divide_attributes_bexp["results"], check_exact=False)

        finally:
            divide_attributes_bexp["config"].tmp.unlink(missing_ok=True)

    def test_calculate_attributes__quartile(
        self,
        divide_attributes_model_config: DivideAttributeModelConfig,
        divide_attributes_twi: dict[str, Any],
    ) -> None:
        """Fixture includes config, VPU03N, and results"""
        try:
            _calculate_attribute(
                divide_attributes_model_config,
                divide_attributes_twi["config"],
                alt_divides_path=divide_attributes_twi["vpu_path"],
            )

            assert divide_attributes_twi["config"].tmp.exists() is True
            df = pd.read_parquet(divide_attributes_twi["config"].tmp)
            assert_frame_equal(
                df[["twi_q25", "twi_q50", "twi_q75", "twi_q100"]],
                divide_attributes_twi["results"],
                check_exact=False,
            )

        finally:
            divide_attributes_twi["config"].tmp.unlink(missing_ok=True)

    def test_calculate_attributes__circmean(
        self,
        divide_attributes_model_config: DivideAttributeModelConfig,
        divide_attributes_aspect: dict[str, Any],
    ) -> None:
        """Fixture includes config, VPU03N, and results. Custom exactextract"""
        try:
            _calculate_attribute(
                divide_attributes_model_config,
                divide_attributes_aspect["config"],
                alt_divides_path=divide_attributes_aspect["vpu_path"],
            )

            assert divide_attributes_aspect["config"].tmp.exists() is True
            df = pd.read_parquet(divide_attributes_aspect["config"].tmp)
            assert_frame_equal(
                df[["aspect_circmean"]],
                divide_attributes_aspect["results"],
                check_exact=False,
            )

        finally:
            divide_attributes_aspect["config"].tmp.unlink(missing_ok=True)

    def test_calculate_attributes__gmean(
        self,
        divide_attributes_model_config: DivideAttributeModelConfig,
        divide_attributes_dksat: dict[str, Any],
    ) -> None:
        """Fixture includes config, VPU03N, and results. Custom exactextract"""
        try:
            _calculate_attribute(
                divide_attributes_model_config,
                divide_attributes_dksat["config"],
                alt_divides_path=divide_attributes_dksat["vpu_path"],
            )

            assert divide_attributes_dksat["config"].tmp.exists() is True
            df = pd.read_parquet(divide_attributes_dksat["config"].tmp)
            assert_frame_equal(
                df[["dksat_geomean"]],
                divide_attributes_dksat["results"],
                check_exact=False,
            )

        finally:
            divide_attributes_dksat["config"].tmp.unlink(missing_ok=True)

    multiprocessing_raster_data = [
        pytest.param(
            2,
            2,
            [
                test_tmp_dir / "bexp_0_0.tif",
                test_tmp_dir / "bexp_0_1.tif",
            ],
            id="multiprocessing rasters 2 processes, 2 divides",
        ),
        pytest.param(
            2,
            3,
            [
                test_tmp_dir / "bexp_0_0.tif",
                test_tmp_dir / "bexp_0_1.tif",
            ],
            id="multiprocessing rasters 3 processes, 2 divides",
        ),
    ]

    @pytest.mark.parametrize("processes,n_divides,expected", multiprocessing_raster_data)
    def test_prep_multiprocessing_raster(
        self,
        divide_attributes_bexp: dict[str, Any],
        divide_attribute_tmp_dir: Path,
        processes: int,
        n_divides: int,
        expected: list[Path],
    ) -> None:
        """Generates raster per process but no more rasters than divides"""
        raster_list = _prep_multiprocessing_rasters(
            processes=processes,
            cfg=divide_attributes_bexp["config"],
            n_divides=n_divides,
            tmp_dir=divide_attribute_tmp_dir,
        )
        try:
            assert raster_list == expected
            for ras in raster_list:
                assert ras.exists() is True
        finally:
            for ras in raster_list:
                ras.unlink() if ras.exists() else None

    multiprocessing_config_data = [
        pytest.param(
            2,
            2,
            [
                test_tmp_dir / "bexp_0_0.tif",
                test_tmp_dir / "bexp_0_1.tif",
            ],
            [
                test_tmp_dir / "tmp_bexp_mode_0.parquet",
                test_tmp_dir / "tmp_bexp_mode_1.parquet",
            ],
            [
                test_tmp_dir / "bexp_0_0.tif",
                test_tmp_dir / "bexp_0_1.tif",
            ],
            id="multiprocessing config generator 2 processes, 2 divides",
        ),
        pytest.param(
            2,
            3,
            [
                test_tmp_dir / "bexp_0_0.tif",
                test_tmp_dir / "bexp_0_1.tif",
            ],
            [
                test_tmp_dir / "tmp_bexp_mode_0.parquet",
                test_tmp_dir / "tmp_bexp_mode_1.parquet",
                test_tmp_dir / "tmp_bexp_mode_2.parquet",
            ],
            [
                test_tmp_dir / "bexp_0_0.tif",
                test_tmp_dir / "bexp_0_1.tif",
                test_tmp_dir / "bexp_0_0.tif",
            ],
            id="multiprocessing config genreator 2 processes, 3 divides",
        ),
    ]

    @pytest.mark.parametrize(
        "processes,n_divides,raster_list,expected_tmp_files,expected_file_names", multiprocessing_config_data
    )
    def test_prep_multiprocessing_config(
        self,
        divide_attributes_bexp: dict[str, Any],
        divide_attribute_tmp_dir: Path,
        processes: int,
        n_divides: int,
        raster_list: list[Path],
        expected_tmp_files: list[Path],
        expected_file_names: list[Path],
    ) -> None:
        """Generates config per process with correct raster and temp name"""
        configs = _prep_multiprocessing_configs(
            processes=processes,
            cfg=divide_attributes_bexp["config"],
            n_divides=n_divides,
            tmp_dir=divide_attribute_tmp_dir,
            raster_list=raster_list,
        )

        for i, cfg in enumerate(configs):
            assert cfg.tmp == expected_tmp_files[i]
            assert cfg.file_name == expected_file_names[i]

    def test_divide_attributes_pipeline_single(
        self, divide_attributes_config_yaml: Path, pipeline_results: pd.DataFrame
    ) -> None:
        """Single pipeline"""
        try:
            cfg = _config_reader(str(divide_attributes_config_yaml))
            divide_attributes_pipeline_single(str(divide_attributes_config_yaml))
            gdf = gpd.read_file(cfg.output)
            assert_frame_equal(
                gdf[
                    [
                        "bexp_mode",
                        "dksat_geomean",
                        "twi_q25",
                        "twi_q50",
                        "twi_q75",
                        "twi_q100",
                        "aspect_circmean",
                    ]
                ],
                pipeline_results,
                check_exact=False,
            )

        finally:
            path_list = cfg.tmp_dir.glob("*")
            for f in path_list:
                f.unlink(missing_ok=True)
            cfg.output.unlink(missing_ok=True)

    def test_divide_attributes_pipeline_parallel(
        self, divide_attributes_config_yaml: Path, pipeline_results: pd.DataFrame
    ) -> None:
        try:
            cfg = _config_reader(str(divide_attributes_config_yaml))
            divide_attributes_pipeline_parallel(str(divide_attributes_config_yaml), processes=2)
            gdf = gpd.read_file(cfg.output)
            assert_frame_equal(
                gdf[
                    [
                        "bexp_mode",
                        "dksat_geomean",
                        "twi_q25",
                        "twi_q50",
                        "twi_q75",
                        "twi_q100",
                        "aspect_circmean",
                    ]
                ],
                pipeline_results,
                check_exact=False,
            )

        finally:
            path_list = cfg.tmp_dir.glob("*")
            for f in path_list:
                f.unlink(missing_ok=True)

            cfg.output.unlink(missing_ok=True)


class TestDivideAttributesStates:
    def test_weighted_circular_mean(self) -> None:
        """astropy is the reference for our function. astropy is test-only dependency.
        astropy input is in radians, our implementation is in degrees
        """
        values = [0, 1, 2]
        weights = [0, 0, 1]

        assert weighted_circular_mean(values, weights) == circmean(np.radians(values), weights=weights)

    def test_weighted_circular_mean__bad_weights(self) -> None:
        """Value error for incorrect weights shape"""
        values = [0, 1, 2]
        weights = [0, 0, 1, 1]

        with pytest.raises(ValueError):
            weighted_circular_mean(values, weights)

    def test_weighted_geometric_mean(self) -> None:
        values = [1, 1, 2]
        weights = [0, 0, 1]

        assert weighted_geometric_mean(values, weights) == np.array([2.0])
