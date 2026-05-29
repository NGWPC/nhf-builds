import shutil
from pathlib import Path

import geopandas as gpd
import pytest
from geopandas.testing import assert_geodataframe_equal
from pyprojroot import here
from shapely import Point, box

from hydrofabric_builds.config import HFConfig, TaskSelection
from hydrofabric_builds.lakes.lakes_pipeline import lakes_pipeline
from hydrofabric_builds.schemas.hydrofabric import BuildHydrofabricConfig
import rasterio
from rasterio.transform import from_bounds
import numpy as np
import pandas as pd


@pytest.fixture
def lakes_root() -> Path:
    return here() / "tests/data/lakes"


@pytest.fixture
def main_lakes_nhf() -> str:
    return "nhf_lakes_test.gpkg"

@pytest.fixture
def dummy_dem(lakes_root)->Path:
    dem = lakes_root / "dummy_dem.tif"

    bbox = (-1728875.0003, 1342405.00029999, -1701024.9997, 1372475.0003)
    width, height = 1000, 1000
    transform = from_bounds(*bbox, width, height)
    data = np.full((height, width), 10, dtype=np.float32)

    with rasterio.open(
        dem,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=5070,
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    return dem

@pytest.fixture
def nid(lakes_root)->Path:
    nid = lakes_root / "nid.csv"
    df = pd.DataFrame(data = {
        "latitude":[33.746400,33.748557],
        "longitude":[-114.701015,-114.703772],
        "nidid":['nid1', 'nid2'],
        "dam_name":['d1', 'd2'],
        "dam_type":['RE', 'RE'],
        "spillway_type":["Orifice","Orifice"],
        "spillway_width":[50, 50],
        "dam_length":[10, 10],
        "dam_height":[10, 10],
        "structural_height":[10, 10],
        "hydraulic_height":[5,5],
        "nid_height":[10, 10],
        "surface_area":[100, 100],
        "wb_areasqkm":[100,100],
        "nid_storage":[100,100],
        "normal_storage":[100,100],
        "max_storage":[150,150],
        "hazard":["H","L"],
        "purposes":["IS","IS"],
    })
    df.to_csv(nid,index=False)

    return nid

@pytest.fixture
def main_cfg(lakes_root, main_lakes_nhf) -> HFConfig:
    return HFConfig(
        output_dir=lakes_root,
        output_name=main_lakes_nhf,
        build=BuildHydrofabricConfig(reference_flowpaths_path=str(lakes_root / "reference_flowpaths.parquet")),
        tasks=TaskSelection(
            build_hydrofabric=False,
            divide_attributes=False,
            flowpath_attributes=False,
            gages=False,
            fp_crosswalk=False,
            validate_hf=False,
            hydrolocations=False,
            lakes=True,
        ),
    )




def test__use_cached(main_cfg: HFConfig, lakes_root):
    """Use a cached lakes layer"""
    try:
        tmp_nhf = lakes_root / "nhf_tmp.gpkg"
        shutil.copy(main_cfg.output_file_path, tmp_nhf)

        tmp_lakes = lakes_root / "tmp_lakes.gpkg"
        expected_lakes = gpd.GeoDataFrame(geometry=[Point(0, 0)], data={"lake_id": [1]}, crs=5070)
        expected_lakes.to_file(tmp_lakes, driver="GPKG")

        cfg = main_cfg.model_copy()
        cfg.output_name = tmp_nhf.name
        cfg.output_file_path = tmp_nhf
        cfg.lakes.use_cached_lakes = True
        cfg.lakes.lakes_path = tmp_lakes

        lakes_pipeline(cfg)

        gdf = gpd.read_file(tmp_nhf, layer="lakes")
        assert_geodataframe_equal(gdf, expected_lakes)

    finally:
        tmp_lakes.unlink(missing_ok=True)
        tmp_nhf.unlink(missing_ok=True)


def test__no_layers(main_cfg: HFConfig, lakes_root):
    """No files are requested to run and a blank layer is written"""
    try:
        tmp_nhf = lakes_root / "nhf_tmp.gpkg"
        shutil.copy(main_cfg.output_file_path, tmp_nhf)

        cfg = main_cfg.model_copy()
        cfg.output_name = tmp_nhf.name
        cfg.output_file_path = tmp_nhf
        cfg.lakes.adhoc.run = False
        cfg.lakes.ref_res.run = False
        cfg.lakes.nwm.run = False
        cfg.lakes.ref_wb.run = False

        expected_lakes = gpd.GeoDataFrame(columns=cfg.lakes.fields + ["geometry"], crs=cfg.crs)
        expected_lakes.to_file(tmp_nhf, layer="lakes", driver="GPKG")

        lakes_pipeline(cfg)

        gdf = gpd.read_file(tmp_nhf, layer="lakes")
        assert_geodataframe_equal(gdf, expected_lakes)

    finally:
        tmp_nhf.unlink(missing_ok=True)

def test__run_nwm(main_cfg: HFConfig, lakes_root, dummy_dem, nid):
    """No files are requested to run and a blank layer is written"""
    try:
        tmp_nhf = lakes_root / "nhf_tmp.gpkg"
        shutil.copy(main_cfg.output_file_path, tmp_nhf)

        cfg = main_cfg.model_copy()
        cfg.output_name = tmp_nhf.name
        cfg.output_file_path = tmp_nhf
        cfg.lakes.dem.path = dummy_dem
        cfg.lakes.nid.path = nid

        tmp_nwm = lakes_root / "tmp_nwm_lakes.gpkg"
        gdf_nwm_lk = gpd.GeoDataFrame(crs=5070, geometry=[box(-1718569.5,1363475.7, -1717437.7, 1363977.3)], data={'newID':[1]})
        gdf_nwm_lk.to_file(tmp_nwm, layer='lakes')
        cfg.lakes.nwm.path = tmp_nwm

        tmp_nwm_attr = lakes_root / "tmp_nwm_attr.gpkg"
        gdf_nwm_attr = gpd.GeoDataFrame(geometry=[Point(-1718569.5,1363475.7)],crs=5070, data ={'lake_id':[1],
                                                                                             'LkArea':[1],
                                                                                             'LkMxE':[1],
                                                                                             'WeirC':[0.4],
                                                                                             'WeirL':[10],
                                                                                             'OrificeC':[0.1],
                                                                                             'OrificeA':[1],
                                                                                             'OrificeE':[2], 'WeirE':[1], 'ifd':[0.8999999761581421], 'Dam_Length':[10]})

        gdf_nwm_attr.to_file(tmp_nwm, layer='lakes_attr')
        cfg.lakes.nwm.attrib_src_layer = tmp_nwm_attr
        cfg.lakes.nwm.attrib_src_layer = 'lakes_attr'
        cfg.lakes.nwm.attrib_src_key = 'lake_id'

        cfg.lakes.nwm.run = True
        cfg.lakes.adhoc.run = False
        cfg.lakes.ref_res.run = False
        cfg.lakes.ref_wb.run = False

        lakes_pipeline(cfg)

        expected_lakes = gpd.GeoDataFrame(geometry=[Point(-1718569.5,1363475.7)], data ={'nhf_lake_id':[1],
                                                                                         'ref_fp_id':[9999572],
                                                                                         "fp_id": [None],
                                                                                         "virtual_fp_id":[1261203455606765],
                                                                                         'dn_nex_id':[None],
                                                                                         'dn_virtual_nex_id':[1261203489061613],
                                                                                         'div_id':[1261204749791466],
                                                                                         'lake_id':[1],
                                                                                             'LkArea':[1],
                                                                                             'LkMxE':[1],
                                                                                             'WeirC':[0.4],
                                                                                             'WeirL':[10],
                                                                                             'OrificeC':[0.1],
                                                                                             'OrificeA':[1],
                                                                                             'OrificeE':[2], 'WeirE':[1], 'ifd':[0.8999999761581421], 'Dam_Length':[10]})

        gdf = gpd.read_file(tmp_nhf, layer="lakes")
        assert_geodataframe_equal(gdf, expected_lakes, gdf)

    finally:
        tmp_nhf.unlink(missing_ok=True)
        tmp_nwm_attr.unlink(missing_ok=True)
        tmp_nwm.unlink(missing_ok=True)
        dummy_dem.unlink(missing_ok=True)
        nid.unlink(missing_ok=True)


    def test_join_nid(cfg, res_df)
