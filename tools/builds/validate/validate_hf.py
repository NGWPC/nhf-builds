import argparse

import geopandas as gpd
import pandas as pd
from pandantic import Pandantic
from pydantic import BaseModel, Field, ValidationError


class Divides(BaseModel):
    """Pydantic class containing the data type, range (if known) of the divide attributes"""

    div_id: int = Field(gt=0)
    vpu_id: str
    type: str
    area_sqkm: float = Field(gt=0.0)
    bexp_mode: float = Field(gt=2.0, lt=15.0)
    isltyp_mode: float = Field(ge=1, le=16)
    ivgtyp_mode: float = Field(ge=1, le=16)
    dksat_geomean: float = Field(gt=1.95e-07, lt=1.41e-03)
    psisat_geomean: float = Field(gt=0.036, lt=0.955)
    cwpvt_mean: float = Field(gt=0.09, lt=0.36)
    mp_mean: float = Field(gt=3.6, lt=12.6)
    mfsno_mean: float = Field(gt=0.5, lt=4.0)
    quartz_mean: float = Field(gt=0.0, lt=1.0)
    refkdt_mean: float = Field(gt=0.1, lt=4.0)
    slope1km_mean: float = Field(gt=0.0, lt=1.0)
    smcmax_mean: float = Field(gt=0.16, lt=0.9)
    smcwlt_mean: float = Field(gt=0.05, lt=0.3)
    vcmx_mean: float = Field(gt=24.0, lt=112.0)
    imperv_mean: float = Field(gt=0.0, lt=1.0)
    twi_q25: float
    twi_q50: float
    twi_q75: float
    twi_q100: float
    twi_q10: float
    twi_q20: float
    twi_q30: float
    twi_q40: float
    twi_q60: float
    twi_q70: float
    twi_q80: float
    twi_q90: float
    elevation_mean: float = Field(gt=-86.0, lt=4422.0)
    slope250m_mean: float = Field(gt=0.0, lt=90.0)
    aspect_circmean: float = Field(gt=0.0, lt=360.0)
    lzfpm_mean: float = Field(gt=40.0, lt=600.0)
    lzpk_mean: float = Field(gt=0.001, lt=0.015)
    lztwm_mean: float = Field(gt=75.0, lt=300.0)
    rexp_mean: float = Field(gt=1.4, lt=3.5)
    uzk_mean: float = Field(gt=0.2, lt=0.5)
    zperc_mean: float = Field(gt=0.0, lt=360.0)
    lzfsm_mean: float = Field(gt=0.0, lt=360.0)
    lzsk_mean: float = Field(gt=0.03, lt=0.2)
    pfree_mean: float = Field(gt=0.0, lt=0.5)
    uzfwm_mean: float = Field(gt=10.0, lt=100.0)
    uztwm_mean: float = Field(gt=25.0, lt=125.0)
    mfmin_mean: float = Field(gt=0.01, lt=0.6)
    mfmax_mean: float = Field(gt=0.0, lt=360.0)
    uadj_mean: float = Field(gt=0.01, lt=0.2)
    a_xinanjiang_inflection_point_parameter: float = Field(gt=-0.5, lt=0.5)
    b_xinanjiang_shape_parameter: float = Field(gt=0.01, lt=10.0)
    x_xinanjiang_shape_parameter: float = Field(gt=0.01, lt=10.0)
    temp_delta_jan_mean: float = Field(gt=0.0)
    temp_delta_feb_mean: float = Field(gt=0.0)
    temp_delta_mar_mean: float = Field(gt=0.0)
    temp_delta_apr_mean: float = Field(gt=0.0)
    temp_delta_may_mean: float = Field(gt=0.0)
    temp_delta_jun_mean: float = Field(gt=0.0)
    temp_delta_jul_mean: float = Field(gt=0.0)
    temp_delta_aug_mean: float = Field(gt=0.0)
    temp_delta_sep_mean: float = Field(gt=0.0)
    temp_delta_oct_mean: float = Field(gt=0.0)
    temp_delta_nov_mean: float = Field(gt=0.0)
    temp_delta_dec_mean: float = Field(gt=0.0)
    lat: float = Field(gt=24, lt=55)
    lon: float = Field(gt=-125, lt=-66)
    glacier_percent: float = Field(ge=0, le=1)
    cgw: float = Field(gt=1.80e-06, lt=0.0018)
    expon: float = Field(gt=1.0, lt=8.0)
    max_gw_storage: float = Field(gt=0.01, lt=0.25)


class Flowpaths(BaseModel):
    """Pydantic class containing the data type, range (if known) of the flowpath attributes"""

    fp_id: int
    dn_nex_id: int
    up_nex_id: float
    div_id: int
    vpu_id: str
    length_km: float = Field(gt=0.0)
    area_sqkm: float = Field(gt=0.0)
    total_da_sqkm: float = Field(gt=0.0)
    mainstem_lp: int
    path_length: float = Field(gt=0.0)
    dn_hydroseq: int
    hydroseq: int
    stream_order: int
    mean_elevation: float = Field(gt=-86.0, lt=4422.0)
    slope: float = Field(gt=0.0, lt=90.0)
    n: float
    r: float
    y: float
    ncc: float
    btmwdth: float
    chslp: float
    musx: float
    musk: int
    topwdth: float
    topwdthcc: float
    topwdthcc_ml: float
    topwdth_ml: float
    y_ml: float
    r_ml: float


def validate_divides(gpkg_path_filename: str) -> None:
    """Validate the divides layer using Pydantic and check for NaNs

    Parameters
    ----------
    gpkg_path_filename : str
        full path and filename of the NHF geopackage

    Returns
    -------
    None
    """
    divides = gpd.read_file(gpkg_path_filename, layer="divides")
    divides = pd.DataFrame(divides)

    print(f"Total number of rows: {len(divides)}\n")

    rows_with_nan = divides[divides.isna().any(axis=1)]
    print(f"Total number of rows with NaNs: {len(rows_with_nan)}\n")

    nan_counts = divides.isna().sum().to_dict()
    print("Number of NaNs per attribute")
    for key, value in nan_counts.items():
        print(f"{key}: {value}")
    print("\n")

    print("Number of NaNs by VPU")
    nan_by_vpu = divides.isna().groupby(divides['vpu_id']).sum()
    with pd.option_context('display.max_columns', None):
       print(nan_by_vpu)
    print("\n")


    validator = Pandantic(schema=Divides)

    try:
        validator.validate(dataframe=divides, errors="raise")
    except ValidationError as e:
        error_details = e.errors()
        for error in error_details:
            print(f"{error['loc']}: {error['msg']}; value is {error['input']} ")


def validate_flowpaths(gpkg_path_filename: str) -> None:
    """Validate the flowpath layer using Pydantic and check for NaNs

    Parameters
    ----------
    gpkg_path_filename : str
        full path and filename of the NHF geopackage

    Returns
    -------
    None
    """
    flowpaths = gpd.read_file(gpkg_path_filename, layer="flowpaths")
    flowpaths = pd.DataFrame(flowpaths)

    rows_with_nan = flowpaths[flowpaths.isna().any(axis=1)]
    print(f"Total number of rows with NaNs: {len(rows_with_nan)}")

    nan_counts = flowpaths.isna().sum().to_dict()
    print("Number of NaNs per attribute")
    for key, value in nan_counts.items():
        print(f"{key}: {value}")

    validator = Pandantic(schema=Flowpaths)

    try:
        validator.validate(dataframe=flowpaths, errors="raise")
    except ValidationError as e:
        error_details = e.errors()
        for error in error_details:
            print(f"{error['loc']}: {error['msg']}; value is {error['input']} ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="validate NHF layers")

    parser.add_argument("--gpkg", type=str, required=True, help="Path and filename of the NHF geopackage")
    parser.add_argument(
        "--layer", type=str, required=True, help="NHF layer to validate: divides or flowpaths"
    )

    args = parser.parse_args()

    if args.layer == "divides":
        validate_divides(gpkg_path_filename=args.gpkg)
    elif args.layer == "flowpaths":
        validate_flowpaths(gpkg_path_filename=args.gpkg)
    else:
        print("layer must be divides or flowpaths")
