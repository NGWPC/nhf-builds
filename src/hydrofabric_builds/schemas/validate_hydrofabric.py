from pydantic import BaseModel, Field


class Divides(BaseModel):
    """Pydantic class containing the data type, range (if known) of the divide attributes"""

    div_id: int = Field(gt=0, description="unique divide identifier")
    vpu_id: str = Field(description="Vector Processing Unit Identifier")
    type: str = Field(description="Divide Type (one of independent, aggregate, connectors)")
    area_sqkm: float = Field(gt=0.0, description="catchment area in square kilometers")
    bexp_mode: float = Field(
        gt=2.0, lt=15.0, description="beta exponent on Clapp-Hornberger (1978) soil water relationship"
    )
    isltyp_mode: float = Field(ge=1, le=16, description="dominant soil type catagory")
    ivgtyp_mode: float = Field(ge=1, le=16, description="domainant vegetation type category")
    dksat_geomean: float = Field(gt=1.95e-07, lt=1.41e-03, description="saturated hydraulic conductivity")
    psisat_geomean: float = Field(gt=0.036, lt=0.955, description="saturated capillary head")
    cwpvt_mean: float = Field(gt=0.09, lt=0.36, description="empirical wind canopy parameter")
    mp_mean: float = Field(gt=3.6, lt=12.6, description="slope of conductance to photosynthesis relationship")
    mfsno_mean: float = Field(gt=0.5, lt=4.0, description="melt factor for snow depletion curve")
    quartz_mean: float = Field(gt=0.0, lt=1.0, description="mean soil quartz content")
    refkdt_mean: float = Field(gt=0.1, lt=4.0, description="reference soil infiltration parameter")
    slope1km_mean: float = Field(
        gt=0.0, lt=1.0, description="Modifies the gradient of the hydraulic head at the soil bottom"
    )
    smcmax_mean: float = Field(gt=0.16, lt=0.9, description="saturated soil moisture content")
    smcwlt_mean: float = Field(gt=0.05, lt=0.3, description="wilting point soil moisture content")
    vcmx_mean: float = Field(
        gt=24.0, lt=112.0, description="Modifies the gradient of the hydraulic head at the soil bottom"
    )
    imperv_mean: float = Field(
        gt=0.0, lt=1.0, description="Modifies the gradient of the hydraulic head at the soil bottom"
    )
    twi_q25: float = Field(description="Topographic wetness index 1st quartile")
    twi_q50: float = Field(description="Topographic wetness index 2nd quartile")
    twi_q75: float = Field(description="Topographic wetness index 3rd quartile")
    twi_q100: float = Field(description="Topographic wetness index 4th quartile")
    twi_q10: float = Field(description="Topographic wetness index 10th percentile")
    twi_q20: float = Field(description="Topographic wetness index 20th percentile")
    twi_q30: float = Field(description="Topographic wetness index 30th percentile")
    twi_q40: float = Field(description="Topographic wetness index 40th percentile")
    twi_q60: float = Field(description="Topographic wetness index 60th percentile")
    twi_q70: float = Field(description="Topographic wetness index 70th percentile")
    twi_q80: float = Field(description="Topographic wetness index 80th percentile")
    twi_q90: float = Field(description="Topographic wetness index 10th percentile")
    elevation_mean: float = Field(gt=-86.0, lt=4422.0, description="terrain elevation")
    slope250m_mean: float = Field(gt=0.0, lt=90.0, description="terrain slope")
    aspect_circmean: float = Field(gt=0.0, lt=360.0, description="terrain aspect")
    lzfpm_mean: float = Field(gt=40.0, lt=600.0, description="Maximum lower zone free water mean (primary)")
    lzpk_mean: float = Field(
        gt=0.001, lt=0.015, description="Lower zone recession coefficient mean (primary)"
    )
    lztwm_mean: float = Field(gt=75.0, lt=300.0, description="Maximum lower zone tension water mean")
    rexp_mean: float = Field(gt=1.4, lt=3.5, description="Percolation equation exponent mean")
    uzk_mean: float = Field(gt=0.2, lt=0.5, description="Upper zone recession coefficient mean")
    zperc_mean: float = Field(gt=0.0, lt=360.0, description="Minimum percolation rate coefficient mean")
    lzfsm_mean: float = Field(
        gt=0.0, lt=360.0, description="Maximum lower zone free water mean (secondary aka supplemental)"
    )
    lzsk_mean: float = Field(
        gt=0.03, lt=0.2, description="Lower zone recession coefficient mean, (secondary aka supplemental)"
    )
    pfree_mean: float = Field(
        gt=0.0,
        lt=0.5,
        description="Fraction of water percolating from upper zone directly to lower zone free water storage (mean)",
    )
    uzfwm_mean: float = Field(gt=10.0, lt=100.0, description="Maximum upper zone free water mean")
    uztwm_mean: float = Field(gt=25.0, lt=125.0, description="Upper zone tension water maximum storage mean")
    mfmin_mean: float = Field(gt=0.01, lt=0.6, description="Minimum non-rain melt factor mean")
    mfmax_mean: float = Field(gt=0.0, lt=360.0, description="Maximum non-rain melt factor mean")
    uadj_mean: float = Field(gt=0.01, lt=0.2, description="Average wind function for rain on snow (mean)")
    a_xinanjiang_inflection_point_parameter: float = Field(
        gt=-0.5,
        lt=0.5,
        description="Inflection point parameter for the Xinanjiang runoff generation model configuration",
    )
    b_xinanjiang_shape_parameter: float = Field(
        gt=0.01,
        lt=10.0,
        description="Inflection point parameter for the Xinanjiang runoff generation model configuration",
    )
    x_xinanjiang_shape_parameter: float = Field(
        gt=0.01,
        lt=10.0,
        description="Main, exponential shape parameter for the Xinanjiang runoff generation model configuration",
    )
    temp_delta_jan_mean: float = Field(gt=0.0, description="mean temperature difference for January")
    temp_delta_feb_mean: float = Field(gt=0.0, description="mean temperature difference for February")
    temp_delta_mar_mean: float = Field(gt=0.0, description="mean temperature difference for March")
    temp_delta_apr_mean: float = Field(gt=0.0, description="mean temperature difference for April")
    temp_delta_may_mean: float = Field(gt=0.0, description="mean temperature difference for May")
    temp_delta_jun_mean: float = Field(gt=0.0, description="mean temperature difference for June")
    temp_delta_jul_mean: float = Field(gt=0.0, description="mean temperature difference for July")
    temp_delta_aug_mean: float = Field(gt=0.0, description="mean temperature difference for August")
    temp_delta_sep_mean: float = Field(gt=0.0, description="mean temperature difference for September")
    temp_delta_oct_mean: float = Field(gt=0.0, description="mean temperature difference for October")
    temp_delta_nov_mean: float = Field(gt=0.0, description="mean temperature difference for November")
    temp_delta_dec_mean: float = Field(gt=0.0, description="mean temperature difference for December")
    lat: float = Field(gt=24, lt=55, description="latitude of divide centroid")
    lon: float = Field(gt=-125, lt=-66, description="longitude of divide centroid")
    glacier_percent: float = Field(ge=0, le=1, description="percentage of glacier cover in the divide")
    cgw: float = Field(gt=1.80e-06, lt=0.0018, description="groundwater coefficient")
    expon: float = Field(gt=1.0, lt=8.0, description="groundwater exponent")
    max_gw_storage: float = Field(gt=0.01, lt=0.25, description="The total height of the baseflow bucket")


class Flowpaths(BaseModel):
    """Pydantic class containing the data type, range (if known) of the flowpath attributes"""

    fp_id: int = Field(description="unique flowpath identifier")
    dn_nex_id: int = Field(description="connected downstream nexus identifier")
    up_nex_id: float = Field(description="connected upstream nexus identifier")
    div_id: int = Field(description="unique divide identifier")
    vpu_id: str = Field(description="Vector Processing Unit Identifier")
    length_km: float = Field(gt=0.0, description="flowpath length in kilometers")
    area_sqkm: float = Field(gt=0.0, description="incremental area of divide in kilometers")
    total_da_sqkm: float = Field(gt=0.0, description="total upstream drainage area in kilometers")
    mainstem_lp: int = Field(description="associated upstream drainage area in square kilometers")
    path_length: float = Field(gt=0.0, description="downstream path length")
    dn_hydroseq: int = Field(description="downstream hydrologic sequence")
    hydroseq: int = Field(description="hydrologic sequence")
    stream_order: int = Field(description="stream order of mapped reference flowpath")
    mean_elevation: float = Field(gt=-86.0, lt=4422.0, description="terrain elevation")
    slope: float = Field(gt=0.0, lt=90.0, description="terrain slope")
    n: float = Field(description="Manning's in channel roughness")
    r: float = Field(description="hydrologic radius")
    y: float = Field(description="estimated depth associated with top width")
    ncc: float = Field(description="compound channel top width")
    btmwdth: float = Field(description="bottom width of channel")
    chslp: float = Field(description="channel side slope")
    musx: float = Field(description="Muskingum weighting factor")
    musk: int = Field(description="Muskingum routing time")
    topwdth: float = Field(description="top width")
    topwdthcc: float = Field(description="compound channel top width")
    topwdthcc_ml: float = Field(description="compound channel top width at maximum levee")
    topwdth_ml: float = Field(description="top width at maximum levee")
    y_ml: float = Field(description="estimated depth associated with top wideth at maximum levee")
    r_ml: float = Field(description="hydraulic radius at maximum levee")
