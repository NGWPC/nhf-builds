from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from hydrofabric_builds.hydrolocations.TXDOT_gages_builder import txdot_read_file
from hydrofabric_builds.hydrolocations.usgs_gages_builder import (
    add_missing_usgs_sites,
    build_usgs_gages_from_kmz,
    merge_gage_xy_into_gages,
    merge_minimal_gages,
    merge_usgs_shapefile_into_gages,
)


def run_poi_builder(
    local_root: Path,
    *,
    update_existing: bool = True,
    exclude_ids: tuple[str, ...] | None = ("15056210", "15493000"),
    output_filename: str = "usgs_gages_all_conus_AK_Pr.gpkg",
) -> gpd.GeoDataFrame:
    """
    Build the unified `gages` GeoDataFrame from local sources.

    Parameters
    ----------
    local_root : Path
        Root directory that contains the `gauge_xy/` subfolders and files.
    update_existing : bool, default True
        If True, geometry (and select name fields where supported) is overwritten for
        matching `site_no`s when a source provides higher-fidelity data.
    exclude_ids : tuple[str,...] | None
        Site numbers to exclude when ingesting generic XY CSV (e.g., AK sites outside domain).
        Set to None to disable.
    output_filename : str, default "usgs_gages_all_conus_AK_Pr.gpkg"
        Name of the output GeoPackage written under `local_root`.

    Returns
    -------
    GeoDataFrame
        The merged `gages` GeoDataFrame that was written to disk.
    """
    # ---------------------------------------------------------------------
    # 1) USGS discontinued (KMZ)
    # ---------------------------------------------------------------------
    """
    State gage file with kmz format can be downloaded from the following USGS link:
    https://waterwatch.usgs.gov/index.php?id=stategage

    choose "past flow/runoff"
    choose option "streamgage locations in KML"
    Then all files for all 50 states plus AK and PR can be downloaded. save them in a directory and point the
    following variable to that directory
    """
    usgs_folder = local_root / "gauge_xy" / "usgs_gages_discontinued"
    gages = build_usgs_gages_from_kmz(usgs_folder)  # scans all streamgages_*.kmz

    # ---------------------------------------------------------------------
    # 2) USGS live (SHP) — merge a set of known shapefiles
    # ---------------------------------------------------------------------
    usgs_active_main_dir = local_root / "gauge_xy" / "usgs_active_gages"
    shp_file_paths = [
        usgs_active_main_dir / "mv01dstx_shp" / "mv01dstx.shp",
        usgs_active_main_dir / "pa01dstx_shp" / "pa01dstx.shp",
        usgs_active_main_dir / "pa07dstx_shp" / "pa07dstx.shp",
        usgs_active_main_dir / "pa14dstx_shp" / "pa14dstx.shp",
        usgs_active_main_dir / "realstx_shp" / "realstx.shp",
    ]
    for shp_path in shp_file_paths:
        # Skip quietly if a listed file isn't present
        if not shp_path.exists():
            print(f"[warn] USGS active shapefile not found, skipping: {shp_path}")
            continue
        gages = merge_usgs_shapefile_into_gages(
            gages=gages,
            shp_path=shp_path,
            update_existing=update_existing,
        )

    # ---------------------------------------------------------------------
    # 3) TXDOT (RDB/TXT) — append/update minimal mapped fields
    # ---------------------------------------------------------------------
    """
    TXDOT_sites = ["08030530","08031005",
    "08031020","08041788","08041790","08041940","08041945","08041970","08042455","08042468","08042470","08042515",
    "08042539","08064990","08065080","08065310","08065340","08065420","08065700","08065820","08065925","08066087",
    "08066138","08066380","08067280","08067505","08067520","08067653","08068020","08068025","08070220","08070550",
    "08070900","08076990","08077110","08077640","08077670","08077888","08078400","08078890","08078910","08078935",
    "08097000","08098295","08100950","08102730","08108705","08108710","08109310","08110520","08111006","08111051",
    "08111056","08111070","08111080","08111085","08111090",'08111110',"08117375","08117403","08117857","08117858",
    "08162580","08163720","08163880","08163900","08164150","08164200","08164410","08167000","08169778","08173210",
    "08174545","08180990","08189298","08189320","08189520","08189585","08189590","08189718"]

    reading TXDOT sites from a .txt file downloaded from the following address.
    As of Oct 2025, it is not publicly available
    https://waterservices.usgs.gov/nwis/site/?format=rdb&siteStatus=all&sites=08030530,08031005,08031020,08041788,08041790,08041940,08041945,08041970,08042455,08042468,08042470,08042515,08042539,08064990,08065080,08065310,08065340,08065420,08065700,08065820,08065925,08066087,08066138,08066380,08067280,08067505,08067520,08067653,08068020,08068025,08070220,08070550,08070900,08076990,08077110,08077640,08077670,08077888,08078400,08078890,08078910,08078935,08097000,08098295,08100950,08102730,08108705,08108710,08109310,08110520,08111006,08111051,08111056,08111070,08111080,08111085,08111090,08111110,08117375,08117403,08117857,08117858,08162580,08163720,08163880,08163900,08164150,08164200,08164410,08167000,08169778,08173210,08174545,08180990,08189298,08189320,08189520,08189585,08189590,08189718
    """
    txdot_path = local_root / "gauge_xy" / "TXDOT_gages" / "TXDOT_gages.txt"
    if txdot_path.exists():
        gdf_TXDOT_gages = txdot_read_file(path=txdot_path)
        gages = merge_minimal_gages(
            gages=gages,
            source=gdf_TXDOT_gages,
            update_existing=update_existing,
        )
    else:
        print(f"[warn] TXDOT file not found, skipping: {txdot_path}")

    # ---------------------------------------------------------------------
    # 4) CADWR/ENVCA/AK/HI/PR & misc. XY CSVs
    # ---------------------------------------------------------------------
    gages_xy_path = local_root / "gauge_xy" / "gage_xy.csv"
    if gages_xy_path.exists():
        gages = merge_gage_xy_into_gages(
            gages=gages,
            gage_xy_csv=gages_xy_path,
            update_existing=update_existing,
            exclude_ids=exclude_ids,
            fill_value="-",
        )
    else:
        print(f"[warn] gage_xy.csv not found, skipping: {gages_xy_path}")

    # ---------------------------------------------------------------------
    # 5) NWM calibration gages — ensure presence; fill missing via NWIS Site Service
    # ---------------------------------------------------------------------
    usgs_cal_gages_path = local_root / "gauge_xy" / "all_gages_gpkgs" / "nwm_calib_gages.txt"
    if usgs_cal_gages_path.exists():
        usgs_cal_gages = pd.read_csv(usgs_cal_gages_path, sep="\t", header=None, dtype=str)
        usgs_cal_gages.columns = ["site_no"]
        missed_gages = usgs_cal_gages.loc[
            ~usgs_cal_gages["site_no"].isin(gages["site_no"].astype(str).unique()), "site_no"
        ].tolist()

        if missed_gages:
            print(f"[info] Calibration gages missing ({len(missed_gages)}); attempting NWIS fetch...")
            gages_updated, usgs_ids, non_usgs, fetched_df = add_missing_usgs_sites(gages, missed_gages)
            print(f"[info] USGS-style IDs fetched: {len(usgs_ids)}; non-USGS IDs: {len(non_usgs)}")
            if non_usgs:
                print(f"[info] Non-USGS examples (not fetched): {non_usgs[:10]}")
            print(f"[info] Added rows: {len(gages_updated) - len(gages)}")
            gages = gages_updated
    else:
        print(f"[warn] NWM calibration list not found, skipping: {usgs_cal_gages_path}")

    # ---------------------------------------------------------------------
    # 6) Write final output and return
    # ---------------------------------------------------------------------
    output = local_root / output_filename
    gages.to_file(output, layer="usgs_gages", driver="GPKG")
    print(f"[ok] Wrote {output}")
    return gages


if __name__ == "__main__":
    """
    Usage:
        python3 examples/poi_gages_builder.py

    Before running, download source files from the S3 bucket into your local user directory,
    e.g. /home/<you>/Documents/Dataset/HF/, preserving the expected subfolder structure:

        HF/
         └─ gauge_xy/
             ├─ usgs_gages_discontinued/      # USGS KMZ bundles
             ├─ usgs_active_gages/            # USGS active shapefiles
             ├─ TXDOT_gages/TXDOT_gages.txt
             ├─ gage_xy.csv                   # CADWR/ENVCA/AK/HI/PR etc.
             └─ all_gages_gpkgs/nwm_calib_gages.txt

    You can change `local-root` below to your path.

    example runs:
        python3 examples/poi_gages_builder.py --local-root /YOUR/LOCAL/HF

    Optional flags:
        Don’t overwrite existing geometries:
            python3 examples/poi_gages_builder.py --local-root /YOUR/LOCAL/HF --no-update-existing

        Change the excluded IDs:
            python3 examples/poi_gages_builder.py --local-root /YOUR/LOCAL/HF --exclude-ids 15056210 15493000 99999999

    """

    parser = argparse.ArgumentParser(description="Build unified gages GeoPackage.")
    parser.add_argument(
        "--local-root",
        required=True,
        help="Path to the local user directory that contains gauge_xy/…",
    )
    parser.add_argument(
        "--no-update-existing",
        action="store_true",
        help="Do not overwrite geometry/name for existing site_no rows.",
    )
    parser.add_argument(
        "--exclude-ids",
        nargs="*",
        default=None,  # ("15056210", "15493000"),
        help="Site numbers to exclude when ingesting the generic XY CSV.",
    )
    args = parser.parse_args()

    run_poi_builder(
        local_root=Path(args.local_root),
        update_existing=not args.no_update_existing,
        exclude_ids=tuple(args.exclude_ids) if args.exclude_ids else None,
    )

    ## if you want to run the code with no parser:
    # local_user_main_dir = Path("/home/farshid.rahmani/Documents/Dataset/HF")
    # run_poi_builder(
    #     local_root=local_user_main_dir,
    #     update_existing=True,
    #     exclude_ids=None,  # Alaska gages that are out of domain : ("15056210", "15493000")
    # )
