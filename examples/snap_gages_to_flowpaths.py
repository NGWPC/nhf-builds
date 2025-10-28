from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
from pathlib import Path

import geopandas as gpd
import numpy as np

from hydrofabric_builds.hydrolocations.assign_fp_to_gage import choose_flowpath_for_gage

# ---- Config ----
AREA_CRS = "EPSG:6933"  # equal-area for basin area
WORK_CRS = "EPSG:5070"  # projection for buffers & intersections

# ---------------------------
# Parallel worker plumbing
# ---------------------------

_FLOWPATHS: gpd.GeoDataFrame  # populated in each worker


def _init_worker_flowpaths(flowpaths_path: str, layer: str = "reference_flowpaths") -> None:
    """Load & project flowpaths once per process; keep only essential columns."""
    global _FLOWPATHS
    fp = gpd.read_file(flowpaths_path, layer=layer)
    keep = [c for c in ("flowpath_id", "totdasqkm", "geometry") if c in fp.columns]
    fp = fp[keep].copy()
    if fp.crs is None:
        raise ValueError("Flowpaths must have a valid CRS.")
    if fp.crs.to_string().upper() != WORK_CRS:
        fp = fp.to_crs(WORK_CRS)
    _FLOWPATHS = fp


def _worker_choose(args: tuple[str, float, float, np.ndarray]) -> tuple[str, str | None, float, str]:
    """
    Prepare the inputs for the main assignment function

    Uses preloaded _FLOWPATHS and slices candidates by indices.

    :param args: includes site_no, lon, lat, cand_idx
    :return: site_no, sel_id, float(basin_info.area_km2), basin_info.source
    """
    site_no, lon, lat, cand_idx = args
    candidates = _FLOWPATHS.iloc[cand_idx] if cand_idx.size else _FLOWPATHS.iloc[0:0].copy()
    sel_id, basin_info, _ = choose_flowpath_for_gage(
        in_buf=candidates,
        site_no=site_no,
        lon=lon,
        lat=lat,
        flow_id_col="flowpath_id",
        area_col="totdasqkm",
        area_match_pct=0.15,
    )
    return site_no, sel_id, float(basin_info.area_km2), basin_info.source


# ---------------------------
# Assignment (serial or parallel)
# ---------------------------


def _prepare_tasks(
    gages: gpd.GeoDataFrame,
    flowpaths_path: Path,
    flowpaths_layer: str,
    buffer_m: float | int,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, list[tuple[str, float, float, np.ndarray]]]:
    """
    Vectorized preparation of candidate indices per gage, no Python loops.

    Returns
    -------
      - `out` (copy of gages, CRS unchanged),
      - `fwd` (flowpaths projected to WORK_CRS),
      - `tasks` (list of tuples: (site_no, lon, lat, cand_idx_array))
    """
    # load flowpaths and project
    flowpaths = gpd.read_file(flowpaths_path, layer=flowpaths_layer)
    if flowpaths.crs is None:
        raise ValueError("flowpaths GeoDataFrame must have a valid CRS.")
    to_work = WORK_CRS if flowpaths.crs.to_string().upper() != WORK_CRS else flowpaths.crs
    fwd = flowpaths.to_crs(to_work)

    # gages → lon/lat & buffer in WORK_CRS
    if gages.crs is None:
        raise ValueError("`gages` must have a CRS (expected EPSG:4326).")
    g4326 = gages if gages.crs.to_string().upper() == "EPSG:4326" else gages.to_crs("EPSG:4326")
    gages_w = g4326.to_crs(to_work).copy()
    gages_w["__gid__"] = np.arange(len(gages_w))
    gages_buf = gages_w.copy()
    gages_buf.set_geometry(gages_w.geometry.buffer(float(buffer_m)), inplace=True)

    # spatial join flowpaths ↔ buffers (vectorized)
    hits = gpd.sjoin(
        fwd.reset_index(names="__fp_row__"),
        gages_buf[["__gid__", "geometry"]],
        predicate="intersects",
        how="inner",
    ).rename(columns={"__gid__": "__gage_row__"})

    # collect candidate row indices per gage row
    cand_idx_series = (
        hits.groupby("__gage_row__")["__fp_row__"]
        .apply(lambda s: s.to_numpy())
        .reindex(range(len(gages_w)), fill_value=np.array([], dtype=int))
    )

    # assemble tasks: light tuples (site_no, lon, lat, cand_idx)
    tasks_df = g4326.assign(
        lon=g4326.geometry.x,
        lat=g4326.geometry.y,
        __gage_row__=np.arange(len(g4326)),
    ).merge(cand_idx_series.rename("cand_idx"), left_on="__gage_row__", right_index=True, how="left")[
        ["site_no", "lon", "lat", "cand_idx"]
    ]

    tasks: list[tuple[str, float, float, np.ndarray]] = list(tasks_df.itertuples(index=False, name=None))

    out = g4326.copy()
    if "flowpath_id" not in out.columns:
        out["flowpath_id"] = None
    out["basin_km2"] = None
    out["basin_source"] = None

    return out, fwd, tasks


def run_assignment(
    gages: gpd.GeoDataFrame,
    flowpaths_path: Path,
    *,
    flowpaths_layer: str = "reference_flowpaths",
    buffer_m: float | int = 500.0,
    parallel: bool = False,
    max_workers: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Assign gages to flowpaths by area match on intersecting candidates.

    - If `parallel=False`: uses a single for-loop over tasks.
    - If `parallel=True` : uses ProcessPoolExecutor with per-process flowpath init.

    Returns a copy of `gages` with 'flowpath_id', 'basin_km2', 'basin_source'.
    """
    out, fwd, tasks = _prepare_tasks(gages, flowpaths_path, flowpaths_layer, buffer_m)

    if not parallel:
        # ---- SERIAL: single for-loop over tasks ----
        for count, (site_no, lon, lat, cand_idx) in enumerate(tasks):
            candidates = fwd.iloc[cand_idx] if cand_idx.size else fwd.iloc[0:0].copy()
            sel_id, basin_info, _ = choose_flowpath_for_gage(
                in_buf=candidates,
                site_no=site_no,
                lon=lon,
                lat=lat,
                flow_id_col="flowpath_id",
                area_col="totdasqkm",
                area_match_pct=0.15,
            )
            mask = out["site_no"].astype(str) == site_no
            if sel_id is not None:
                out.loc[mask, "flowpath_id"] = sel_id
            out.loc[mask, "basin_km2"] = float(basin_info.area_km2)
            out.loc[mask, "basin_source"] = basin_info.source
            print(count)
        return out

    # ---- PARALLEL: per-process init loads flowpaths once ----
    max_workers = max_workers or max(1, os.cpu_count() or 1)
    with cf.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker_flowpaths,
        initargs=(str(flowpaths_path), flowpaths_layer),
    ) as ex:
        for site_no, sel_id, area_km2, source in ex.map(_worker_choose, tasks, chunksize=100):
            mask = out["site_no"].astype(str) == site_no
            if sel_id is not None:
                out.loc[mask, "flowpath_id"] = sel_id
            out.loc[mask, "basin_km2"] = area_km2
            out.loc[mask, "basin_source"] = source

    return out


if __name__ == "__main__":
    """
    Running with argeparse

    ### Serial Run Example: ###
    python3 examples/snap_gages_to_flowpaths.py \
  --flowpaths /home/farshid.rahmani/Documents/Dataset/HF/HF_beta_v2.3/sc_reference_fabric.gpkg \
  --gages /home/farshid.rahmani/Documents/Dataset/HF/usgs_gages_all_conus_AK_Pr.gpkg \
  --buffer-m 500

    ### Parallel Run Example: ###
    python3 examples/snap_gages_to_flowpaths.py \
  --flowpaths /home/farshid.rahmani/Documents/Dataset/HF/HF_beta_v2.3/sc_reference_fabric.gpkg \
  --gages /home/farshid.rahmani/Documents/Dataset/HF/usgs_gages_all_conus_AK_Pr.gpkg \
  --parallel --max-workers 2

    """
    parser = argparse.ArgumentParser(
        description="Assign USGS gages (points) to flowpaths using basin area matching."
    )
    parser.add_argument(
        "--flowpaths",
        required=True,
        help="Path to the flowpaths GeoPackage (e.g., sc_reference_fabric.gpkg).",
    )
    parser.add_argument(
        "--flowpaths-layer",
        default="reference_flowpaths",
        help="Layer name in the flowpaths GeoPackage (default: reference_flowpaths).",
    )
    parser.add_argument(
        "--gages",
        required=True,
        help="Path to the gages GeoPackage (e.g., usgs_gages_all_conus_AK_Pr.gpkg).",
    )
    parser.add_argument(
        "--gages-layer",
        default="usgs_gages",
        help="Layer name in the gages GeoPackage (default: usgs_gages).",
    )
    parser.add_argument(
        "--buffer-m",
        type=float,
        default=500.0,
        help="Search buffer radius in meters around each gage (default: 500).",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing (ProcessPoolExecutor).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max worker processes when --parallel is set (default: all CPUs).",
    )
    parser.add_argument(
        "--output",
        help=(
            "Output GeoPackage path. "
            "Default: <directory of --gages>/usgs_gages_all_conus_AK_Pr_flowpath_id.gpkg"
        ),
    )

    args = parser.parse_args()

    flowpaths_path = Path(args.flowpaths)
    gages_path = Path(args.gages)

    # Default output next to the gages file if not provided
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = gages_path.parent / "usgs_gages_all_conus_AK_Pr_flowpath_id.gpkg"

    # Load gages (filter out rows without geometry)
    gages = gpd.read_file(gages_path, layer=args.gages_layer)
    gages = gages[gages.geometry.notna()].copy()

    # Run assignment (serial or parallel based on flag)
    assigned = run_assignment(
        gages=gages,
        flowpaths_path=flowpaths_path,
        flowpaths_layer=args.flowpaths_layer,
        buffer_m=float(args.buffer_m),
        parallel=bool(args.parallel),  # decides to run in serial or parallel
        max_workers=(args.max_workers if args.parallel else None),
    )

    # Save result
    assigned.to_file(output_path, layer=args.gages_layer, driver="GPKG")
    print(f"[ok] wrote {output_path}")

    """Running with no argeparse. Good for debugging"""
    # flowpaths_path = Path(r"/home/farshid.rahmani/Documents/Dataset/HF/HF_beta_v2.3/sc_reference_fabric.gpkg")
    # gages_path = Path(r"/home/farshid.rahmani/Documents/Dataset/HF/usgs_gages_all_conus_AK_Pr.gpkg")
    #
    # gages = gpd.read_file(gages_path, layer="usgs_gages")
    # gages = gages[gages.geometry.notna()].copy()
    #
    # # Serial
    # assigned = run_assignment(
    #     gages=gages,
    #     flowpaths_path=flowpaths_path,
    #     flowpaths_layer="reference_flowpaths",
    #     buffer_m=500.0,
    #     parallel=False,  ### serial or parallel
    #     max_workers=None,  ### None: if serial
    # )

    # assigned.to_file(
    #     gages_path.parent / "usgs_gages_all_conus_AK_Pr_flowpath_id.gpkg",
    #     layer="usgs_gages",
    #     driver="GPKG",
    # )
    print("done.")
