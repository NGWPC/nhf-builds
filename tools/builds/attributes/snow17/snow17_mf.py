import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import rasterio
import rioxarray
import xarray as xr
from rasterio.enums import Resampling


def melt_factors(
    data_dir: Path,
    irr_mar_file: Path,
    irr_dec_file: Path,
    irr_jun_file: Path,
    irr_mar_flat_file: Path,
    wind_file: Path,
    forest_file: Path,
) -> None:
    """Computes melt factors MFMAX and MFMIN

    Parameters
    ----------
    data_ir : Path
        Path directory containing input rasters
    irr_mar_file: Path
        Filename of March irradiance raster
    irr_dec_file: Path
        Filename of December irradiance raster
    irr_jun_file: Path
        Filename of June irradiance raster
    irr_mar_flat_file: Path
        Filename of March irradiance with no topography raster
    wind_file: Path
        Filename of wind raster
    forest_file: Path
        Filename of forest percentage raster

    Returns
    -------
    None

    """
    irr_mar = read_raster(file_name=Path.joinpath(data_dir, irr_mar_file))
    irr_march_flat = read_raster(file_name=Path.joinpath(data_dir, irr_mar_flat_file))
    irr_dec = read_raster(file_name=Path.joinpath(data_dir, irr_dec_file))
    irr_jun = read_raster(file_name=Path.joinpath(data_dir, irr_jun_file))
    forest = read_raster(file_name=Path.joinpath(data_dir, forest_file))
    wind = read_raster(file_name=Path.joinpath(data_dir, wind_file))
    width, height, transform, crs = raster_info(file_name=Path.joinpath(data_dir, irr_mar_file))

    irr_mar = np.squeeze(irr_mar)
    irr_dec = np.squeeze(irr_dec)
    irr_march_flat = np.squeeze(irr_march_flat)
    irr_jun = np.squeeze(irr_jun)
    forest = np.squeeze(forest)
    wind = np.squeeze(wind)

    # convert forest percent to decimal
    forest = forest / 100

    rdb = irr_mar / irr_march_flat
    r = irr_dec / irr_jun

    mfmax = ((1.03 * (1 - forest) * rdb) + 2.04 + (0.42 * wind)) / (2 * (r + 1))
    mfmax = np.squeeze(mfmax)
    mfmin = mfmax * r

    with rasterio.open(
        "mfmax.tif",
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=mfmax.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(mfmax, 1)

    with rasterio.open(
        "mfmin.tif",
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=mfmax.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(mfmin, 1)


def uadj(data_dir: Path, wind_file: Path, forest_file: Path) -> None:
    """Computes UADJ

    Parameters
    ----------
    data_ir : Path
        Path directory containing input rasters
    wind_file: Path
        Filename of wind raster
    forest_file: Path
        Filename of forest percentage raster

    Returns
    -------
    None

    """
    wind = read_raster(file_name=Path.joinpath(data_dir, wind_file))
    forest = read_raster(file_name=Path.joinpath(data_dir, forest_file))
    width, height, transform, crs = raster_info(file_name=Path.joinpath(data_dir, forest_file))

    exp_min = 0.1
    exp_max = 0.25
    wind_adj = wind * (1 / 10) ** (exp_min + (exp_max - exp_min) * forest)
    wind_adj_kmh = wind_adj * 3.6
    wind_travel = wind_adj_kmh * 6
    uadj = wind_travel * 0.002
    uadj = np.squeeze(uadj)

    with rasterio.open(
        "uadj.tif",
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=uadj.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(uadj, 1)


def combine_rasters(data_dir: Path, superconus_raster: Path, conus_raster: Path) -> None:
    """Combines the NWS CONUS raster and the superconus temporary raster

    Parameters
    ----------
    data_dir : Path
        Path to directory containing the superconus parameter raster and conus NWS raster
    superconus_raster: Path
        Filename of the superconus parameter raster
    conus_raster: list[float]
        Filename of the conus NWS paramter raster

    Returns
    -------
    None

    """
    # Read temporary superconus raster and NWM conus raster
    superconus = rioxarray.open_rasterio(os.path.join(data_dir, superconus_raster)).squeeze()
    conus = rioxarray.open_rasterio(os.path.join(data_dir, conus_raster)).squeeze()

    # resample superconus raster to the resolution of the NWS raster
    superconus = superconus.rio.reproject(
        superconus.rio.crs, resolution=conus.rio.resolution(), resampling=Resampling.bilinear
    )

    # Expand the extent of the conus raster to match that of the superconus raster
    conus_matched = conus.rio.reproject_match(superconus)
    conus_matched_data = conus_matched.to_numpy()
    superconus_data = superconus.to_numpy()
    no_data = conus.rio.nodata

    # Create a numpy array where pixels with in CONUS contain the NWS value and those outside contain
    # the superconus value.
    new_array = np.where(conus_matched_data != no_data, conus_matched_data, superconus_data)

    # create new raster
    new_raster = xr.DataArray(
        new_array, coords=superconus.coords, dims=superconus.dims, attrs=superconus.attrs
    )

    # create results directory and save raster
    Path(f"{data_dir}/combined").mkdir(parents=True, exist_ok=True)
    new_filename = Path(f"{data_dir}/combined/{conus_raster}")
    new_raster.rio.to_raster(new_filename, tiled=True, compress="deflate")

    # remove temporary file
    try:
        os.remove(os.path.join(data_dir, superconus_raster))
        print("Removed temporary file")
    except FileNotFoundError:
        print(f"Temporary {os.path.join(data_dir, superconus_raster)} file does not exist.")


def read_raster(file_name: Path) -> npt.NDArray[np.float32]:
    """Read raster

    Parameters
    ----------
    file_name : Path

    Returns
    -------
    Raster object

    """
    with rasterio.open(file_name) as src:
        return src.read()


def raster_info(file_name: Path) -> Any:
    """Read raster

    Parameters
    ----------
    file_name : Path

    Returns
    -------
    Raster

    """
    with rasterio.open(file_name) as src:
        return src.width, src.height, src.transform, src.crs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to create rasters for Snow17 MFMAX, MFMIN and UADJ"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to directory containing input rasters where the output rasters will be written.",
    )
    parser.add_argument(
        "--irr_march",
        type=str,
        help="Filename for March irradiance raster",
    )
    parser.add_argument(
        "--irr_march_flat", type=str, help="Filename for March irradiance raster with no topography"
    )
    parser.add_argument(
        "--irr_december",
        type=str,
        help="Filename for December irradiance raster",
    )
    parser.add_argument("--irr_june", type=str, help="Filename for June irradiance raster")
    parser.add_argument("--forest", type=str, help="Filename for forest percentage raster")
    parser.add_argument("--wind", type=str, help="Filename for wind raster")
    parser.add_argument("--conus_mfmax", type=str, help="Filename for conus mfmax raster")
    parser.add_argument("--conus_mfmin", type=str, help="Filename for conus mfmin raster")
    parser.add_argument("--conus_uadj", type=str, help="Filename for conus uadj raster")

    args = parser.parse_args()

    melt_factors(
        data_dir=Path(args.data_dir),
        irr_mar_file=Path(args.irr_march),
        irr_dec_file=Path(args.irr_december),
        irr_jun_file=Path(args.irr_june),
        irr_mar_flat_file=Path(args.irr_march_flat),
        wind_file=Path(args.wind),
        forest_file=Path(args.forest),
    )

    uadj(data_dir=Path(args.data_dir), wind_file=Path(args.wind), forest_file=Path(args.forest))

    combine_rasters(
        data_dir=Path(args.data_dir),
        superconus_raster=Path("mfmax.tif"),
        conus_raster=Path(args.conus_raster),
    )
    combine_rasters(
        data_dir=Path(args.data_dir),
        superconus_raster=Path("mfmin.tif"),
        conus_raster=Path(args.conus_raster),
    )
    combine_rasters(
        data_dir=Path(args.data_dir), superconus_raster=Path("uadj.tif"), conus_raster=Path(args.conus_raster)
    )
