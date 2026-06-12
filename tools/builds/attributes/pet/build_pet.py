import argparse
import os
import rioxarray
import numpy as np
import xarray as xr
from pathlib import Path

from tools.builds.attributes.schemas.attributes import VegetationTypes, VegetationTypesCombined




params = {"veg_height":(2.0, 10.0, 0.20),
          "zero_plane":(1.3, 7.0, 0.13),
          "momentum_transfer":(0.26, 1.3, 0.13),
          "heat_transfer":(0.03, 0.13, 0.003),
          "longwave_emissivity":(0.95, 0.7, 0.10),
          "shortwave_albedo":(0.40, 0.15, 0.25)
}


moderate = [VegetationTypes.DRYLAND_CROPLAND_AND_PASTURE.value,
            VegetationTypes.IRRIGATED_CROPLAND_AND_PASTURE.value,
            VegetationTypes.MIXED_DRYLAND_IRRIGATED_CROPLAND_AND_PASTURE.value,
            VegetationTypes.CROPLAND_GRASSLAND_MOSAIC.value,
            VegetationTypes.CROPLAND_WOODLAND_MOSAIC.value,
            VegetationTypes.GRASSLAND.value,
            VegetationTypes.SHRUBLAND.value,
            VegetationTypes.MIXED_SHRUBLAND_GRASSLAND.value,
            VegetationTypes.SAVANNA,
            VegetationTypes.HERBACEOUS_WETLAND.value,
            VegetationTypes.WOODED_WETLAND.value
]

forested = [VegetationTypes.DECIDUOUS_BROADLEAF_FOREST.value,
            VegetationTypes.DECIDUOUS_NEEDLELEAF_FOREST.value,
            VegetationTypes.EVERGREEN_BROADLEAF_FOREST.value,
            VegetationTypes.EVERGREEN_NEEDLELEAF_FOREST.value,
            VegetationTypes.MIXED_FOREST.value]

sparse = [VegetationTypes.WOODED_TUNDRA.value,
          VegetationTypes.HERBACEOUS_TUNDRA.value,
          VegetationTypes.MIXED_TUNDRA.value]

no_veg = [VegetationTypes.WATER_BODIES.value,
          VegetationTypes.SNOW_OR_ICE.value,
          VegetationTypes.PLAYA.value,
          VegetationTypes.LAVA.value,
          VegetationTypes.URBAN_AND_BUILT_UP_LAND.value,
          VegetationTypes.BARREN_OR_SPARSELY_VEGETATED.value,
          VegetationTypes.BARE_GROUND_TUNDRA.value]
print(VegetationTypes.BARE_GROUND_TUNDRA.value)


def built_pet(ivgtyp_path: Path, ivgtyp_file: Path) -> None:

    ivgtyp_raster = rioxarray.open_rasterio(os.path.join(ivgtyp_path, ivgtyp_file))

    if np.isnan(ivgtyp_raster.rio.nodata):
        ivgtyp_raster = ivgtyp_raster.rio.write_nodata(-9999)
    ivgtyp_raster = ivgtyp_raster.fillna(ivgtyp_raster.rio.nodata)
    ivgtyp_raster = ivgtyp_raster.astype(np.int32)
    grouped = xr.apply_ufunc(group_types, ivgtyp_raster, vectorize=True)
    grouped = grouped.fillna(ivgtyp_raster.rio.nodata)
    grouped = grouped.astype(np.int32)

    for key, value in params.items():
        new_array =  grouped.where(grouped != 1, value[0]) \
                            .where(grouped != 2, value[1]) \
                            .where(grouped != 3, value[2]) \
                            .where(grouped != 4, ivgtyp_raster.rio.nodata)


        new_array = new_array.assign_coords(
                    x=ivgtyp_raster.x,
                    y=ivgtyp_raster.y,
                    band=ivgtyp_raster.band
                    )

        new_array = new_array.rio.write_crs(ivgtyp_raster.rio.crs)
        new_array.rio.write_transform(ivgtyp_raster.rio.transform(), inplace=True)
        new_array.attrs = ivgtyp_raster.attrs
        new_array.rio.write_nodata = ivgtyp_raster.rio.nodata
        filename = os.path.join(ivgtyp_path, f"{key}.tif")
        new_array.rio.to_raster(filename, tiled=True, compress="deflate")

    new_array = grouped
    new_array = new_array.assign_coords(
                    x=ivgtyp_raster.x,
                    y=ivgtyp_raster.y,
                    band=ivgtyp_raster.band
                    )

    new_array = new_array.rio.write_crs(ivgtyp_raster.rio.crs)
    new_array.rio.write_transform(ivgtyp_raster.rio.transform(), inplace=True)
    new_array.attrs = ivgtyp_raster.attrs
    new_array.rio.write_nodata = ivgtyp_raster.rio.nodata
    filename = os.path.join(ivgtyp_path, "grouped.tif")
    new_array.rio.to_raster(filename, tiled=True, compress="deflate")

def group_types(x):
    if(x in moderate):
        return VegetationTypesCombined.MODERATE
    elif(x in forested):
        return VegetationTypesCombined.FOREST
    elif(x in sparse):
        return VegetationTypesCombined.SPARSE
    elif(x in no_veg):
        return VegetationTypesCombined.NA

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script create PET parameters based on vegetation type")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to directory containing the input vegetation type raster and where the output rasters will be written.",
    )
    parser.add_argument("--global_raster_file", type=str, help="filename for global sac-sma raster")
    parser.add_argument(
        "--veg_type_file",
        type=str,
        help="the vegetation type raster filename, e.g., IVGTYP.tif",
    )

    args = parser.parse_args()


built_pet(ivgtyp_path=args.data_dir, ivgtyp_file=args.veg_type_file)