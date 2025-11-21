## POI/Gages Builder — Integration Guide

This document explains how to assemble a single, canonical gages layer by merging USGS (active + discontinued) and partner gage sources (TXDOT, CADWR, ENVCA, NWM calibration sets, AK/HI/PR supplements). You’ll download the source files from the S3 bucket (see below), place them in your local user directory.
### Download source files:
Download the entire set of source files from the project’s S3 bucket (Data account) can be found here:

s3://hydrofabric-data/POI_gage_sources/gages/

Before running, download source files from the S3 bucket into your local user directory,
e.g. /home/<you>/Documents/hydrofabric-builds/data/gages/, preserving the expected subfolder structure:

    data/
     └─ gages/
         ├─ usgs_gages_discontinued/      # USGS KMZ bundles
         ├─ usgs_active_gages/            # USGS active shapefiles
         ├─ TXDOT_gages/TXDOT_gages.txt
         ├─ gage_xy.csv                   # CADWR/ENVCA/AK/HI/PR etc.
         └─ all_gages_gpkgs/nwm_calib_gages.txt
         ├─ nldi_upstream_basins.gpkg     # USGS API, optional
`

## What you’ll get (final dataset)
* Live USGS gages: ~12,000
* Discontinued USGS gages: ~14,700 (some overlap with “live”)
* TXDOT gages: ~77
* CADWR gages: ~25
* ENVCA gages: ~27
* NWM calibration gages that were missing: ~68
* Alaska, Puerto Rico, Hawaii: ~30
* Total: 26,754 point features

## How it works (high-level):

The gages task in hydrofabric-builds coordinates these steps:

### 1- USGS discontinued (KMZ)

* build_usgs_gages_from_kmz() scans a folder of USGS KML-in-KMZ files and extracts points + site_no.

* Result becomes the initial gages GeoDataFrame.

### 2-USGS live (SHP)

* merge_usgs_shapefile_into_gages() reads several USGS shapefiles and adds missing sites.

* With update_existing=True, it overwrites geometry/name/state for matching site_no when the shapefile has better data.

### 3- TXDOT RDB/TXT

* txdot_read_file() parses the RDB-style text.

* merge_minimal_gages() maps: geometry→geometry, site_no→site_no, station_nm→name_raw, and fills everything else with "-".

* With update_existing=True, coordinates/names are refreshed for matches.

### 4- Generic XY CSV (CADWR, ENVCA, AK/HI/PR, misc.)

* merge_gage_xy_into_gages() maps gageid→site_no, (lon,lat)→geometry.

* Appends new sites and (optionally) updates geometry for existing ones.

* You can exclude IDs (e.g., two Alaska gages outside the domain) if desired.

### 5- NWM Calibration gages

* The list is checked against gages.

* Any missing USGS-style IDs are fetched from the NWIS Site Service via add_missing_usgs_sites() and appended.

* Non-USGS IDs (e.g., Canadian IDs with letters) are reported and skipped.

### 6) Finding upstream area for USGS gages using API

### 7) Assign NLDI basins column to gages

the upstream area are read from USGS API and are compared with total upstream area calculated in hydrofabric. It is a method to make sure the flowpaths are assigned cor rectly to gages.

### 8) Assign flowpath to gages

Here we assign the nearest flowpaths to gages, wherever the upstream areas were not matched.

## 9) drop the columns we don't need
`keep_cols = ["site_no", "geometry", "status", "USGS_basin_km2", "fp_id", "method_fp_to_gage"]
`
### 6- Write final output

* Exports a single GeoPackage with the unified usgs_gages layer.
