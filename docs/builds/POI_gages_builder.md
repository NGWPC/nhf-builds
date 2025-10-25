## POI/Gages Builder — Integration Guide

This document explains how to assemble a single, canonical gages layer by merging USGS (active + discontinued) and partner gage sources (TXDOT, CADWR, ENVCA, NWM calibration sets, AK/HI/PR supplements). You’ll download the source files from the S3 bucket (see below), place them in your local user directory.
### Download source files:
Download the entire set of source files from the project’s S3 bucket (Data account) can be found here:

s3://hydrofabric-data/POI_gage_sources/HF/

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

### example runs:
    python3 examples/poi_gages_builder.py --local-root /YOUR/LOCAL/HF

### Optional flags:
* Don’t overwrite existing geometries:

    `python3 examples/poi_gages_builder.py --local-root /YOUR/LOCAL/HF --no-update-existing`

* Change the excluded IDs:

`    python3 examples/poi_gages_builder.py --local-root /YOUR/LOCAL/HF --exclude-ids 15056210 15493000 99999999
`

## What you’ll get (final dataset)
* Live USGS gages: ~9,000
* Discontinued USGS gages: ~22,000 (some overlap with “live”)
* TXDOT gages: ~80
* CADWR gages: ~25
* ENVCA gages: ~27
* NWM calibration gages that were missing: ~68
* Alaska, Puerto Rico, Hawaii: ~30
* Total: 26,752 point features

## How it works (high-level):

The script examples/poi_gages_builder.py coordinates these steps:

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

### 6- Write final output

* Exports a single GeoPackage with the unified usgs_gages layer.
