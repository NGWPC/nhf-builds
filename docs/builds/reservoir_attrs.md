## Description:


Below is a summary of what the Reservoir Python script is doing and the dependencies it assumes. potential future action items for each data source has been descussed here for record.
This issue is mainly about capturing and solidifying the current workflow so we can (a) understand the data dependencies, and (b) reproduce the same processing in our Python-based toolchain.

### 1. NID (USACE National Inventory of Dams)


The script uses an archived CSV hosted here in S3 data account:
s3://hydrofabric-data/reservoirs/source_files/NID2019_U.csv

USACE  most up-to-date data can be downloaded from here:
`https://nid.sec.usace.army.mil/nid/#/downloads
`

and here in S3 data account:

`s3://hydrofabric-data/reservoirs/source_files/NID_2025_11_02.csv
s3://hydrofabric-data/reservoirs/source_files/NID_2025_11_02.gpkg`


### 2. Reference Reservoirs:
These come from OWP and are shared internally. The file is stored in:

`s3://hydrofabric-data/reservoirs/reference_reservoirs/reference-reservoirs-v1.gpkg
`
### 3. Reference waterbodies
The spatial geometries of waterbodies in the USA has been shared internally by OWP and is stored in:

`s3://hydrofabric-data/reservoirs/reference_reservoirs/reference_waterbodies.gpkg
`

### 4. OSM dams & waterbodies:
the CONUS source file being used here is stored in S3 data account:
s3://hydrofabric-data/reservoirs/source_files/osm_dams_all.gpkg

which has been collected from the original state-based dataset that can be downloaded using [osmextract R package](https://cran.r-project.org/web/packages/osmextract/vignettes/osmextract.html).

#### Steps performed on CONUS OSM data

- Reads all state .gpkg files, extracts the lines layer, filters waterway == 'dam'.
- Binds them into a single dataset and writes data/osm_dams_all.gpkg.

#### Future Action Items
- Download per state dataset.
- Document that building OSM data requires ~90GB disk and time.
-  Decide whether we:
(a) Expect users to prebuild osm_dams_all.gpkg themselves, or
(b) Provide a prebuilt OSM dams layer somewhere.

- Reproduce the “merge all states into one OSM dams layer” logic in Python (if we support a full pipeline). The code is available in here however it has not been directly tested by state dataset yet

#### Note:
it takes around 30 minutes to run and create "rc_da_hydraulics-v1.gpkg" file
