# Lakes

## Explanation

## Data sources
### Inputs
s3: `s3://edfs-data/lakes/input`
local: `data/lakes/input`

#### Lake polygons:
OWP sent a file called `nwm_lakes.gpkg` in January 2026. This polygon file was split into `nwm_lakes_hi_input.gpkg`, `nwm_lakes_prvi_input.gpkg`, and `nwm_lakes_sconus_input.gpkg`. There are no AK polygons. Polygons are associated with most downstream flowpath intersecting.

#### Attribute data:
Attribute data from HF 2.2 is joined to retain all attributes and RFC-DA categories. Defaults are filled in for lakes not present in 2.2 (Canada, ~30 missing CONUS lakes)
- AK: `ak_lakeparm.gpkg` - lakeparm points from NWMv3
- HI: `hi_nextgenwork_around.gpkg` The final HF 2.2 Hawaii version used by NGWPC
- PRVI: `prvi_nextgen_workaround.gpkg` The final HF 2.2 PRVI version used by NGWPC
- SuperCONUS: `nwm_patch_conus_nextgen.gpg` The final HF 2.2 CONUS version used by NGWPC

### Outputs:
s3: `s3://edfs-data/lakes/output`
local: `data/lakes/output`
The output of the flowpath-associated and merged file is saved when flowpath_association is run. It can also be loaded a priori to skip running flowpath association.

- AK: `ak_lakes_fp_associated.gpkg`
- HI: `hawaii_lakes_fp_associated.gpkg`
- PRVI: `prvi_lakes_fp_associated.gpkg`
- SuperCONUS: `sconus_lakes_fp_associated.gpkg`

These outputs are used to create the final `lakes` layer in the NHF gpkg.

## Adding a new data source
Data sources can be added to the lakes pipeline.
1. Create a pydantic model in hydrofabric_builds.hydrofabric.schemas. Follow the templates like NWMLakeInput, RefWaterbodyInput, ReferenceReservoirs Input
2. If new lakes are in reference waterbodies, add them to adhoc_lakes.gpkg and flag as true. These will be picked up by the Reference Waterbody step
3. New polygon sources may need to be associated with flowpaths. The `polygon_outlet` flowpath association method requires the fields:
- path
- layer
- tmp_path - will save out associated flowpaths
- run - Flag to run input. Must be set to false if file is not present.
- id_field - ID field in input
- output_id_field - The ID field will be changed to this. Leave the same if it will not change. For example, some datasets refer to "comid" as different valus - "comid", "lake_id", and "newId". These will all be changed to the same output.
- associate_flowpaths - Flag to run flowpath association
- flowpath_association_method - Type of flowpath association. Options are `polygon_outlet` or `nearest_point`
- search_radius_m - Distance from flowpath to search when associating flowpaths
- min_preferred_intersection_len_m - Minimum prefered intersection lenght when associating polygons with flowpaths. If the flowpath intersection is extremely short, it can sometimes be almost entirely on a long downstream flowpath.
- attrib_src_path (If none, set it to null but keep field. If using an attribute source, also add attrib_src_layer, attrib_src_key)
- fields - Optional. If you need to retain fields from attributes.
4. Add the new config to LakesConfig.[name] and set the default to the instantiate the class.
5. Add file paths to LakesConfig.inject_dirs. If attrib path is optional, make it optional.
6. Add functions related to processing the code to hydrofabric_builds.lakes.lakes.
7. Add a function to handle elevation called `_calcuate_elevation__[name]`. Look at other elevation functions to determine how elevation should be calculated.
You will need to fill out `ref_elev` and `dam_elev`. `ref_elev` is a normal pool proxy. It is the mean of the lake polygon. `dam_elev` is the elevation of the point outlet/dam.
8. In hydrofabric_builds.lakes.lakes_pipeline, add a step before `concat all lakes`. The step should check if `run` is set to true. It should append the completed geodataframe to the geodataframe list. This list will pick up the geodataframe when lakes are concatenated.
