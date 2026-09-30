# About the Data
## Schema

The following schema is the proposed data model for NGWPC hydrofabric datasets produced by this repo.

TODO: Update

<img style="display: block; margin-left: auto; margin-right: auto;" src="docs/img/nhf_v1.1.2_schema.png" alt="nhf_v1.1.2_schema.png" width="100%" height="100%"/>

## Flowpaths FACT Table

The central table (or FACT Table) is `Flowpaths`. Each `flowpath` has a downstream, and upstream `nexus` point, allowing for traversal of a river network through a single table. Additionally, there is a 1:1 relationship between `flowpath` and `divide`.

## NGEN Tables

The tables highlighted in green are the infomation needed for lumped modeling to take place. Lumped models require attributes, the shape of the `divide` that is being modeled, and a `nexus` point for flow to be aggregated to.

## Routing Tables

The tables highlighted in blue contain the information needed for routing at a high resolution. T-Route is expected to run at a fine-scale (~300m segments) with many `virtual_flowpaths`. Each virtual flowpath is delineated based on the reference fabric, and there should be a many -> one relationship between `virtual_flowpaths` and `flowpaths`, with some `virtual flowpaths` not being represented in the `flowpaths` table. These non-represented `flowpaths` have the parameter of `routing_segment` set to False, and will have flow estimated through flow-scaling.

The `reservoir_da` table encodes crosswalks between lakes and gages with an assigned data assimilation code. The `lakes_polygons` layer mirrors the traditional `lakes` point layer, but includes the polygon representation. This polygon representation is used to derive the flowpaths associated with lakes for routing. The `lake_vfp_crosswalk` table contains the intersection of lake polygons and virtual flowpaths so that T-route treats all lake flowpaths as lakes rather than channels.

## Reference Crosswalks

The NGWPC Hydrofabric is built using many reference materials:
- Reference Flowpaths
- Reference Reservoirs
- Reference Waterbodies
- NWM v3 Lakes
- National Inventory of Dams
- USGS/ENVCA/CADWR/TXDOT/RFC/USBR/USACE Streamflow Gages
- NHD+

To ensure `flowpaths` can be mapped to back to the materials that created them, each of the reference materials is mapped to `flowpaths`, `hydrolocations`, and `virtual flowpaths`. The following IDs pairings are used:

- Reference Flowpaths -> `ref_fp_id`
- Reference Reservoirs -> `dam_id`
- Reference Reservoirs -> `ref_fab_wb` is `lake_id` / NHD `COMID`
- Streamflow Gages -> `site_no`
- NHD+ -> `nhd_feature_id`

## Validation
The `validate_hf` task in the pipeline produces a JSON report called `nhf_{version}_validation.json`. This report details various metrics from the built product, such as: number of null divide attributes, number of attributes out of defined minimum and maxium range, and assertions that necessary lakes and gages are present and assigned to flowpaths.
