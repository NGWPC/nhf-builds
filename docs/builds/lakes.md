# Lakes vs Waterbodies

Separate "lakes" and "waterbodies" layers are currently retained to reflect separate workstreams. These workstreams will be fused in future work for a single "lakes" layer.

## Lakes
The "lakes" layer is built to retain all active NWM lakes. These are derived from `nwm_lakes.gpkg` polygons and Hydrofabric v2.2 attributes. Lakes are spatially placed at the most downstream nexus intersecting the waterbody. Attributes are retained from Hydrofabric 2.2 when available. If not available, defaults are used.

## Waterbodies
The "waterbodies" layer includes both active NWM lakes and potential RFC-DA candidates based on the "reference reservoirs" dataset. The waterbodies layer may not include all NWM lakes. Spatial placement may differ.

## Future Work
The workstreams will be fused to create one lakes layer including all active NWM lakes and additional RFC-DA candidates identified in waterbodies. New data collected in waterbodies will improve NWM lake attributes where available.
